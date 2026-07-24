"""Record fixed-length or terminal Phase 3 recurrent-policy videos."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from robosoccer.phase3 import ACTION_NAMES, make_phase3_environment
from scripts.evaluate_phase3 import load_policy


def annotate(frame, actions, possession, profile, defender_style):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = " | ".join(
        [
            "actions "
            + ", ".join(
                agent.replace("attacker_", "A")
                + ":"
                + ACTION_NAMES[action]
                for agent, action in actions.items()
            ),
            "possession " + str(possession),
            "defender " + str(defender_style),
            "profile " + str(profile),
        ]
    )
    draw.rounded_rectangle(
        (18, image.height - 48, image.width - 18, image.height - 12),
        radius=8,
        fill=(9, 31, 25),
    )
    draw.text((30, image.height - 37), text, fill=(245, 247, 240))
    return np.asarray(image)


def manifest_record_key(record):
    return "|".join(
        str(record.get(name, ""))
        for name in [
            "checkpoint",
            "simulator",
            "scenario",
            "defender_style",
            "profile",
            "seed",
            "recording_mode",
        ]
    )


def merge_manifest_records(existing, additions):
    merged = {}
    order = []
    for record in [*(existing or []), *(additions or [])]:
        key = manifest_record_key(record)
        if key not in merged:
            order.append(key)
        merged[key] = record
    return [merged[key] for key in order]


def load_manifest(path):
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if not isinstance(records, list):
        raise ValueError("Phase 3 video manifest must contain a JSON list")
    return records


def record_phase3_videos(args):
    full_match = bool(args.until_terminal or args.full_match)
    if not full_match and not 15.0 <= args.seconds <= 30.0:
        raise ValueError("--seconds must be between 15 and 30 in fixed-duration mode")
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    config, actor, normalizer, checkpoint = load_policy(
        run_dir, args.checkpoint, device
    )
    env = make_phase3_environment(
        config,
        simulator=args.simulator,
        scenario=args.scenario,
        profile_name=args.profile,
        defender_style=args.defender_style,
        render_mode="rgb_array",
    )
    output_dir = run_dir / "videos" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = int(config["video"].get("fps", 20))
    repeats = max(
        1,
        round(
            fps
            * float(config["environment"]["dt"])
            * int(config["environment"]["macro_action_repeat"])
        ),
    )
    maximum_frames = None if full_match else int(args.seconds * fps)
    records = []
    hidden_size = int(config["phase3"]["recurrent"]["hidden_size"])
    agent_names = env.possible_agents
    observation_size = env.observation_dimension
    action_size = env.action_size
    seeds = (
        [int(args.seed)]
        if args.seed is not None
        else [int(args.seed_base) + episode for episode in range(args.episodes)]
    )
    try:
        for seed in seeds:
            observations, _ = env.reset(seed=seed)
            hidden = torch.zeros(1, len(agent_names), hidden_size, device=device)
            frames = []
            terminal_metrics = None
            clip_end_metrics = env.metrics_snapshot()
            while env.agents and (
                maximum_frames is None or len(frames) < maximum_frames
            ):
                raw = np.zeros(
                    (len(agent_names), observation_size), dtype=np.float32
                )
                masks = np.zeros(
                    (len(agent_names), action_size), dtype=np.float32
                )
                for index, agent in enumerate(agent_names):
                    if agent in env.active_agents:
                        raw[index] = observations[agent]
                        masks[index] = env.action_mask(agent)
                normalized = normalizer.normalize(
                    raw, config["observations"]["clip"]
                )
                with torch.no_grad():
                    logits, hidden = actor(
                        torch.as_tensor(normalized, device=device), hidden
                    )
                    logits = logits.masked_fill(
                        torch.as_tensor(masks, device=device) < 0.5, -1e9
                    )
                    selected = torch.argmax(logits, dim=-1).cpu().numpy()
                actions = {
                    agent: int(selected[index])
                    for index, agent in enumerate(agent_names)
                    if agent in env.active_agents
                }
                frame = annotate(
                    env.render(),
                    actions,
                    env.ball["possession"],
                    env.selected_profile,
                    env.defender_style,
                )
                copies = repeats
                if maximum_frames is not None:
                    copies = min(repeats, maximum_frames - len(frames))
                frames.extend([frame] * copies)
                observations, _, _, _, infos = env.step(actions)
                clip_end_metrics = env.metrics_snapshot()
                if not env.agents:
                    terminal_metrics = next(iter(infos.values()))["episode_metrics"]
            if not frames:
                raise RuntimeError("Phase 3 recorder produced no frames")
            if maximum_frames is not None:
                while len(frames) < maximum_frames:
                    frames.append(frames[-1])
            metrics = terminal_metrics or clip_end_metrics
            outcome = (
                "cooperative"
                if metrics.get("cooperative_success")
                else "goal"
                if metrics.get("success")
                else metrics.get("terminal_reason") or "partial"
            )
            recording_mode = "until_terminal" if full_match else "fixed_duration"
            path = output_dir / (
                f"{args.simulator}_{args.scenario}_{args.defender_style}_"
                f"seed{seed}_{recording_mode}_{outcome}.mp4"
            )
            imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=7)
            terminal = terminal_metrics is not None
            record = {
                "path": str(path.relative_to(run_dir)),
                "policy_run_id": run_dir.name,
                "seed": seed,
                "seed_category": args.seed_category,
                "outcome": outcome,
                "frames": len(frames),
                "seconds": len(frames) / fps,
                "checkpoint": str(checkpoint),
                "simulator": args.simulator,
                "scenario": args.scenario,
                "defender_style": args.defender_style,
                "actual_defender_style": env.defender_style,
                "profile": args.profile,
                "recording_mode": recording_mode,
                "terminal": terminal,
                "clip_end_reason": (
                    metrics.get("terminal_reason")
                    if terminal
                    else "video_time_limit"
                ),
                "terminal_reason": metrics.get("terminal_reason"),
                "terminal_metrics": terminal_metrics,
                "clip_end_metrics": clip_end_metrics,
                "metrics": metrics,
            }
            records.append(record)
    finally:
        env.close()
    manifest = output_dir / "video_manifest.json"
    merged = merge_manifest_records(load_manifest(manifest), records)
    manifest.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return records, manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument(
        "--simulator", choices=["abstract", "pymunk"], default="pymunk"
    )
    parser.add_argument("--scenario", default="phase3_2v2_pass_required")
    parser.add_argument("--profile", default="nominal")
    parser.add_argument(
        "--defender-style",
        choices=["lane_block", "predictive", "zonal", "press", "mixed"],
        default="mixed",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=340000)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--seed-category",
        choices=["training", "validation", "audit", "gate", "video", "evaluation"],
        default="video",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--until-terminal", action="store_true")
    parser.add_argument("--full-match", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    records, manifest = record_phase3_videos(build_parser().parse_args())
    print(json.dumps(records, indent=2))
    print(f"Merged video manifest: {manifest}")


if __name__ == "__main__":
    main()
