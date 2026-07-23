"""Record matched 15--30 second Phase 3 recurrent-policy videos."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from robosoccer.phase3 import ACTION_NAMES, make_phase3_environment
from scripts.evaluate_phase3 import load_policy


def annotate(frame, actions, possession, profile):
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
            "profile " + str(profile),
        ]
    )
    draw.rounded_rectangle((18, image.height - 48, image.width - 18, image.height - 12), radius=8, fill=(9, 31, 25))
    draw.text((30, image.height - 37), text, fill=(245, 247, 240))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--simulator", choices=["abstract", "pymunk"], default="pymunk")
    parser.add_argument("--scenario", default="phase3_2v2_pass_required")
    parser.add_argument("--profile", default="nominal")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=340000)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 15.0 <= args.seconds <= 30.0:
        raise ValueError("--seconds must be between 15 and 30")
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
    maximum_frames = int(args.seconds * fps)
    records = []
    hidden_size = int(config["phase3"]["recurrent"]["hidden_size"])
    agent_names = env.possible_agents
    observation_size = env.observation_dimension
    action_size = env.action_size
    try:
        for episode in range(args.episodes):
            seed = args.seed_base + episode
            observations, _ = env.reset(seed=seed)
            hidden = torch.zeros(1, len(agent_names), hidden_size, device=device)
            frames = []
            final_metrics = {}
            while env.agents and len(frames) < maximum_frames:
                raw = np.zeros(
                    (len(agent_names), observation_size), dtype=np.float32
                )
                masks = np.zeros((len(agent_names), action_size), dtype=np.float32)
                for index, agent in enumerate(agent_names):
                    if agent in env.active_agents:
                        raw[index] = observations[agent]
                        masks[index] = env.action_mask(agent)
                normalized = normalizer.normalize(raw, config["observations"]["clip"])
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
                    env.render(), actions, env.ball["possession"], env.selected_profile
                )
                frames.extend([frame] * min(repeats, maximum_frames - len(frames)))
                observations, _, _, _, infos = env.step(actions)
                if not env.agents:
                    final_metrics = next(iter(infos.values()))["episode_metrics"]
            if frames:
                while len(frames) < maximum_frames:
                    frames.append(frames[-1])
            outcome = (
                "cooperative"
                if final_metrics.get("cooperative_success")
                else "goal"
                if final_metrics.get("success")
                else final_metrics.get("terminal_reason", "partial")
            )
            path = output_dir / (
                f"{args.simulator}_{args.scenario}_seed{seed}_{outcome}.mp4"
            )
            imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=7)
            records.append(
                {
                    "path": str(path),
                    "seed": seed,
                    "outcome": outcome,
                    "frames": len(frames),
                    "seconds": len(frames) / fps,
                    "checkpoint": str(checkpoint),
                    "metrics": final_metrics,
                }
            )
    finally:
        env.close()
    manifest = output_dir / "video_manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
