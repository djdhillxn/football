"""Evaluate a frozen recurrent Phase 3 checkpoint on declared scenarios."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from robosoccer.phase3 import make_phase3_environment
from robosoccer.recurrent import RecurrentSharedActor
from robosoccer.utils import RunningMeanStd, write_json


def resolve_checkpoint(run_dir, checkpoint):
    candidate = Path(checkpoint)
    if candidate.is_file():
        return candidate
    names = {
        "best": "best_composite_checkpoint.pt",
        "best_nominal": "best_nominal_checkpoint.pt",
        "best_cooperation": "best_cooperation_checkpoint.pt",
        "best_composite": "best_composite_checkpoint.pt",
        "final": "final_checkpoint.pt",
    }
    selected = run_dir / "models" / names.get(checkpoint, checkpoint)
    if not selected.is_file() and checkpoint == "best":
        selected = run_dir / "models" / "best_nominal_checkpoint.pt"
    if not selected.is_file():
        raise FileNotFoundError("Phase 3 checkpoint was not found: " + str(selected))
    return selected


def load_policy(run_dir, checkpoint_name, device):
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    stage_name = config["phase3"].get("active_stage", "stage_a")
    stage = config["phase3"]["stages"].get(stage_name, {})
    config["phase3"]["match_mode"] = bool(
        stage.get("match_mode", config["phase3"].get("match_mode", False))
    )
    checkpoint_path = resolve_checkpoint(run_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    probe = make_phase3_environment(config)
    observation_size = probe.observation_space(probe.possible_agents[0]).shape[0]
    action_size = probe.action_space(probe.possible_agents[0]).n
    probe.close()
    actor = RecurrentSharedActor(
        observation_size,
        action_size,
        config["model"],
        config["phase3"]["recurrent"],
    ).to(device)
    actor.load_state_dict(checkpoint["actor_weights"])
    actor.eval()
    normalizer = RunningMeanStd((observation_size,))
    normalizer.load_state_dict(checkpoint["observation_normalization"])
    return config, actor, normalizer, checkpoint_path


def evaluate(config, actor, normalizer, device, simulator, scenario, episodes, seed_base, profile):
    env = make_phase3_environment(
        config, simulator=simulator, scenario=scenario, profile_name=profile
    )
    hidden_size = int(config["phase3"]["recurrent"]["hidden_size"])
    agent_names = env.possible_agents
    observation_size = env.observation_dimension
    action_size = env.action_size
    rows = []
    try:
        for episode in range(int(episodes)):
            observations, _ = env.reset(seed=int(seed_base) + episode)
            hidden = torch.zeros(1, len(agent_names), hidden_size, device=device)
            team_return = 0.0
            action_counts = np.zeros(action_size, dtype=np.int64)
            while env.agents:
                raw = np.zeros(
                    (len(agent_names), observation_size), dtype=np.float32
                )
                masks = np.zeros((len(agent_names), action_size), dtype=np.float32)
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
                actions = {}
                for index, agent in enumerate(agent_names):
                    if agent in env.active_agents:
                        actions[agent] = int(selected[index])
                        action_counts[int(selected[index])] += 1
                observations, rewards, _, _, infos = env.step(actions)
                team_return += float(next(iter(rewards.values())))
            metrics = next(iter(infos.values()))["episode_metrics"]
            row = dict(metrics)
            row.update(
                {
                    "episode": episode,
                    "seed": int(seed_base) + episode,
                    "scenario": scenario,
                    "simulator": simulator,
                    "profile": profile,
                    "team_return": team_return,
                    "meaningful_action_count": int(np.count_nonzero(action_counts)),
                    "action_counts": json.dumps(action_counts.tolist()),
                }
            )
            rows.append(row)
    finally:
        env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--simulator", choices=["abstract", "pymunk"], default="pymunk")
    parser.add_argument(
        "--scenario",
        default="phase3_2v2_pass_required",
        choices=[
            "phase3_2v2_open",
            "phase3_2v2_pass_required",
            "phase3_3v2_open",
            "phase3_3v2_press",
        ],
    )
    parser.add_argument("--profile", default="nominal")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=330000)
    parser.add_argument("--prefix", default="phase3_evaluation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    config, actor, normalizer, checkpoint = load_policy(
        run_dir, args.checkpoint, device
    )
    rows = evaluate(
        config,
        actor,
        normalizer,
        device,
        args.simulator,
        args.scenario,
        args.episodes,
        args.seed_base,
        args.profile,
    )
    output_dir = run_dir / "eval" / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "simulator": args.simulator,
        "scenario": args.scenario,
        "profile": args.profile,
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "cooperative_success_rate": float(
            np.mean([row["cooperative_success"] for row in rows])
        ),
        "mean_return": float(np.mean([row["team_return"] for row in rows])),
        "pass_completion_rate": sum(row["completed_receptions"] for row in rows)
        / max(1, sum(row["valid_pass_attempts"] for row in rows)),
        "mean_meaningful_action_count": float(
            np.mean([row["meaningful_action_count"] for row in rows])
        ),
        "median_success_time": (
            float(
                np.median(
                    [
                        row["time_to_score"]
                        for row in rows
                        if row.get("time_to_score") is not None
                    ]
                )
            )
            if any(row.get("time_to_score") is not None for row in rows)
            else None
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
