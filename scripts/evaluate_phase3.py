"""Evaluate a frozen recurrent Phase 3 checkpoint on declared scenarios."""

import argparse
import copy
import csv
import hashlib
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
        "best_stage_r": "best_stage_r_checkpoint.pt",
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


def evaluate(
    config,
    actor,
    normalizer,
    device,
    simulator,
    scenario,
    episodes,
    seed_base,
    profile,
    defender_style="mixed",
    seed_category="evaluation",
    policy_run_id=None,
    checkpoint=None,
):
    env = make_phase3_environment(
        config,
        simulator=simulator,
        scenario=scenario,
        profile_name=profile,
        defender_style=defender_style,
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
                    "defender_style": metrics["defender_style"],
                    "requested_defender_style": defender_style,
                    "seed_category": seed_category,
                    "policy_run_id": policy_run_id,
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "team_return": team_return,
                    "meaningful_action_count": int(np.count_nonzero(action_counts)),
                    "action_counts": json.dumps(action_counts.tolist()),
                }
            )
            rows.append(row)
    finally:
        env.close()
    return rows


def summarize_rows(rows):
    successful_returns = [
        float(row["team_return"]) for row in rows if int(row["success"]) == 1
    ]
    failed_returns = [
        float(row["team_return"]) for row in rows if int(row["success"]) == 0
    ]
    successful_mean = (
        float(np.mean(successful_returns)) if successful_returns else None
    )
    failed_mean = float(np.mean(failed_returns)) if failed_returns else None
    return_gap = (
        successful_mean - failed_mean
        if successful_mean is not None and failed_mean is not None
        else None
    )
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "cooperative_success_rate": float(
            np.mean([row["cooperative_success"] for row in rows])
        ),
        "mean_return": float(np.mean([row["team_return"] for row in rows])),
        "mean_successful_return": successful_mean,
        "mean_failed_return": failed_mean,
        "success_failure_return_gap": return_gap,
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


def write_episode_csv(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_stage_r_r0_development_summary(result):
    if int(result["episodes_per_cell"]) < 100:
        return
    summary_path = Path("reports/phase3_development_summary.json")
    if not summary_path.is_file():
        return
    development = json.loads(summary_path.read_text(encoding="utf-8"))
    stage_r = development.setdefault("stage_r", {})
    stage_r["implementation_status"] = "ready"
    stage_r["r0_audit"] = copy.deepcopy(result)
    stage_r["r0_source_run_id"] = result["policy_run_id"]
    write_json(summary_path, development)


def run_stage_r_r0_audit(
    run_dir,
    checkpoint_name,
    device,
    episodes=100,
    seed_base=360000,
):
    config, actor, normalizer, checkpoint = load_policy(
        run_dir, checkpoint_name, device
    )
    styles = ["lane_block", "predictive", "zonal", "press"]
    scenarios = ["phase3_2v2_open", "phase3_2v2_pass_required"]
    simulators = ["abstract", "pymunk"]
    block = max(int(episodes), 100)
    rows = []
    cells = []
    cell_index = 0
    for simulator in simulators:
        for scenario in scenarios:
            for style in styles:
                cell_seed = int(seed_base) + cell_index * block
                cell_rows = evaluate(
                    config,
                    actor,
                    normalizer,
                    device,
                    simulator,
                    scenario,
                    episodes,
                    cell_seed,
                    "nominal",
                    defender_style=style,
                    seed_category="stage_r_r0_audit",
                    policy_run_id=run_dir.name,
                    checkpoint=checkpoint,
                )
                if any(row["defender_style"] != style for row in cell_rows):
                    raise RuntimeError("R0 fixed-style evaluation returned a mixed style")
                summary = summarize_rows(cell_rows)
                summary.update(
                    {
                        "simulator": simulator,
                        "scenario": scenario,
                        "defender_style": style,
                        "seed_start": cell_seed,
                        "seed_end": cell_seed + int(episodes) - 1,
                    }
                )
                rows.extend(cell_rows)
                cells.append(summary)
                cell_index += 1
    output_dir = run_dir / "eval" / "stage_r_r0_audit"
    write_episode_csv(output_dir / "episodes.csv", rows)
    write_episode_csv(output_dir / "summary.csv", cells)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    resolved_config = run_dir / "resolved_config.yaml"
    result = {
        "schema_version": 1,
        "audit": "Stage-R R0 frozen Stage-D matrix",
        "scientific_status": (
            "complete_evaluation" if int(episodes) >= 100 else "smoke_non_scientific"
        ),
        "policy_run_id": run_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "resolved_config": str(resolved_config),
        "resolved_config_sha256": hashlib.sha256(
            resolved_config.read_bytes()
        ).hexdigest(),
        "episodes_per_cell": int(episodes),
        "seed_base": int(seed_base),
        "seed_block_size": block,
        "seed_category": "stage_r_r0_audit",
        "matrix": {
            "simulators": simulators,
            "scenarios": scenarios,
            "defender_styles": styles,
        },
        "phase3_configuration": copy.deepcopy(config["phase3"]),
        "reward_configuration": copy.deepcopy(config["phase3_reward"]),
        "cells": cells,
        "episode_rows_path": str(output_dir / "episodes.csv"),
    }
    write_json(output_dir / "summary.json", result)
    update_stage_r_r0_development_summary(result)
    return result


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
    parser.add_argument(
        "--defender-style",
        choices=["lane_block", "predictive", "zonal", "press", "mixed"],
        default="mixed",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=330000)
    parser.add_argument(
        "--seed-category",
        choices=["training", "validation", "audit", "gate", "video", "evaluation"],
        default="evaluation",
    )
    parser.add_argument("--prefix", default="phase3_evaluation")
    parser.add_argument(
        "--stage-r-r0-audit",
        action="store_true",
        help="Run the frozen 4-style x 2-scenario x 2-backend Stage-R audit matrix.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    config, actor, normalizer, checkpoint = load_policy(
        run_dir, args.checkpoint, device
    )
    if args.stage_r_r0_audit:
        result = run_stage_r_r0_audit(
            run_dir,
            args.checkpoint,
            device,
            episodes=args.episodes,
            seed_base=args.seed_base,
        )
        print(json.dumps(result, indent=2))
        return
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
        defender_style=args.defender_style,
        seed_category=args.seed_category,
        policy_run_id=run_dir.name,
        checkpoint=checkpoint,
    )
    output_dir = run_dir / "eval" / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    write_episode_csv(output_dir / "episodes.csv", rows)
    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "policy_run_id": run_dir.name,
        "simulator": args.simulator,
        "scenario": args.scenario,
        "defender_style": args.defender_style,
        "profile": args.profile,
        "seed_base": args.seed_base,
        "seed_category": args.seed_category,
        **summarize_rows(rows),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
