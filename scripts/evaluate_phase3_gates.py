"""Evaluate locked recurrent nominal or CC-FDR Phase 3 development gates."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from robosoccer.utils import write_json
from scripts.evaluate_phase3 import evaluate, load_policy


def compact_result_for_console(result):
    """Hide raw episode rows on stdout while leaving the saved result untouched."""
    compact = copy.deepcopy(result)
    for container in [compact, compact.get("nominal"), compact.get("cc_fdr")]:
        if not isinstance(container, dict):
            continue
        rows = container.pop("episode_rows", None)
        if rows is not None:
            container["episode_counts"] = {
                name: len(episodes) for name, episodes in rows.items()
            }
    return compact


def reduce_rows(rows):
    return {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "cooperative_success_rate": float(
            np.mean([row["cooperative_success"] for row in rows])
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
        "mean_meaningful_action_count": float(
            np.mean([row["meaningful_action_count"] for row in rows])
        ),
    }


def gate_b(run_dir, checkpoint, episodes, seed_base, device):
    config, actor, normalizer, resolved = load_policy(run_dir, checkpoint, device)
    cases = {}
    episode_rows = {}
    for index, scenario in enumerate(
        ["phase3_2v2_open", "phase3_3v2_open", "phase3_2v2_pass_required"]
    ):
        rows = evaluate(
            config,
            actor,
            normalizer,
            device,
            "pymunk",
            scenario,
            episodes,
            seed_base + index * 10000,
            "nominal",
        )
        episode_rows[scenario] = rows
        cases[scenario] = reduce_rows(rows)
    gates = config["phase3"]["gates"]
    checks = {
        "2v2_open_success": {
            "observed": cases["phase3_2v2_open"]["success_rate"],
            "criterion": gates["minimum_2v2_open_success"],
        },
        "3v2_open_success": {
            "observed": cases["phase3_3v2_open"]["success_rate"],
            "criterion": gates["minimum_3v2_open_success"],
        },
        "pass_required_cooperation": {
            "observed": cases["phase3_2v2_pass_required"][
                "cooperative_success_rate"
            ],
            "criterion": gates["minimum_cooperative_success"],
        },
        "successful_sequence_duration": {
            "observed": cases["phase3_2v2_pass_required"]["median_success_time"],
            "criterion": gates["minimum_successful_sequence_seconds"],
        },
        "meaningful_actions": {
            "observed": min(
                case["mean_meaningful_action_count"] for case in cases.values()
            ),
            "criterion": gates["minimum_meaningful_actions"],
        },
    }
    for check in checks.values():
        check["passed"] = (
            check["observed"] is not None
            and float(check["observed"]) >= float(check["criterion"])
        )
    return {
        "gate": "B",
        "checkpoint": str(resolved),
        "cases": cases,
        "episode_rows": episode_rows,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def policy_robustness(run_dir, checkpoint, episodes, seed_base, device):
    config, actor, normalizer, resolved = load_policy(run_dir, checkpoint, device)
    profiles = {}
    episode_rows = {}
    names = [name for name in config["phase3"]["profiles"] if name != "nominal"]
    per_profile = max(2, int(episodes) // max(1, len(names)))
    for index, profile in enumerate(names):
        rows = evaluate(
            config,
            actor,
            normalizer,
            device,
            "pymunk",
            "phase3_2v2_open",
            per_profile,
            seed_base + index * 1000,
            profile,
        )
        profiles[profile] = reduce_rows(rows)["success_rate"]
        episode_rows["profile_" + profile] = rows
    grid_names = [
        name
        for name in ["delay_low", "delay_high", "localization", "combined"]
        if name in profiles
    ]
    nominal_rows = evaluate(
        config,
        actor,
        normalizer,
        device,
        "pymunk",
        "phase3_2v2_open",
        episodes,
        seed_base + 20000,
        "nominal",
    )
    nominal = reduce_rows(nominal_rows)
    cooperation_rows = evaluate(
        config,
        actor,
        normalizer,
        device,
        "pymunk",
        "phase3_2v2_pass_required",
        episodes,
        seed_base + 30000,
        "nominal",
    )
    cooperation = reduce_rows(cooperation_rows)
    episode_rows["nominal"] = nominal_rows
    episode_rows["cooperation"] = cooperation_rows
    return {
        "checkpoint": str(resolved),
        "profile_mean": float(np.mean(list(profiles.values()))),
        "grid_auc": float(np.mean([profiles[name] for name in grid_names])),
        "profiles": profiles,
        "nominal_success": nominal["success_rate"],
        "cooperative_success": cooperation["cooperative_success_rate"],
        "episode_rows": episode_rows,
    }, config


def gate_c(nominal_run, nominal_checkpoint, cc_run, cc_checkpoint, episodes, seed_base, device):
    nominal, config = policy_robustness(
        nominal_run, nominal_checkpoint, episodes, seed_base, device
    )
    cc_fdr, _ = policy_robustness(
        cc_run, cc_checkpoint, episodes, seed_base, device
    )
    gates = config["phase3"]["gates"]
    comparisons = {
        "profile_improvement": cc_fdr["profile_mean"] - nominal["profile_mean"],
        "nominal_drop": nominal["nominal_success"] - cc_fdr["nominal_success"],
        "grid_regression": nominal["grid_auc"] - cc_fdr["grid_auc"],
        "cooperation_drop": nominal["cooperative_success"]
        - cc_fdr["cooperative_success"],
    }
    checks = {
        "profile_improvement": {
            "passed": comparisons["profile_improvement"]
            >= float(gates["minimum_profile_improvement"]),
            "observed": comparisons["profile_improvement"],
            "criterion": gates["minimum_profile_improvement"],
        },
        "nominal_noninferiority": {
            "passed": comparisons["nominal_drop"]
            <= float(gates["nominal_regression_margin"]),
            "observed": comparisons["nominal_drop"],
            "criterion": gates["nominal_regression_margin"],
        },
        "grid_noninferiority": {
            "passed": comparisons["grid_regression"]
            <= float(gates["maximum_grid_regression"]),
            "observed": comparisons["grid_regression"],
            "criterion": gates["maximum_grid_regression"],
        },
        "cooperation_noninferiority": {
            "passed": comparisons["cooperation_drop"] <= 0.0,
            "observed": comparisons["cooperation_drop"],
            "criterion": 0.0,
        },
    }
    return {
        "gate": "C",
        "nominal": nominal,
        "cc_fdr": cc_fdr,
        "comparisons": comparisons,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=["b", "c"], required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint", default="best_nominal")
    parser.add_argument("--nominal-run")
    parser.add_argument("--nominal-checkpoint", default="best_nominal")
    parser.add_argument("--cc-fdr-run")
    parser.add_argument("--cc-fdr-checkpoint", default="best_composite")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=350000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.gate == "b":
        if not args.run_dir:
            raise ValueError("--run-dir is required for Gate B")
        result = gate_b(
            Path(args.run_dir),
            args.checkpoint,
            args.episodes,
            args.seed_base,
            device,
        )
        default_output = Path(args.run_dir) / "eval" / "phase3_gate_b.json"
    else:
        if not args.nominal_run or not args.cc_fdr_run:
            raise ValueError("--nominal-run and --cc-fdr-run are required for Gate C")
        result = gate_c(
            Path(args.nominal_run),
            args.nominal_checkpoint,
            Path(args.cc_fdr_run),
            args.cc_fdr_checkpoint,
            args.episodes,
            args.seed_base,
            device,
        )
        default_output = Path(args.cc_fdr_run) / "eval" / "phase3_gate_c.json"
    output = Path(args.output) if args.output else default_output
    write_json(output, result)
    print(json.dumps(compact_result_for_console(result), indent=2))
    print(f"Full episode-level gate artifact: {output}")
    if not result["passed"]:
        print(
            f"Gate {result['gate']} did not pass; the artifact was saved before "
            "the fail-closed exit."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
