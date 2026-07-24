"""Evaluate locked recurrent nominal or CC-FDR Phase 3 development gates."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from robosoccer.config import load_config
from robosoccer.phase3 import run_stage_r_reward_invariants
from robosoccer.utils import write_json
from scripts.evaluate_phase3 import (
    evaluate,
    load_policy,
    summarize_rows,
    write_episode_csv,
)


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
    return summarize_rows(rows)


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


def _minimum_check(observed, criterion):
    return {
        "observed": observed,
        "criterion": criterion,
        "passed": observed is not None and float(observed) >= float(criterion),
    }


def _stage_r_safeguards(cases, fixed_styles, stage_r_gates):
    style_success = {
        style: fixed_styles[style]["success_rate"] for style in fixed_styles
    }
    worst_style = min(style_success.values())
    lane_predictive_mean = float(
        np.mean([style_success["lane_block"], style_success["predictive"]])
    )
    return {
        "pass_required_cooperation": _minimum_check(
            cases["phase3_2v2_pass_required"]["cooperative_success_rate"],
            stage_r_gates["minimum_pass_required_cooperation"],
        ),
        "3v2_open_success": _minimum_check(
            cases["phase3_3v2_open"]["success_rate"],
            stage_r_gates["minimum_3v2_open_success"],
        ),
        "2v2_open_worst_style_success": _minimum_check(
            worst_style,
            stage_r_gates["minimum_worst_style_success"],
        ),
        "2v2_open_lane_predictive_mean": _minimum_check(
            lane_predictive_mean,
            stage_r_gates["minimum_lane_predictive_mean"],
        ),
    }


def _reward_alignment_checks(fixed_styles, minimum_gap):
    checks = {}
    for style, summary in fixed_styles.items():
        gap = summary["success_failure_return_gap"]
        if gap is None:
            checks[style] = {
                "observed": None,
                "criterion": minimum_gap,
                "status": "not_applicable",
                "passed": None,
            }
        else:
            checks[style] = {
                "observed": gap,
                "criterion": minimum_gap,
                "status": "evaluated",
                "passed": float(gap) >= float(minimum_gap),
            }
    applicable = [check["passed"] for check in checks.values() if check["passed"] is not None]
    return checks, all(applicable)


def gate_b_r(
    run_dir,
    checkpoint,
    episodes,
    seed_base,
    device,
    reward_invariant_path,
):
    config, actor, normalizer, resolved = load_policy(run_dir, checkpoint, device)
    if int(config["phase3"].get("reward_schema_version", 1)) < 2:
        raise ValueError("Gate B-R requires the Stage-R reward schema")
    block = max(int(episodes), 100)
    rows = {}
    summaries = {}
    cell_index = 0

    def run_case(simulator, scenario, style, label):
        nonlocal cell_index
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
            seed_category="stage_r_gate_b_r",
            policy_run_id=run_dir.name,
            checkpoint=resolved,
        )
        if style != "mixed" and any(
            row["defender_style"] != style for row in cell_rows
        ):
            raise RuntimeError("Gate B-R fixed-style cell returned a mixed style")
        rows[label] = cell_rows
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
        summaries[label] = summary
        cell_index += 1

    for simulator in ["pymunk", "abstract"]:
        prefix = simulator + "__"
        for scenario in [
            "phase3_2v2_open",
            "phase3_2v2_pass_required",
            "phase3_3v2_open",
        ]:
            run_case(simulator, scenario, "mixed", prefix + scenario + "__mixed")
        for style in ["lane_block", "predictive", "zonal", "press"]:
            run_case(
                simulator,
                "phase3_2v2_open",
                style,
                prefix + "phase3_2v2_open__" + style,
            )

    def simulator_cases(simulator):
        prefix = simulator + "__"
        cases = {
            scenario: summaries[prefix + scenario + "__mixed"]
            for scenario in [
                "phase3_2v2_open",
                "phase3_2v2_pass_required",
                "phase3_3v2_open",
            ]
        }
        fixed = {
            style: summaries[prefix + "phase3_2v2_open__" + style]
            for style in ["lane_block", "predictive", "zonal", "press"]
        }
        return cases, fixed

    pymunk_cases, pymunk_fixed = simulator_cases("pymunk")
    abstract_cases, abstract_fixed = simulator_cases("abstract")
    gates = config["phase3"]["gates"]
    original_checks = {
        "2v2_open_success": _minimum_check(
            pymunk_cases["phase3_2v2_open"]["success_rate"],
            gates["minimum_2v2_open_success"],
        ),
        "3v2_open_success": _minimum_check(
            pymunk_cases["phase3_3v2_open"]["success_rate"],
            gates["minimum_3v2_open_success"],
        ),
        "pass_required_cooperation": _minimum_check(
            pymunk_cases["phase3_2v2_pass_required"][
                "cooperative_success_rate"
            ],
            gates["minimum_cooperative_success"],
        ),
        "successful_sequence_duration": _minimum_check(
            pymunk_cases["phase3_2v2_pass_required"]["median_success_time"],
            gates["minimum_successful_sequence_seconds"],
        ),
        "meaningful_actions": _minimum_check(
            min(
                case["mean_meaningful_action_count"]
                for case in pymunk_cases.values()
            ),
            gates["minimum_meaningful_actions"],
        ),
    }
    stage_r_gates = config["phase3"]["stage_r"]["gate_b_r"]
    pymunk_safeguards = _stage_r_safeguards(
        pymunk_cases, pymunk_fixed, stage_r_gates
    )
    abstract_safeguards = _stage_r_safeguards(
        abstract_cases, abstract_fixed, stage_r_gates
    )
    alignment_checks, alignment_passed = _reward_alignment_checks(
        pymunk_fixed,
        stage_r_gates["minimum_success_failure_return_gap"],
    )
    invariant_path = Path(reward_invariant_path)
    if not invariant_path.is_file():
        raise FileNotFoundError(
            "Stage-R reward-invariant artifact was not found: " + str(invariant_path)
        )
    reward_invariants = json.loads(invariant_path.read_text(encoding="utf-8"))
    reward_invariants_passed = (
        reward_invariants.get("reward_schema_version") == 2
        and reward_invariants.get("passed", False)
    )
    original_passed = all(check["passed"] for check in original_checks.values())
    pymunk_safeguards_passed = all(
        check["passed"] for check in pymunk_safeguards.values()
    )
    abstract_safeguards_passed = all(
        check["passed"] for check in abstract_safeguards.values()
    )
    passed = (
        original_passed
        and pymunk_safeguards_passed
        and abstract_safeguards_passed
        and alignment_passed
        and reward_invariants_passed
    )
    if passed:
        next_branch = "cc_fdr_authorized"
    elif abstract_safeguards_passed and reward_invariants_passed:
        next_branch = "targeted_abstract_dynamics_envelope"
    else:
        next_branch = "rerun_stage_a_through_r_with_corrected_reward"
    return {
        "schema_version": 1,
        "gate": "B-R",
        "historical_gate_b_unchanged": True,
        "policy_run_id": run_dir.name,
        "checkpoint": str(resolved),
        "episodes_per_cell": int(episodes),
        "seed_base": int(seed_base),
        "seed_block_size": block,
        "seed_category": "stage_r_gate_b_r",
        "summaries": summaries,
        "episode_rows": rows,
        "original_gate_b_checks": original_checks,
        "pymunk_stage_r_safeguards": pymunk_safeguards,
        "abstract_stage_r_safeguards": abstract_safeguards,
        "reward_alignment_checks": alignment_checks,
        "reward_invariants": reward_invariants,
        "component_passes": {
            "original_gate_b": original_passed,
            "pymunk_stage_r_safeguards": pymunk_safeguards_passed,
            "abstract_stage_r_safeguards": abstract_safeguards_passed,
            "reward_alignment": alignment_passed,
            "reward_invariants": reward_invariants_passed,
        },
        "passed": passed,
        "cc_fdr_authorized": passed,
        "next_branch": next_branch,
    }


def update_stage_r_development_summary(run_dir, result):
    summary_path = Path("reports/phase3_development_summary.json")
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    compact = compact_result_for_console(result)
    stage_r = summary.setdefault("stage_r", {})
    stage_r.update(
        {
            "implementation_status": "ready",
            "scientific_status": "gate_b_r_complete",
            "run_id": Path(run_dir).name,
            "checkpoint": result["checkpoint"],
            "gate_b_r": compact,
            "cc_fdr_authorized": result["cc_fdr_authorized"],
            "next_branch": result["next_branch"],
        }
    )
    historical_stage_d = next(
        (
            entry["run_id"]
            for entry in summary.get("nominal_stages", [])
            if entry.get("stage") == "D"
        ),
        None,
    )
    r0_path = (
        Path("runs")
        / str(historical_stage_d)
        / "eval"
        / "stage_r_r0_audit"
        / "summary.json"
    )
    if r0_path.is_file():
        stage_r["r0_audit"] = json.loads(r0_path.read_text(encoding="utf-8"))
    training_path = Path(run_dir) / "logs" / "phase3_training_summary.json"
    if training_path.is_file():
        training_summary = json.loads(training_path.read_text(encoding="utf-8"))
        stage_r["training_summary"] = training_summary
        stage_r["abstract_validation"] = training_summary.get("last_validation")
    decision = summary.setdefault("decision", {})
    decision["gate_b_r_passed"] = bool(result["passed"])
    decision["cc_fdr_authorized"] = bool(result["cc_fdr_authorized"])
    decision["final_seed_confirmation_authorized"] = False
    decision["next_branch"] = result["next_branch"]
    write_json(summary_path, summary)


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
    parser.add_argument(
        "--gate", choices=["b", "b-r", "c", "reward-invariants"], required=True
    )
    parser.add_argument("--config", default="configs/phase3_stage_r.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint", default="best_nominal")
    parser.add_argument("--nominal-run")
    parser.add_argument("--nominal-checkpoint", default="best_nominal")
    parser.add_argument("--cc-fdr-run")
    parser.add_argument("--cc-fdr-checkpoint", default="best_composite")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--reward-invariants")
    args = parser.parse_args()
    if args.gate == "reward-invariants":
        result = run_stage_r_reward_invariants(load_config(args.config))
        output = (
            Path(args.output)
            if args.output
            else Path("runs/logs/reward_invariant_summary.json")
        )
        write_json(output, result)
        print(json.dumps(result, indent=2))
        print(f"Reward-invariant artifact: {output}")
        if not result["passed"]:
            raise SystemExit(2)
        return
    device = torch.device(args.device)
    seed_base = args.seed_base
    if seed_base is None:
        seed_base = 370000 if args.gate == "b-r" else 350000
    if args.gate == "b":
        if not args.run_dir:
            raise ValueError("--run-dir is required for Gate B")
        result = gate_b(
            Path(args.run_dir),
            args.checkpoint,
            args.episodes,
            seed_base,
            device,
        )
        default_output = Path(args.run_dir) / "eval" / "phase3_gate_b.json"
    elif args.gate == "b-r":
        if not args.run_dir:
            raise ValueError("--run-dir is required for Gate B-R")
        run_dir = Path(args.run_dir)
        reward_invariants = (
            Path(args.reward_invariants)
            if args.reward_invariants
            else run_dir / "logs" / "reward_invariant_summary.json"
        )
        result = gate_b_r(
            run_dir,
            args.checkpoint,
            args.episodes,
            seed_base,
            device,
            reward_invariants,
        )
        output_dir = run_dir / "eval" / "phase3_gate_b_r"
        flattened_rows = []
        for label, rows in result["episode_rows"].items():
            for row in rows:
                copied = dict(row)
                copied["gate_cell"] = label
                flattened_rows.append(copied)
        write_episode_csv(output_dir / "episodes.csv", flattened_rows)
        summary_rows = []
        for label, summary in result["summaries"].items():
            copied = dict(summary)
            copied["gate_cell"] = label
            summary_rows.append(copied)
        write_episode_csv(output_dir / "summary.csv", summary_rows)
        write_json(
            output_dir / "summary.json",
            compact_result_for_console(result),
        )
        default_output = run_dir / "eval" / "phase3_gate_b_r.json"
    else:
        if not args.nominal_run or not args.cc_fdr_run:
            raise ValueError("--nominal-run and --cc-fdr-run are required for Gate C")
        result = gate_c(
            Path(args.nominal_run),
            args.nominal_checkpoint,
            Path(args.cc_fdr_run),
            args.cc_fdr_checkpoint,
            args.episodes,
            seed_base,
            device,
        )
        default_output = Path(args.cc_fdr_run) / "eval" / "phase3_gate_c.json"
    output = Path(args.output) if args.output else default_output
    write_json(output, result)
    if args.gate == "b-r":
        update_stage_r_development_summary(Path(args.run_dir), result)
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
