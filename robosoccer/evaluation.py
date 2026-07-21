"""Policy evaluation, baseline studies, robustness grids, video, and aggregation."""

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from robosoccer.config import load_config, save_config
from robosoccer.environment import (
    AGENTS,
    DEFENDER_MODES,
    available_profile_names,
    baseline_actions,
    make_environment,
)
from robosoccer.training import load_checkpoint_actor
from robosoccer.utils import (
    create_run_directory,
    finalize_run,
    get_pyplot,
    initial_metadata,
    json_safe,
    setup_logging,
    utc_now,
    write_json,
)

logger = logging.getLogger(__name__)


def resolve_checkpoint(run_dir, checkpoint):
    run_dir = Path(run_dir).expanduser().resolve()
    if checkpoint in {"best", "final"}:
        path = run_dir / "models" / (checkpoint + "_checkpoint.pt")
    else:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = (run_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: " + str(path))
    return path


def policy_actions(actor, normalizer, observations, deterministic, device, clip):
    batch = np.asarray([observations[agent] for agent in AGENTS], dtype=np.float32)
    normalized = normalizer.normalize(batch, clip)
    with torch.no_grad():
        logits = actor(torch.as_tensor(normalized, device=device))
        if deterministic:
            selected = torch.argmax(logits, dim=-1)
        else:
            selected = torch.distributions.Categorical(logits=logits).sample()
    return {agent: int(selected[index].cpu()) for index, agent in enumerate(AGENTS)}


def flatten_episode_metrics(metrics, team_return, seed, method, simulator, profile, defender_mode):
    row = {
        "seed": int(seed),
        "method": method,
        "simulator": simulator,
        "profile": profile,
        "defender_mode": defender_mode,
        "team_return": float(team_return),
    }
    for key, value in metrics.items():
        if key == "sampled_parameters":
            row["sampled_parameters"] = json.dumps(json_safe(value), sort_keys=True)
        elif isinstance(value, str | int | float | bool) or value is None:
            row[key] = value
    steps = max(1, int(metrics.get("episode_steps", 1)))
    row["possession_fraction"] = metrics.get("possession_steps", 0) / steps
    row["redundant_chase_fraction"] = metrics.get("redundant_ball_chasing_steps", 0) / steps
    agent_action_steps = steps * len(AGENTS)
    row["invalid_action_fraction"] = (
        metrics.get("invalid_kick_attempts", 0) + metrics.get("invalid_pass_attempts", 0)
    ) / agent_action_steps
    row["mean_attacker_separation"] = metrics.get("separation_sum", 0.0) / steps
    row["mean_support_quality"] = metrics.get("support_quality_sum", 0.0) / steps
    for name in ["possession_fraction", "redundant_chase_fraction", "invalid_action_fraction"]:
        if not 0.0 <= row[name] <= 1.0:
            raise ValueError(name + " must be in [0, 1], observed " + str(row[name]))
    return row


def run_episode(
    config,
    simulator,
    seed,
    method,
    actor=None,
    normalizer=None,
    deterministic=True,
    profile=None,
    sampled_parameters=None,
    defender_mode=None,
    capture_frames=False,
    device="cpu",
):
    """Run one learned or scripted episode and return a flat row plus optional frames."""
    env = make_environment(config, simulator, profile_name=profile, render_mode="rgb_array")
    memory = {}
    frames = []
    options = {}
    if sampled_parameters is not None:
        options["sampled_parameters"] = sampled_parameters
    if defender_mode is not None:
        options["defender_mode"] = defender_mode
    observations, _ = env.reset(seed=seed, options=options)
    team_return = 0.0
    try:
        if capture_frames:
            frames.append(env.render())
        while env.agents:
            if actor is None:
                actions = baseline_actions(env, method, memory)
            else:
                actions = policy_actions(
                    actor,
                    normalizer,
                    observations,
                    deterministic,
                    device,
                    config["observations"]["clip"],
                )
            observations, rewards, _, _, infos = env.step(actions)
            team_return += float(rewards[AGENTS[0]])
            if capture_frames:
                frames.append(env.render())
        metrics = infos[AGENTS[0]]["episode_metrics"]
        row = flatten_episode_metrics(
            metrics,
            team_return,
            seed,
            method,
            simulator,
            env.selected_profile,
            env.defender["mode"],
        )
        return row, frames
    finally:
        env.close()


def bootstrap_interval(values, statistic, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [None, None]
    estimates = []
    for _ in range(int(samples)):
        draw = values[rng.integers(0, len(values), len(values))]
        estimates.append(float(statistic(draw)))
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def summarize_episodes(data, bootstrap_samples=1000, seed=0):
    if isinstance(data, list):
        data = pd.DataFrame(data)
    if data.empty:
        return {"episode_count": 0}
    returns = data["team_return"].astype(float).to_numpy()
    successes = data["success"].astype(float).to_numpy()
    successful_times = data.loc[data["success"].astype(bool), "time_to_score"].dropna().astype(float)
    attempts = float(data.get("pass_attempts", pd.Series(dtype=float)).fillna(0).sum())
    completed = float(data.get("completed_passes", pd.Series(dtype=float)).fillna(0).sum())
    worst_count = max(1, math.ceil(0.10 * len(returns)))
    sorted_returns = np.sort(returns)
    rng = np.random.default_rng(seed)
    summary = {
        "episode_count": len(data),
        "mean_return": float(np.mean(returns)),
        "standard_deviation_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_time_to_score": float(successful_times.mean()) if len(successful_times) else None,
        "median_time_to_score": float(successful_times.median()) if len(successful_times) else None,
        "pass_completion_rate": completed / attempts if attempts > 0 else 0.0,
        "possession_fraction": float(data["possession_fraction"].mean()),
        "redundant_chase_fraction": float(data["redundant_chase_fraction"].mean()),
        "collision_rate": float((data.get("attacker_collisions", 0) > 0).mean()),
        "out_of_bounds_rate": float((data.get("termination_reason", "") == "out_of_bounds").mean()),
        "invalid_action_rate": float(data["invalid_action_fraction"].mean()),
        "mean_action_switches": float(data.get("action_switches", pd.Series([0])).mean()),
        "worst_decile_return": float(np.mean(sorted_returns[:worst_count])),
        "cvar_10_return": float(np.mean(sorted_returns[:worst_count])),
        "mean_return_95_ci": bootstrap_interval(
            returns, np.mean, bootstrap_samples, rng
        ),
        "success_rate_95_ci": bootstrap_interval(
            successes, np.mean, bootstrap_samples, rng
        ),
    }
    if "profile" in data and data["profile"].nunique(dropna=False) > 1:
        profile_rates = data.groupby("profile")["success"].mean()
        summary["minimum_profile_success_rate"] = float(profile_rates.min())
        summary["mean_profile_success_rate"] = float(profile_rates.mean())
    return summary


def add_group_success_statistics(summary, data, column, label):
    """Add explicitly named min/mean success statistics for an evaluation grouping."""
    if column not in data or data.empty:
        return summary
    rates = data.groupby(column)["success"].mean()
    if rates.empty:
        return summary
    summary["minimum_" + label + "_success_rate"] = float(rates.min())
    summary["mean_" + label + "_success_rate"] = float(rates.mean())
    return summary


def phase1_readiness_audit(config, baseline_summary, abstract_summary=None, transfer_summary=None):
    """Evaluate transparent launch safeguards from completed Phase 1 summaries."""
    thresholds = config["evaluation"].get("phase1_gate", {})
    maximum_random = float(thresholds.get("maximum_random_pymunk_success", 0.50))
    minimum_spread = float(thresholds.get("minimum_pymunk_method_success_spread", 0.15))
    minimum_intercept = float(thresholds.get("minimum_nominal_intercept_success", 0.50))
    minimum_episodes = int(config["evaluation"].get("episodes", 100))
    checks = {}

    def add_check(name, passed, observed, criterion):
        checks[name] = {
            "passed": bool(passed),
            "observed": json_safe(observed),
            "criterion": criterion,
        }

    fraction_names = ["possession_fraction", "redundant_chase_fraction", "invalid_action_rate"]
    summaries = list(baseline_summary.values())
    if abstract_summary is not None:
        summaries.append(abstract_summary)
        summaries.extend(abstract_summary.get("by_defender_mode", {}).values())
    if transfer_summary is not None:
        summaries.append(transfer_summary)
        summaries.extend(transfer_summary.get("by_defender_mode", {}).values())
    bounded = all(
        0.0 <= float(summary[name]) <= 1.0
        for summary in summaries
        for name in fraction_names
        if name in summary
    )
    add_check("bounded_fraction_metrics", bounded, bounded, "every reported fraction is in [0, 1]")

    required_baselines = [
        method + "__" + simulator
        for method in ["random", "double_chase", "role_based"]
        for simulator in ["abstract", "pymunk"]
    ]
    baselines_complete = all(
        key in baseline_summary
        and int(baseline_summary[key].get("episode_count", 0)) >= minimum_episodes
        for key in required_baselines
    )
    add_check(
        "baseline_evaluation_complete",
        baselines_complete,
        {
            key: baseline_summary.get(key, {}).get("episode_count", 0)
            for key in required_baselines
        },
        "each scripted method/simulator cell has at least " + str(minimum_episodes) + " episodes",
    )

    pymunk_keys = [key for key in baseline_summary if key.endswith("__pymunk")]
    pymunk_rates = [float(baseline_summary[key]["success_rate"]) for key in pymunk_keys]
    random_pymunk = float(baseline_summary["random__pymunk"]["success_rate"])
    spread = max(pymunk_rates) - min(pymunk_rates)
    add_check(
        "pymunk_random_not_saturated",
        random_pymunk <= maximum_random,
        random_pymunk,
        "random Pymunk success <= " + str(maximum_random),
    )
    add_check(
        "pymunk_policy_sensitive",
        spread >= minimum_spread,
        spread,
        "Pymunk scripted-method success spread >= " + str(minimum_spread),
    )

    role_better = all(
        baseline_summary["role_based__" + simulator]["success_rate"]
        > baseline_summary["random__" + simulator]["success_rate"]
        for simulator in ["abstract", "pymunk"]
    )
    add_check(
        "role_success_exceeds_random",
        role_better,
        {
            simulator: {
                "role": baseline_summary["role_based__" + simulator]["success_rate"],
                "random": baseline_summary["random__" + simulator]["success_rate"],
            }
            for simulator in ["abstract", "pymunk"]
        },
        "role-based success exceeds random in both simulators",
    )
    role_coordinates = all(
        baseline_summary["role_based__" + simulator]["redundant_chase_fraction"]
        < baseline_summary["double_chase__" + simulator]["redundant_chase_fraction"]
        for simulator in ["abstract", "pymunk"]
    )
    add_check(
        "role_reduces_redundant_chasing",
        role_coordinates,
        {
            simulator: {
                "role": baseline_summary["role_based__" + simulator][
                    "redundant_chase_fraction"
                ],
                "double_chase": baseline_summary["double_chase__" + simulator][
                    "redundant_chase_fraction"
                ],
            }
            for simulator in ["abstract", "pymunk"]
        },
        "role-based redundant chasing is below double-chase in both simulators",
    )

    baseline_names = [
        "bounded_fraction_metrics",
        "baseline_evaluation_complete",
        "pymunk_random_not_saturated",
        "pymunk_policy_sensitive",
        "role_success_exceeds_random",
        "role_reduces_redundant_chasing",
    ]
    baseline_ready = all(checks[name]["passed"] for name in baseline_names)

    if abstract_summary is not None:
        intercept = abstract_summary.get("by_defender_mode", {}).get("intercept", {})
        intercept_success = float(intercept.get("success_rate", 0.0))
        add_check(
            "nominal_ippo_learns_intercept",
            intercept_success >= minimum_intercept,
            intercept_success,
            "abstract intercept success >= " + str(minimum_intercept),
        )
        strongest_scripted = max(
            float(baseline_summary[method + "__abstract"]["success_rate"])
            for method in ["random", "double_chase", "role_based"]
        )
        add_check(
            "nominal_ippo_exceeds_scripted_abstract",
            intercept_success > strongest_scripted,
            {"ippo_intercept": intercept_success, "strongest_scripted": strongest_scripted},
            "nominal IPPO intercept success exceeds every scripted abstract baseline",
        )
        mode_counts = {
            mode: abstract_summary.get("by_defender_mode", {}).get(mode, {}).get(
                "episode_count", 0
            )
            for mode in DEFENDER_MODES
        }
        add_check(
            "standard_defender_modes_complete",
            all(int(count) >= minimum_episodes for count in mode_counts.values()),
            mode_counts,
            "each defender mode has at least " + str(minimum_episodes) + " episodes",
        )
        add_check(
            "defender_mode_minimum_reported",
            "minimum_defender_mode_success_rate" in abstract_summary,
            abstract_summary.get("minimum_defender_mode_success_rate"),
            "standard summary contains an explicit defender-mode minimum",
        )
    if transfer_summary is not None:
        add_check(
            "frozen_pymunk_evaluation_completed",
            int(transfer_summary.get("episode_count", 0)) >= minimum_episodes,
            transfer_summary.get("episode_count", 0),
            "at least " + str(minimum_episodes) + " frozen-policy Pymunk episodes are present",
        )

    learned_names = [
        "nominal_ippo_learns_intercept",
        "nominal_ippo_exceeds_scripted_abstract",
        "standard_defender_modes_complete",
        "defender_mode_minimum_reported",
        "frozen_pymunk_evaluation_completed",
    ]
    phase2_ready = baseline_ready and all(
        name in checks and checks[name]["passed"] for name in learned_names
    )
    return {"baseline_ready": baseline_ready, "phase2_ready": phase2_ready, "checks": checks}


def save_episode_outputs(rows, output_dir, config, extra_summary=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(output_dir / "episodes.csv", index=False)
    summary = summarize_episodes(
        data,
        bootstrap_samples=config["evaluation"].get("bootstrap_samples", 1000),
        seed=config["experiment"]["seed"],
    )
    if extra_summary:
        summary.update(extra_summary)
    write_json(output_dir / "summary.json", summary)
    return data, summary


def plot_profile_success(data, output_path):
    plt = get_pyplot()
    if data.empty or "profile" not in data:
        return
    grouped = data.groupby("profile")["success"].mean().sort_values()
    figure, axis = plt.subplots(figsize=(8.0, max(4.2, len(grouped) * 0.28)))
    axis.barh(grouped.index, grouped.values)
    axis.set_title("Goal success by perturbation profile")
    axis.set_xlabel("Success rate")
    axis.set_ylabel("Perturbation profile")
    axis.set_xlim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def robustness_grid(config, actor, normalizer, simulator, method, output_dir, seed_base, device):
    plt = get_pyplot()
    delays = config["evaluation"]["action_delay_grid"]
    noises = config["evaluation"]["position_noise_grid"]
    episodes = int(config["evaluation"]["robustness_episodes_per_cell"])
    rows = []
    total = len(delays) * len(noises) * episodes
    progress = tqdm(total=total, desc="Robustness grid", dynamic_ncols=True)
    try:
        for delay_index, delay in enumerate(delays):
            for noise_index, noise in enumerate(noises):
                for episode in range(episodes):
                    seed = seed_base + delay_index * 10000 + noise_index * 1000 + episode
                    row, _ = run_episode(
                        config,
                        simulator,
                        seed,
                        method,
                        actor,
                        normalizer,
                        True,
                        profile="nominal",
                        sampled_parameters={
                            "action_latency": int(delay),
                            "localization_noise": float(noise),
                        },
                        device=device,
                    )
                    row["action_delay"] = int(delay)
                    row["position_noise"] = float(noise)
                    rows.append(row)
                    progress.update(1)
    finally:
        progress.close()
    data = pd.DataFrame(rows)
    data.to_csv(Path(output_dir) / "robustness_grid_long.csv", index=False)
    metric_specs = [
        ("success", "success", "Success rate"),
        ("team_return", "return", "Mean team return"),
        ("time_to_score", "time_to_score", "Mean time to score (s)"),
    ]
    for metric, stem, title in metric_specs:
        pivot = data.pivot_table(
            index="position_noise", columns="action_delay", values=metric, aggfunc="mean"
        )
        pivot.to_csv(Path(output_dir) / ("robustness_" + stem + "_pivot.csv"))
        figure, axis = plt.subplots(figsize=(6.8, 5.2))
        image = axis.imshow(pivot.values, origin="lower", aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        axis.set_yticks(range(len(pivot.index)), labels=[f"{value:.2f}" for value in pivot.index])
        axis.set_xlabel("Action delay (macro steps)")
        axis.set_ylabel("Localization noise std. dev. (m)")
        axis.set_title(title + " under joint perturbations")
        figure.colorbar(image, ax=axis, label=title)
        figure.tight_layout()
        figure.savefig(Path(output_dir) / ("robustness_" + stem + "_heatmap.png"), dpi=180)
        plt.close(figure)
    success_curve = data.groupby(["action_delay", "position_noise"])["success"].mean()
    normalized_auc = float(success_curve.mean())
    return rows, normalized_auc


def update_transfer_gaps(run_dir):
    run_dir = Path(run_dir)
    abstract_path = run_dir / "eval" / "abstract_standard" / "summary.json"
    transfer_path = run_dir / "eval" / "pymunk_transfer" / "summary.json"
    if not abstract_path.is_file() or not transfer_path.is_file():
        return None
    abstract = json.loads(abstract_path.read_text(encoding="utf-8"))
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    config = load_config(run_dir / "resolved_config.yaml")
    abstract_reference = abstract.get("by_defender_mode", {}).get(
        config["opponent"]["mode"], abstract
    )
    gaps = {}
    for metric in ["success_rate", "mean_return", "pass_completion_rate"]:
        gaps["transfer_gap_" + metric.replace("_rate", "")] = abstract_reference.get(
            metric, 0.0
        ) - transfer.get(metric, 0.0)
    abstract_time = abstract_reference.get("mean_time_to_score")
    transfer_time = transfer.get("mean_time_to_score")
    gaps["transfer_gap_time_to_score"] = (
        abstract_time - transfer_time
        if abstract_time is not None and transfer_time is not None
        else None
    )
    write_json(run_dir / "eval" / "transfer_gaps.json", gaps)
    return gaps


def evaluate_learned_run(
    run_dir,
    checkpoint="best",
    simulator="abstract",
    suite="standard",
    episodes=None,
    seed=None,
    deterministic=True,
    prefix=None,
    device="cpu",
):
    run_dir = Path(run_dir).expanduser().resolve()
    config = load_config(run_dir / "resolved_config.yaml")
    setup_logging(
        run_dir,
        config["logging"].get("console_level", "INFO"),
        config["logging"].get("file_level", "DEBUG"),
        filename="evaluation_" + simulator + "_" + suite + ".log",
    )
    checkpoint_path = resolve_checkpoint(run_dir, checkpoint)
    torch_device = torch.device(device)
    actor, normalizer, _ = load_checkpoint_actor(config, checkpoint_path, torch_device)
    method = config["experiment"]["name"]
    requested_episodes = episodes
    episodes = int(episodes or config["evaluation"]["episodes"])
    if seed is None:
        seed_key = "transfer_test" if simulator == "pymunk" else "abstract_test"
        seed = int(config["evaluation"]["seed_bases"][seed_key])
    default_name = simulator + "_" + suite
    output_dir = run_dir / "eval" / (prefix or default_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    normalized_auc = None
    requested_suites = [suite] if suite != "all" else ["standard", "profiles", "robustness"]
    for current_suite in requested_suites:
        if current_suite in {"standard", "transfer"}:
            defender_modes = DEFENDER_MODES if current_suite == "standard" else [config["opponent"]["mode"]]
            progress = tqdm(
                total=len(defender_modes) * episodes,
                desc=current_suite.title(),
                dynamic_ncols=True,
            )
            try:
                for defender_mode in defender_modes:
                    for episode in range(episodes):
                        row, _ = run_episode(
                            config,
                            simulator,
                            seed + episode,
                            method,
                            actor,
                            normalizer,
                            deterministic,
                            profile="nominal",
                            defender_mode=defender_mode,
                            device=torch_device,
                        )
                        rows.append(row)
                        progress.update(1)
            finally:
                progress.close()
        elif current_suite == "profiles":
            profile_episodes = int(requested_episodes or config["evaluation"]["profile_episodes"])
            profiles = available_profile_names(config, include_nominal=True)
            progress = tqdm(total=len(profiles) * profile_episodes, desc="Profiles", dynamic_ncols=True)
            try:
                for profile_index, profile in enumerate(profiles):
                    for episode in range(profile_episodes):
                        row, _ = run_episode(
                            config,
                            simulator,
                            seed + profile_index * 1000 + episode,
                            method,
                            actor,
                            normalizer,
                            deterministic,
                            profile=profile,
                            device=torch_device,
                        )
                        rows.append(row)
                        progress.update(1)
            finally:
                progress.close()
        elif current_suite == "robustness":
            grid_rows, normalized_auc = robustness_grid(
                config, actor, normalizer, simulator, method, output_dir, seed, torch_device
            )
            rows.extend(grid_rows)
        else:
            raise ValueError("Unsupported evaluation suite: " + str(current_suite))
    data, summary = save_episode_outputs(
        rows,
        output_dir,
        config,
        {
            "method": method,
            "simulator": simulator,
            "suite": suite,
            "checkpoint": str(checkpoint_path),
            "normalized_area_under_robustness_curve": normalized_auc,
        },
    )
    if "defender_mode" in data and suite in {"standard", "transfer"}:
        summary["by_defender_mode"] = {
            mode: summarize_episodes(
                group,
                config["evaluation"].get("bootstrap_samples", 1000),
                config["experiment"]["seed"],
            )
            for mode, group in data.groupby("defender_mode")
        }
        add_group_success_statistics(summary, data, "defender_mode", "defender_mode")
        write_json(output_dir / "summary.json", summary)
    if suite == "profiles":
        plot_profile_success(data, output_dir / "success_by_profile.png")
    canonical_transfer_evaluation = prefix is None and (
        (simulator == "abstract" and suite == "standard")
        or (simulator == "pymunk" and suite == "transfer")
    )
    gaps = update_transfer_gaps(run_dir) if canonical_transfer_evaluation else None
    if gaps is not None:
        summary["transfer_gaps"] = gaps
        write_json(output_dir / "summary.json", summary)
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        outputs = metadata.setdefault("output_artifact_paths", {})
        evaluations = outputs.setdefault("evaluations", {})
        evaluations[prefix or default_name] = {
            "summary": str(output_dir / "summary.json"),
            "episodes": str(output_dir / "episodes.csv"),
        }
        write_json(metadata_path, metadata)
    logger.info(
        "Evaluation complete | %s %s | episodes %d | success %.3f",
        simulator,
        suite,
        summary["episode_count"],
        summary["success_rate"],
    )
    return output_dir, summary


def evaluate_baselines(
    config,
    episodes=100,
    methods=None,
    run_name=None,
    videos=False,
    source_config=None,
    parsed_args=None,
):
    methods = methods or ["random", "double_chase", "role_based"]
    run_dir = create_run_directory(config, run_name=run_name or "baselines", method="heuristic")
    setup_logging(
        run_dir,
        config["logging"].get("console_level", "INFO"),
        config["logging"].get("file_level", "DEBUG"),
        filename="evaluation.log",
    )
    save_config(config, run_dir / "resolved_config.yaml")
    metadata = initial_metadata(
        config,
        run_dir,
        source_config,
        parsed_args or {"episodes": episodes, "methods": methods},
    )
    metadata["algorithm"] = "heuristic_baselines"
    write_json(run_dir / "run_metadata.json", metadata)
    rows = []
    seed_bases = config["evaluation"]["seed_bases"]
    progress = tqdm(total=len(methods) * 2 * int(episodes), desc="Baselines", dynamic_ncols=True)
    try:
        for method in methods:
            for simulator in ["abstract", "pymunk"]:
                seed_base = seed_bases["abstract_test" if simulator == "abstract" else "transfer_test"]
                for episode in range(int(episodes)):
                    row, _ = run_episode(
                        config,
                        simulator,
                        seed_base + episode,
                        method,
                        profile="nominal",
                    )
                    rows.append(row)
                    progress.update(1)
        data = pd.DataFrame(rows)
        data.to_csv(run_dir / "eval" / "baseline_episodes.csv", index=False)
        grouped_summary = {}
        for (method, simulator), group in data.groupby(["method", "simulator"]):
            method_summary = summarize_episodes(
                group, config["evaluation"].get("bootstrap_samples", 1000), config["experiment"]["seed"]
            )
            add_group_success_statistics(
                method_summary, group, "defender_mode", "defender_mode"
            )
            grouped_summary[method + "__" + simulator] = method_summary
        write_json(run_dir / "eval" / "baseline_summary.json", grouped_summary)
        plot_baseline_comparison(data, run_dir / "plots" / "baseline_success.png")
        if videos:
            for method in methods:
                record_videos(config, run_dir, "abstract", 1, baseline=method)
        metadata["status"] = "complete"
        metadata["utc_completion"] = utc_now()
        metadata["output_artifact_paths"] = {
            "episodes": str(run_dir / "eval" / "baseline_episodes.csv"),
            "summary": str(run_dir / "eval" / "baseline_summary.json"),
        }
        write_json(run_dir / "run_metadata.json", metadata)
        finalize_run(config, run_dir, metadata)
        logger.info("Completed baseline evaluation: %s", run_dir)
        return run_dir, grouped_summary
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["utc_completion"] = utc_now()
        metadata["failure_exception"] = str(exc)
        write_json(run_dir / "run_metadata.json", metadata)
        logger.exception("Baseline evaluation failed: %s", exc)
        raise
    finally:
        progress.close()


def plot_baseline_comparison(data, output_path):
    plt = get_pyplot()
    grouped = data.groupby(["method", "simulator"])["success"].mean().unstack(fill_value=0.0)
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    grouped.plot.bar(ax=axis)
    axis.set_title("Heuristic baseline goal success")
    axis.set_xlabel("Method")
    axis.set_ylabel("Success rate")
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Simulator")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def annotate_frame(frame, method, simulator, outcome):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    label = method + " | " + simulator + " | " + outcome
    draw.rectangle([12, 62, 12 + min(image.width - 24, 14 + len(label) * 7), 92], fill=(20, 25, 30))
    draw.text((20, 71), label, fill=(245, 245, 245))
    return np.asarray(image, dtype=np.uint8)


def record_videos(
    config,
    run_dir,
    simulator,
    episodes,
    checkpoint="best",
    baseline=None,
    seed=None,
    deterministic=True,
    profile="nominal",
):
    run_dir = Path(run_dir).expanduser().resolve()
    setup_logging(
        run_dir,
        config["logging"].get("console_level", "INFO"),
        config["logging"].get("file_level", "DEBUG"),
        filename="video.log",
    )
    method = baseline
    actor = None
    normalizer = None
    device = torch.device("cpu")
    if baseline is None:
        checkpoint_path = resolve_checkpoint(run_dir, checkpoint)
        actor, normalizer, _ = load_checkpoint_actor(config, checkpoint_path, device)
        method = config["experiment"]["name"]
    if seed is None:
        seed = int(config["evaluation"]["seed_bases"]["video"])
    candidate_rows = []
    candidate_count = int(episodes) if int(episodes) == 1 else int(episodes) * 5
    for candidate in range(candidate_count):
        row, _ = run_episode(
            config,
            simulator,
            seed + candidate,
            method,
            actor,
            normalizer,
            deterministic,
            profile=profile,
            capture_frames=False,
            device=device,
        )
        candidate_rows.append((candidate, row))
    selected = []
    if int(episodes) > 1:
        successes = [item for item in candidate_rows if item[1]["success"]]
        failures = [item for item in candidate_rows if not item[1]["success"]]
        if successes:
            selected.append(successes[0])
        if failures:
            selected.append(failures[0])
    for item in candidate_rows:
        if len(selected) >= int(episodes):
            break
        if all(item[0] != chosen[0] for chosen in selected):
            selected.append(item)
    output_paths = []
    for candidate, _ in selected:
        row, frames = run_episode(
            config,
            simulator,
            seed + candidate,
            method,
            actor,
            normalizer,
            deterministic,
            profile=profile,
            capture_frames=True,
            device=device,
        )
        outcome = "success" if row["success"] else row.get("termination_reason", "failure")
        annotated = [annotate_frame(frame, method, simulator, outcome) for frame in frames]
        filename = f"{method}_{simulator}_seed{seed + candidate}_{outcome}.mp4"
        path = run_dir / "videos" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with imageio.get_writer(
                path,
                fps=int(config["video"]["fps"]),
                codec="libx264",
                quality=7,
                macro_block_size=8,
            ) as writer:
                for frame in annotated:
                    writer.append_data(frame)
        except Exception as exc:
            raise RuntimeError("Video encoding failed; verify imageio-ffmpeg is installed: " + str(exc)) from exc
        output_paths.append(path)
        logger.info("Recorded %s", path)
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        outputs = metadata.setdefault("output_artifact_paths", {})
        existing = outputs.setdefault("videos", [])
        for path in output_paths:
            if str(path) not in existing:
                existing.append(str(path))
        write_json(metadata_path, metadata)
    return output_paths


def read_completed_manifests(runs_root):
    manifest = Path(runs_root) / "experiment_manifest.jsonl"
    if not manifest.is_file():
        return []
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") == "complete":
            entries.append(entry)
    return entries


def comparison_rows(run_paths):
    rows = []
    for run_path in run_paths:
        run_dir = Path(run_path).expanduser().resolve()
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            logger.warning("Skipping run without metadata: %s", run_dir)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        baseline_path = run_dir / "eval" / "baseline_summary.json"
        if baseline_path.is_file():
            summaries = json.loads(baseline_path.read_text(encoding="utf-8"))
            for key, summary in summaries.items():
                method, simulator = key.split("__", 1)
                rows.append({"run_dir": str(run_dir), "method": method, "simulator": simulator, **summary})
        for summary_path in (run_dir / "eval").glob("*/summary.json"):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "simulator" not in summary:
                continue
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "method": metadata.get("experiment_name", run_dir.name),
                    **summary,
                }
            )
    return rows


def latex_escape(text):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(text))


def export_generated_results(data, phase):
    destination = Path("reports/generated_results.tex")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if data.empty:
        content = (
            "% Generated placeholder: no final experiment artifacts were discovered.\n"
            "\\newcommand{\\ResultsAvailable}{false}\n"
            "\\newcommand{\\ResultsStatus}{Final empirical results have not yet been generated.}\n"
        )
    else:
        aggregate = data.groupby(["method", "simulator"], as_index=False).agg(
            success_rate=("success_rate", "mean"), mean_return=("mean_return", "mean")
        )
        lines = [
            "% Automatically generated from completed run artifacts.",
            "\\newcommand{\\ResultsAvailable}{true}",
            "\\newcommand{\\ResultsStatus}{"
            + ("Smoke infrastructure results; not final evidence." if phase == "smoke" else "Completed experimental artifacts.")
            + "}",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Observed cross-simulator evaluation summaries.}",
            "\\begin{tabular}{llrr}",
            "\\toprule Method & Simulator & Success & Mean return \\\\",
            "\\midrule",
        ]
        for _, row in aggregate.iterrows():
            lines.append(
                f"{latex_escape(row['method'])} & {latex_escape(row['simulator'])} & "
                f"{row['success_rate']:.3f} & {row['mean_return']:.3f} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        content = "\n".join(lines) + "\n"
    destination.write_text(content, encoding="utf-8")
    return destination


def compare_runs(run_paths=None, phase="final", runs_root="runs", export_report=False):
    setup_logging(None)
    if run_paths:
        paths = [Path(path) for path in run_paths]
    else:
        entries = read_completed_manifests(runs_root)
        paths = [Path(entry["run_directory"]) for entry in entries]
    if phase == "smoke":
        paths = [path for path in paths if "smoke" in path.name or "baseline" in path.name]
    elif phase == "final":
        paths = [path for path in paths if "smoke" not in path.name]
    rows = comparison_rows(paths)
    data = pd.DataFrame(rows)
    output_dir = Path(runs_root) / "comparisons" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + phase)
    output_dir.mkdir(parents=True, exist_ok=True)
    if data.empty:
        pd.DataFrame(columns=["method", "simulator", "success_rate", "mean_return"]).to_csv(
            output_dir / "main_comparison.csv", index=False
        )
        write_json(output_dir / "main_comparison.json", [])
        logger.warning("No compatible completed evaluation summaries were found.")
    else:
        data.to_csv(output_dir / "main_comparison.csv", index=False)
        write_json(output_dir / "main_comparison.json", data.to_dict(orient="records"))
        plot_comparisons(data, output_dir)
    plot_run_histories(paths, output_dir)
    plot_consolidated_profiles(paths, output_dir)
    plot_consolidated_heatmaps(paths, output_dir)
    if export_report:
        export_generated_results(data, phase)
    logger.info("Comparison artifacts written to %s", output_dir)
    return output_dir, data


def plot_comparisons(data, output_dir):
    plt = get_pyplot()
    aggregate = data.groupby(["method", "simulator"])["success_rate"].agg(["mean", "std"]).reset_index()
    pivot = aggregate.pivot(index="method", columns="simulator", values="mean").fillna(0.0)
    variability = aggregate.pivot(index="method", columns="simulator", values="std").fillna(0.0)
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    pivot.plot.bar(ax=axis, yerr=variability, capsize=3)
    axis.set_title("Method comparison across simulator fidelities")
    axis.set_xlabel("Method")
    axis.set_ylabel("Goal success rate")
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Simulator")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "method_success_comparison.png", dpi=180)
    figure.savefig(Path("reports/figures/method_success_comparison.png"), dpi=180)
    plt.close(figure)
    paired = data.pivot_table(index=["run_dir", "method"], columns="simulator", values="success_rate")
    if "abstract" in paired and "pymunk" in paired:
        gaps = (paired["abstract"] - paired["pymunk"]).dropna()
        if len(gaps):
            figure, axis = plt.subplots(figsize=(7.2, 4.3))
            gaps.groupby(level="method").mean().sort_values().plot.barh(ax=axis)
            axis.set_title("Zero-shot cross-simulator success gap")
            axis.set_xlabel("Abstract success minus Pymunk success")
            axis.set_ylabel("Method")
            axis.axvline(0.0, color="black", linewidth=0.8)
            figure.tight_layout()
            figure.savefig(Path(output_dir) / "transfer_gap.png", dpi=180)
            figure.savefig(Path("reports/figures/transfer_gap.png"), dpi=180)
            plt.close(figure)


def plot_run_histories(run_paths, output_dir):
    """Consolidate training-return and curriculum histories across completed seeds."""
    plt = get_pyplot()
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    plotted = False
    for run_path in run_paths:
        run_path = Path(run_path)
        metrics_path = run_path / "logs" / "metrics.csv"
        metadata_path = run_path / "run_metadata.json"
        if not metrics_path.is_file() or not metadata_path.is_file():
            continue
        metrics = pd.read_csv(metrics_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "environment_steps" not in metrics or "mean_episodic_return" not in metrics:
            continue
        label = metadata.get("experiment_name", run_path.name) + " seed " + str(metadata.get("seed", "?"))
        axis.plot(metrics["environment_steps"], metrics["mean_episodic_return"], label=label)
        plotted = True
    if plotted:
        axis.set_title("Training return across completed runs")
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Mean episodic team return")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(Path(output_dir) / "training_return.png", dpi=180)
        figure.savefig(Path("reports/figures/training_return.png"), dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    plotted = False
    for run_path in run_paths:
        run_path = Path(run_path)
        history_path = run_path / "logs" / "curriculum_history.csv"
        metadata_path = run_path / "run_metadata.json"
        if not history_path.is_file() or not metadata_path.is_file():
            continue
        history = pd.read_csv(history_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        method = metadata.get("experiment_name", run_path.name)
        for profile, group in history.groupby("profile"):
            axis.plot(
                group["update"],
                group["sampling_probability"],
                label=method + ": " + profile,
            )
            plotted = True
    if plotted:
        axis.set_title("Failure-directed curriculum probabilities")
        axis.set_xlabel("PPO update")
        axis.set_ylabel("Sampling probability")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6, ncol=2)
        figure.tight_layout()
        figure.savefig(Path(output_dir) / "curriculum_probabilities.png", dpi=180)
        figure.savefig(Path("reports/figures/curriculum_probabilities.png"), dpi=180)
    plt.close(figure)


def plot_consolidated_profiles(run_paths, output_dir):
    profile_rows = []
    for run_path in run_paths:
        run_path = Path(run_path)
        metadata_path = run_path / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        method = metadata.get("experiment_name", run_path.name)
        for episodes_path in (run_path / "eval").glob("*/episodes.csv"):
            episodes = pd.read_csv(episodes_path)
            if "profile" not in episodes or episodes["profile"].nunique() <= 1:
                continue
            grouped = episodes.groupby("profile")["success"].mean()
            for profile, success in grouped.items():
                profile_rows.append({"method": method, "profile": profile, "success": success})
    if not profile_rows:
        return
    data = pd.DataFrame(profile_rows)
    pivot = data.groupby(["profile", "method"])["success"].mean().unstack(fill_value=0.0)
    plt = get_pyplot()
    figure, axis = plt.subplots(figsize=(9.0, max(4.8, len(pivot) * 0.3)))
    pivot.plot.barh(ax=axis)
    axis.set_title("Robustness success by perturbation profile")
    axis.set_xlabel("Goal success rate")
    axis.set_ylabel("Perturbation profile")
    axis.set_xlim(0.0, 1.0)
    axis.legend(title="Method", fontsize=7)
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "robustness_profiles.png", dpi=180)
    figure.savefig(Path("reports/figures/robustness_profiles.png"), dpi=180)
    plt.close(figure)


def plot_consolidated_heatmaps(run_paths, output_dir):
    entries = []
    for run_path in run_paths:
        run_path = Path(run_path)
        metadata_path = run_path / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        method = metadata.get("experiment_name", run_path.name)
        for pivot_path in (run_path / "eval").glob("*/robustness_success_pivot.csv"):
            entries.append((method, pd.read_csv(pivot_path, index_col=0)))
    if not entries:
        return
    plt = get_pyplot()
    figure, axes = plt.subplots(1, len(entries), figsize=(5.3 * len(entries), 4.5), squeeze=False)
    image = None
    for axis, (method, pivot) in zip(axes[0], entries, strict=True):
        image = axis.imshow(pivot.values, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        axis.set_yticks(range(len(pivot.index)), labels=[f"{float(value):.2f}" for value in pivot.index])
        axis.set_title(method)
        axis.set_xlabel("Action delay (macro steps)")
        axis.set_ylabel("Localization noise std. dev. (m)")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Goal success rate", shrink=0.82)
    figure.suptitle("Consolidated cross-simulator robustness grids")
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.14, top=0.84, wspace=0.32)
    figure.savefig(Path(output_dir) / "consolidated_robustness_heatmaps.png", dpi=180)
    figure.savefig(Path("reports/figures/consolidated_robustness_heatmaps.png"), dpi=180)
    plt.close(figure)
