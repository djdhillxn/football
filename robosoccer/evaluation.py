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
    ACTION_NAMES,
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
    initial_state=None,
    capture_frames=False,
    device="cpu",
):
    """Run one learned or scripted episode and return a flat row plus optional frames."""
    env = make_environment(config, simulator, profile_name=profile, render_mode="rgb_array")
    memory = {}
    frames = []
    options = {"apply_disabled_parameters": False}
    if sampled_parameters is not None:
        options["sampled_parameters"] = sampled_parameters
    if defender_mode is not None:
        options["defender_mode"] = defender_mode
    if initial_state is not None:
        options["initial_state"] = initial_state
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
    def total(column):
        return float(data.get(column, pd.Series(dtype=float)).fillna(0).sum())

    attempts = total("pass_attempts")
    completed = total("completed_passes")
    opportunities = total("pass_opportunities")
    opportunity_attempts = total("pass_attempts_on_opportunity")
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
        "pass_attempt_count": int(attempts),
        "completed_pass_count": int(completed),
        "intercepted_pass_count": int(total("intercepted_passes")),
        "pass_opportunity_count": int(opportunities),
        "pass_attempts_on_opportunity_count": int(opportunity_attempts),
        "pass_opportunity_action_rate": (
            opportunity_attempts / opportunities if opportunities > 0 else 0.0
        ),
        "receiver_possession_after_pass_count": int(total("receiver_possessions_after_pass")),
        "post_pass_goal_count": int(total("goals_after_completed_pass")),
        "cooperative_success_rate": float(
            data.get("cooperative_probe_success", pd.Series([False] * len(data)))
            .astype(float)
            .mean()
        ),
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


def cooperation_probe_initial_state(config, seed):
    """Create a mirrored pass-needed state determined only by the evaluation seed."""
    probe = config["evaluation"].get("cooperation_probe", {})
    rng = np.random.default_rng(int(seed))
    side = 1.0 if int(seed) % 2 == 0 else -1.0
    carrier = AGENTS[(int(seed) // 2) % len(AGENTS)]
    receiver = AGENTS[1 - AGENTS.index(carrier)]
    jitter = float(probe.get("position_jitter", 0.08))
    ball_position = np.array(
        [0.30 + rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
        dtype=np.float64,
    )
    receiver_position = ball_position + np.array(
        [
            float(probe.get("receiver_forward_offset", 1.20)),
            side * float(probe.get("receiver_lateral_offset", 1.20)),
        ]
    )
    carrier_position = ball_position - np.array(
        [float(probe.get("carrier_ball_distance", 0.33)), 0.0]
    )
    defender_position = ball_position + np.array(
        [float(probe.get("defender_forward_offset", 1.55)), 0.0]
    )
    pass_heading = math.atan2(
        receiver_position[1] - ball_position[1],
        receiver_position[0] - ball_position[0],
    )
    state = {
        "players": {
            carrier: {
                "position": carrier_position.tolist(),
                "velocity": [0.0, 0.0],
                "heading": pass_heading,
            },
            receiver: {
                "position": receiver_position.tolist(),
                "velocity": [0.0, 0.0],
                "heading": 0.0,
            },
        },
        "ball": {
            "position": ball_position.tolist(),
            "velocity": [0.0, 0.0],
            "last_touch": carrier,
        },
        "defender": {
            "position": defender_position.tolist(),
            "velocity": [0.0, 0.0],
            "heading": math.pi,
        },
    }
    return state, {"probe_carrier": carrier, "probe_receiver": receiver, "probe_side": int(side)}


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
        if suite == "cooperation":
            seed_key = "cooperation_probe"
        else:
            seed_key = "transfer_test" if simulator == "pymunk" else "abstract_test"
        seed = int(config["evaluation"]["seed_bases"][seed_key])
    default_name = simulator + "_" + suite
    output_dir = run_dir / "eval" / (prefix or default_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    normalized_auc = None
    requested_suites = [suite] if suite != "all" else [
        "standard",
        "profiles",
        "robustness",
        "cooperation",
    ]
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
        elif current_suite == "cooperation":
            probe_episodes = int(
                requested_episodes
                or config["evaluation"].get("cooperation_probe", {}).get("episodes", 100)
            )
            progress = tqdm(total=probe_episodes, desc="Cooperation probe", dynamic_ncols=True)
            try:
                for episode in range(probe_episodes):
                    episode_seed = seed + episode
                    initial_state, probe_metadata = cooperation_probe_initial_state(
                        config, episode_seed
                    )
                    row, _ = run_episode(
                        config,
                        simulator,
                        episode_seed,
                        method,
                        actor,
                        normalizer,
                        deterministic,
                        profile="nominal",
                        sampled_parameters={
                            "action_latency": 0,
                            "observation_latency": 0,
                            "communication_latency": 0,
                            "packet_loss": 0.0,
                            "localization_noise": 0.0,
                            "defender_speed_multiplier": 0.0,
                        },
                        defender_mode="intercept",
                        initial_state=initial_state,
                        device=torch_device,
                    )
                    row.update(probe_metadata)
                    rows.append(row)
                    progress.update(1)
            finally:
                progress.close()
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
    scenario="nominal",
    matched=False,
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
        seed_key = "cooperation_probe" if scenario == "cooperation" else "video"
        seed = int(config["evaluation"]["seed_bases"][seed_key])
    if scenario not in {"nominal", "cooperation"}:
        raise ValueError("Unsupported video scenario: " + str(scenario))

    def episode_options(episode_seed):
        if scenario != "cooperation":
            return {}
        initial_state, _ = cooperation_probe_initial_state(config, episode_seed)
        return {
            "initial_state": initial_state,
            "sampled_parameters": {
                "action_latency": 0,
                "observation_latency": 0,
                "communication_latency": 0,
                "packet_loss": 0.0,
                "localization_noise": 0.0,
                "defender_speed_multiplier": 0.0,
            },
            "defender_mode": "intercept",
        }

    candidate_rows = []
    candidate_count = int(episodes) if matched or int(episodes) == 1 else int(episodes) * 5
    for candidate in range(candidate_count):
        options = episode_options(seed + candidate)
        row, _ = run_episode(
            config,
            simulator,
            seed + candidate,
            method,
            actor,
            normalizer,
            deterministic,
            profile=profile,
            sampled_parameters=options.get("sampled_parameters"),
            defender_mode=options.get("defender_mode"),
            initial_state=options.get("initial_state"),
            capture_frames=False,
            device=device,
        )
        candidate_rows.append((candidate, row))
    selected = []
    if matched:
        selected = candidate_rows[: int(episodes)]
    elif int(episodes) > 1:
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
        options = episode_options(seed + candidate)
        row, frames = run_episode(
            config,
            simulator,
            seed + candidate,
            method,
            actor,
            normalizer,
            deterministic,
            profile=profile,
            sampled_parameters=options.get("sampled_parameters"),
            defender_mode=options.get("defender_mode"),
            initial_state=options.get("initial_state"),
            capture_frames=True,
            device=device,
        )
        outcome = "success" if row["success"] else row.get("termination_reason", "failure")
        annotated = [annotate_frame(frame, method, simulator, outcome) for frame in frames]
        filename = f"{method}_{simulator}_{scenario}_seed{seed + candidate}_{outcome}.mp4"
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


def trace_learned_action_delays(
    run_dir,
    delays=(0, 1, 2, 3, 4, 5),
    seed=None,
    simulator="pymunk",
    output_name="confirmatory_delay_traces",
    checkpoint="best",
    localization_noise=0.0,
    device="cpu",
):
    """Replay one matched episode per delay and persist requested/queued/applied actions."""
    run_dir = Path(run_dir).expanduser().resolve()
    config = load_config(run_dir / "resolved_config.yaml")
    output_dir = run_dir / "eval" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        run_dir,
        config["logging"].get("console_level", "INFO"),
        config["logging"].get("file_level", "DEBUG"),
        filename="action_delay_trace.log",
    )
    torch_device = torch.device(device)
    actor, normalizer, _ = load_checkpoint_actor(
        config, resolve_checkpoint(run_dir, checkpoint), torch_device
    )
    if seed is None:
        seed = int(config["evaluation"]["seed_bases"].get("delay_audit", 260000))
    method = config["experiment"]["name"]
    trace_rows = []
    outcome_rows = []
    for delay in [int(value) for value in delays]:
        episode_seed = int(seed) + delay
        env = make_environment(config, simulator, profile_name="nominal", render_mode="rgb_array")
        observations, _ = env.reset(
            seed=episode_seed,
            options={
                "apply_disabled_parameters": False,
                "sampled_parameters": {
                    "action_latency": delay,
                    "localization_noise": float(localization_noise),
                },
                "defender_mode": config["opponent"]["mode"],
            },
        )
        team_return = 0.0
        final_infos = None
        try:
            while env.agents:
                requested = policy_actions(
                    actor,
                    normalizer,
                    observations,
                    True,
                    torch_device,
                    config["observations"]["clip"],
                )
                observations, rewards, _, _, infos = env.step(requested)
                team_return += float(rewards[AGENTS[0]])
                final_infos = infos
                for agent in AGENTS:
                    trace_rows.append(
                        {
                            "method": method,
                            "simulator": simulator,
                            "training_seed": int(config["experiment"]["seed"]),
                            "episode_seed": episode_seed,
                            "action_latency": delay,
                            "macro_step": int(env.step_count),
                            "agent": agent,
                            "requested_action": int(infos[agent]["requested_action"]),
                            "requested_action_name": ACTION_NAMES[
                                int(infos[agent]["requested_action"])
                            ],
                            "queued_actions": json.dumps(infos[agent]["queued_actions"]),
                            "applied_action": int(infos[agent]["applied_action"]),
                            "applied_action_age_steps": infos[agent][
                                "applied_action_age_steps"
                            ],
                            "applied_action_name": ACTION_NAMES[
                                int(infos[agent]["applied_action"])
                            ],
                            "player_x": float(env.players[agent]["position"][0]),
                            "player_y": float(env.players[agent]["position"][1]),
                            "ball_x": float(env.ball["position"][0]),
                            "ball_y": float(env.ball["position"][1]),
                        }
                    )
            metrics = final_infos[AGENTS[0]]["episode_metrics"]
            outcome_rows.append(
                {
                    "method": method,
                    "simulator": simulator,
                    "training_seed": int(config["experiment"]["seed"]),
                    "episode_seed": episode_seed,
                    "action_latency": delay,
                    "success": bool(metrics["success"]),
                    "termination_reason": metrics["termination_reason"],
                    "episode_steps": int(metrics["episode_steps"]),
                    "team_return": float(team_return),
                    "action_switches": int(metrics["action_switches"]),
                }
            )
        finally:
            env.close()
    pd.DataFrame(trace_rows).to_csv(output_dir / "action_trace.csv", index=False)
    pd.DataFrame(outcome_rows).to_csv(output_dir / "outcomes.csv", index=False)
    summary = {
        "method": method,
        "simulator": simulator,
        "training_seed": int(config["experiment"]["seed"]),
        "episode_seed_base": int(seed),
        "delays": [int(value) for value in delays],
        "localization_noise": float(localization_noise),
        "episode_count": len(outcome_rows),
        "trace_row_count": len(trace_rows),
        "outcomes": outcome_rows,
    }
    write_json(output_dir / "summary.json", summary)
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        evaluations = metadata.setdefault("output_artifact_paths", {}).setdefault(
            "evaluations", {}
        )
        evaluations[output_name] = {
            "summary": str(output_dir / "summary.json"),
            "action_trace": str(output_dir / "action_trace.csv"),
            "outcomes": str(output_dir / "outcomes.csv"),
        }
        write_json(metadata_path, metadata)
    logger.info("Action-delay traces written to %s", output_dir)
    return output_dir, summary


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
    """Read evaluation summaries without erasing suite or training-seed identity."""
    rows = []
    for run_path in run_paths:
        run_dir = Path(run_path).expanduser().resolve()
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            logger.warning("Skipping run without metadata: %s", run_dir)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training_seed = int(metadata.get("seed", 0))
        git_revision = metadata.get("git_commit")
        opponent_mode = "intercept"
        resolved_config = run_dir / "resolved_config.yaml"
        if resolved_config.is_file():
            opponent_mode = load_config(resolved_config).get("opponent", {}).get(
                "mode", opponent_mode
            )
        baseline_path = run_dir / "eval" / "baseline_summary.json"
        if baseline_path.is_file():
            summaries = json.loads(baseline_path.read_text(encoding="utf-8"))
            for key, summary in summaries.items():
                method, simulator = key.split("__", 1)
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "training_seed": training_seed,
                        "git_commit": git_revision,
                        "evaluation_name": "baseline",
                        "suite": "baseline",
                        "method": method,
                        "simulator": simulator,
                        "canonical_success_rate": summary.get("success_rate"),
                        **summary,
                    }
                )
        for summary_path in (run_dir / "eval").glob("*/summary.json"):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not {"simulator", "suite", "success_rate"}.issubset(summary):
                continue
            canonical_success = summary.get("success_rate")
            if summary.get("simulator") == "abstract" and summary.get("suite") == "standard":
                canonical_success = summary.get("by_defender_mode", {}).get(
                    opponent_mode, summary
                ).get("success_rate", canonical_success)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "training_seed": training_seed,
                    "git_commit": git_revision,
                    "evaluation_name": summary_path.parent.name,
                    "method": metadata.get("experiment_name", run_dir.name),
                    "canonical_success_rate": canonical_success,
                    **summary,
                }
            )
    return rows


def suite_aggregate(data):
    """Aggregate across independent training seeds, never across evaluation suites."""
    if data.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "simulator",
                "suite",
                "evaluation_name",
                "training_seed_count",
                "success_rate_mean",
                "success_rate_std",
                "mean_return_mean",
            ]
        )
    keys = ["method", "simulator", "suite", "evaluation_name"]
    aggregate = data.groupby(keys, dropna=False).agg(
        training_seed_count=("training_seed", "nunique"),
        success_rate_mean=("success_rate", "mean"),
        success_rate_std=("success_rate", "std"),
        mean_return_mean=("mean_return", "mean"),
    )
    return aggregate.reset_index()


def canonical_transfer_rows(data):
    """Pair only the declared abstract-intercept and Pymunk transfer suites per run."""
    pairs = [
        ("abstract_standard", "pymunk_transfer", "pilot"),
        (
            "confirmatory_abstract_standard",
            "confirmatory_pymunk_transfer",
            "confirmatory",
        ),
    ]
    rows = []
    if data.empty:
        return pd.DataFrame(rows)
    for run_dir, group in data.groupby("run_dir"):
        for abstract_name, transfer_name, protocol in pairs:
            abstract = group[group["evaluation_name"] == abstract_name]
            transfer = group[group["evaluation_name"] == transfer_name]
            if len(abstract) != 1 or len(transfer) != 1:
                continue
            abstract_row = abstract.iloc[0]
            transfer_row = transfer.iloc[0]
            abstract_success = float(abstract_row["canonical_success_rate"])
            transfer_success = float(transfer_row["canonical_success_rate"])
            rows.append(
                {
                    "run_dir": run_dir,
                    "method": abstract_row["method"],
                    "training_seed": int(abstract_row["training_seed"]),
                    "protocol": protocol,
                    "abstract_evaluation_name": abstract_name,
                    "transfer_evaluation_name": transfer_name,
                    "abstract_intercept_success_rate": abstract_success,
                    "pymunk_transfer_success_rate": transfer_success,
                    "transfer_gap_success": abstract_success - transfer_success,
                }
            )
    return pd.DataFrame(rows)


def _seed_interval(values, samples=4000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [None, None]
    if len(values) == 1:
        value = float(values[0])
        return [value, value]
    return bootstrap_interval(values, np.mean, samples, np.random.default_rng(seed))


def build_replication_summary(data, config, protocol="confirmatory"):
    """Evaluate the predeclared Phase-2 confirmation gate at training-seed level."""
    methods = ["mappo_nominal", "mappo_uniform_dr", "mappo_failure_dr"]
    if data.empty:
        data = pd.DataFrame(columns=["method", "training_seed", "evaluation_name"])
    prefix = "" if protocol == "pilot" else protocol + "_"
    names = {
        "abstract_intercept_success": prefix + "abstract_standard",
        "profile_mean_success": prefix + "pymunk_profiles",
        "grid_auc": prefix + "pymunk_robustness",
        "cooperative_success": prefix + "pymunk_cooperation",
    }
    value_columns = {
        "abstract_intercept_success": "canonical_success_rate",
        "profile_mean_success": "mean_profile_success_rate",
        "grid_auc": "normalized_area_under_robustness_curve",
        "cooperative_success": "cooperative_success_rate",
    }
    required_metrics = {
        "abstract_intercept_success",
        "profile_mean_success",
        "grid_auc",
    }
    per_seed = []
    for method in methods:
        method_data = data[data["method"] == method]
        seeds = sorted(int(value) for value in method_data["training_seed"].dropna().unique())
        for training_seed in seeds:
            row = {"method": method, "training_seed": training_seed}
            complete = True
            for metric, evaluation_name in names.items():
                match = method_data[
                    (method_data["training_seed"] == training_seed)
                    & (method_data["evaluation_name"] == evaluation_name)
                ]
                if len(match) != 1 or value_columns[metric] not in match:
                    if metric in required_metrics:
                        complete = False
                    row[metric] = None
                    continue
                value = match.iloc[0].get(value_columns[metric])
                if value is None or pd.isna(value):
                    if metric in required_metrics:
                        complete = False
                    row[metric] = None
                else:
                    row[metric] = float(value)
            if complete:
                per_seed.append(row)

    per_seed_frame = pd.DataFrame(per_seed)
    aggregate = {}
    for method in methods:
        method_rows = (
            per_seed_frame[per_seed_frame["method"] == method]
            if not per_seed_frame.empty
            else pd.DataFrame()
        )
        metric_summary = {}
        for metric in names:
            values = method_rows.get(metric, pd.Series(dtype=float)).dropna().astype(float).to_numpy()
            metric_summary[metric] = {
                "training_seed_count": len(values),
                "mean": float(np.mean(values)) if len(values) else None,
                "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
                "mean_95_ci": _seed_interval(values, seed=7919 + len(values)),
            }
        aggregate[method] = metric_summary

    gate_config = config["evaluation"].get("replication_gate", {})
    minimum_seeds = int(gate_config.get("minimum_training_seeds", 3))
    required_directions = int(gate_config.get("required_direction_seeds", 2))
    maximum_grid_regression = float(gate_config.get("maximum_grid_regression", 0.05))
    maximum_abstract_regression = float(
        gate_config.get("maximum_abstract_intercept_regression", 0.10)
    )
    seed_counts = {
        method: int(
            len(per_seed_frame[per_seed_frame["method"] == method])
            if not per_seed_frame.empty
            else 0
        )
        for method in methods
    }
    enough_seeds = all(count >= minimum_seeds for count in seed_counts.values())

    indexed = (
        per_seed_frame.set_index(["method", "training_seed"])
        if not per_seed_frame.empty
        else pd.DataFrame()
    )
    common_seeds = sorted(
        set.intersection(
            *[
                set(
                    per_seed_frame.loc[
                        per_seed_frame["method"] == method, "training_seed"
                    ].astype(int)
                )
                for method in methods
            ]
        )
        if not per_seed_frame.empty
        else []
    )
    profile_direction_seeds = []
    delay_failure_seeds = []
    for training_seed in common_seeds:
        fdr = indexed.loc[("mappo_failure_dr", training_seed)]
        nominal = indexed.loc[("mappo_nominal", training_seed)]
        uniform = indexed.loc[("mappo_uniform_dr", training_seed)]
        if fdr["profile_mean_success"] > max(
            nominal["profile_mean_success"], uniform["profile_mean_success"]
        ):
            profile_direction_seeds.append(int(training_seed))
        if fdr["grid_auc"] < nominal["grid_auc"] - maximum_grid_regression:
            delay_failure_seeds.append(int(training_seed))

    def mean(method, metric):
        return aggregate[method][metric]["mean"]

    means_available = all(
        mean(method, metric) is not None
        for method in methods
        for metric in ["abstract_intercept_success", "profile_mean_success", "grid_auc"]
    )
    profile_mean_advantage = means_available and mean(
        "mappo_failure_dr", "profile_mean_success"
    ) > max(
        mean("mappo_nominal", "profile_mean_success"),
        mean("mappo_uniform_dr", "profile_mean_success"),
    )
    profile_direction_stable = len(profile_direction_seeds) >= required_directions
    grid_noninferior = means_available and mean("mappo_failure_dr", "grid_auc") >= mean(
        "mappo_nominal", "grid_auc"
    ) - maximum_grid_regression
    abstract_noninferior = means_available and mean(
        "mappo_failure_dr", "abstract_intercept_success"
    ) >= mean("mappo_nominal", "abstract_intercept_success") - maximum_abstract_regression
    delay_failure_replicated = len(delay_failure_seeds) >= required_directions
    checks = {
        "minimum_training_seeds": {
            "passed": bool(enough_seeds),
            "observed": seed_counts,
            "criterion": f"at least {minimum_seeds} complete seeds per method",
        },
        "profile_mean_advantage": {
            "passed": bool(profile_mean_advantage),
            "observed": {
                method: mean(method, "profile_mean_success") for method in methods
            },
            "criterion": "failure-directed mean exceeds both equal-budget controls",
        },
        "profile_direction_stability": {
            "passed": bool(profile_direction_stable),
            "observed": profile_direction_seeds,
            "criterion": f"failure-directed exceeds both controls in at least {required_directions} matched seeds",
        },
        "grid_noninferiority": {
            "passed": bool(grid_noninferior),
            "observed": {
                "failure_directed": mean("mappo_failure_dr", "grid_auc"),
                "nominal": mean("mappo_nominal", "grid_auc"),
            },
            "criterion": f"failure-directed grid AUC no more than {maximum_grid_regression:.3f} below nominal",
        },
        "abstract_intercept_noninferiority": {
            "passed": bool(abstract_noninferior),
            "observed": {
                "failure_directed": mean(
                    "mappo_failure_dr", "abstract_intercept_success"
                ),
                "nominal": mean("mappo_nominal", "abstract_intercept_success"),
            },
            "criterion": f"failure-directed abstract intercept no more than {maximum_abstract_regression:.3f} below nominal",
        },
    }
    confirmation_passed = enough_seeds and all(
        checks[name]["passed"]
        for name in [
            "profile_mean_advantage",
            "profile_direction_stability",
            "grid_noninferiority",
            "abstract_intercept_noninferiority",
        ]
    )
    return {
        "protocol": protocol,
        "complete_training_seed_counts": seed_counts,
        "common_training_seeds": common_seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "checks": checks,
        "confirmation_passed": bool(confirmation_passed),
        "delay_failure_replicated": bool(delay_failure_replicated),
        "delay_failure_seeds": delay_failure_seeds,
        "delay_ablation_authorized": bool(
            enough_seeds
            and profile_mean_advantage
            and profile_direction_stable
            and delay_failure_replicated
        ),
    }


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


def _format_seed_estimate(metric):
    if metric is None or metric.get("mean") is None:
        return "--"
    if int(metric["training_seed_count"]) <= 1:
        return f"{metric['mean']:.3f}"
    return f"{metric['mean']:.3f} $\\pm$ {metric['standard_deviation']:.3f}"


def export_generated_results(data, phase, replication=None):
    """Export suite-specific, seed-aware tables consumed by the LaTeX report."""
    destination = Path("reports/generated_results.tex")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if data.empty:
        content = (
            "% Generated placeholder: no final experiment artifacts were discovered.\n"
            "\\newcommand{\\ResultsAvailable}{false}\n"
            "\\newcommand{\\ResultsStatus}{Final empirical results have not yet been generated.}\n"
        )
    else:
        if replication is None:
            protocol = (
                "confirmatory"
                if data["evaluation_name"].astype(str).str.startswith("confirmatory_").any()
                else "pilot"
            )
            replication = build_replication_summary(
                data, load_config("configs/base.yaml"), protocol=protocol
            )
        protocol = replication["protocol"]
        seed_counts = replication["complete_training_seed_counts"]
        complete_count = min(seed_counts.values()) if seed_counts else 0
        status = (
            "Smoke infrastructure results; not final evidence."
            if phase == "smoke"
            else f"{protocol.title()} evaluation: {complete_count} complete training seed(s) per method; "
            + (
                "the predeclared replication gate passed."
                if replication["confirmation_passed"]
                else "the predeclared replication gate has not passed."
            )
        )
        lines = [
            "% Automatically generated from suite-specific completed run artifacts.",
            "\\newcommand{\\ResultsAvailable}{true}",
            "\\newcommand{\\ResultsStatus}{" + latex_escape(status) + "}",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{lrrrrr}",
            r"\toprule Method & Training seeds & Abstract intercept & Profile mean & Grid AUC & Cooperative success \\",
            "\\midrule",
        ]
        display_names = {
            "mappo_nominal": "Nominal MAPPO",
            "mappo_uniform_dr": "Uniform DR MAPPO",
            "mappo_failure_dr": "Failure-directed MAPPO",
        }
        for method, display_name in display_names.items():
            aggregate = replication["aggregate"].get(method, {})
            lines.append(
                f"{display_name} & {seed_counts.get(method, 0)} & "
                f"{_format_seed_estimate(aggregate.get('abstract_intercept_success'))} & "
                f"{_format_seed_estimate(aggregate.get('profile_mean_success'))} & "
                f"{_format_seed_estimate(aggregate.get('grid_auc'))} & "
                f"{_format_seed_estimate(aggregate.get('cooperative_success'))} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Suite-specific seed-level summaries for the "
                + latex_escape(protocol)
                + " protocol. Values are mean $\\pm$ sample standard deviation when more than one independent training seed is complete. Unlike evaluation suites are never pooled.}",
                "\\label{tab:phase2-suite-results}",
                "\\end{table}",
                "",
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                "\\begin{tabular}{lp{7.0cm}c}",
                r"\toprule Gate & Predeclared criterion & Result \\",
                "\\midrule",
            ]
        )
        gate_names = {
            "minimum_training_seeds": "Independent seeds",
            "profile_mean_advantage": "Primary profile metric",
            "profile_direction_stability": "Direction stability",
            "grid_noninferiority": "Delay--noise non-inferiority",
            "abstract_intercept_noninferiority": "Abstract competence",
        }
        for key, name in gate_names.items():
            check = replication["checks"][key]
            lines.append(
                f"{latex_escape(name)} & {latex_escape(check['criterion'])} & "
                + ("PASS" if check["passed"] else "NOT YET")
                + " \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Automated confirmation gate evaluated across independent training seeds. Episode-level replication within one trained actor is not treated as an independent seed.}",
                "\\label{tab:phase2-paired-results}",
                "\\end{table}",
            ]
        )
        content = "\n".join(lines) + "\n"
    destination.write_text(content, encoding="utf-8")
    return destination


def compare_runs(run_paths=None, phase="final", runs_root="runs", export_report=False):
    if run_paths:
        paths = [Path(path) for path in run_paths]
    else:
        entries = read_completed_manifests(runs_root)
        paths = [Path(entry["run_directory"]) for entry in entries]
    paths = list(dict.fromkeys(path.expanduser().resolve() for path in paths))
    if phase == "smoke":
        paths = [path for path in paths if "smoke" in path.name or "baseline" in path.name]
    elif phase == "final":
        paths = [path for path in paths if "smoke" not in path.name]
    rows = comparison_rows(paths)
    data = pd.DataFrame(rows)
    output_dir = Path(runs_root) / "comparisons" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + phase
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, filename="comparison.log")
    config_path = paths[0] / "resolved_config.yaml" if paths else Path("configs/base.yaml")
    config = load_config(config_path if config_path.is_file() else "configs/base.yaml")
    suite_data = suite_aggregate(data)
    transfer_data = canonical_transfer_rows(data)
    protocol = (
        "confirmatory"
        if not data.empty
        and data["evaluation_name"].astype(str).str.startswith("confirmatory_").any()
        else "pilot"
    )
    replication = build_replication_summary(data, config, protocol=protocol)
    if data.empty:
        pd.DataFrame(columns=["method", "simulator", "success_rate", "mean_return"]).to_csv(
            output_dir / "main_comparison.csv", index=False
        )
        write_json(output_dir / "main_comparison.json", [])
        logger.warning("No compatible completed evaluation summaries were found.")
    else:
        data.to_csv(output_dir / "main_comparison.csv", index=False)
        write_json(output_dir / "main_comparison.json", data.to_dict(orient="records"))
        plot_comparisons(suite_data, transfer_data, output_dir)
    suite_data.to_csv(output_dir / "suite_comparison.csv", index=False)
    write_json(output_dir / "suite_comparison.json", suite_data.to_dict(orient="records"))
    transfer_data.to_csv(output_dir / "canonical_transfer_gaps.csv", index=False)
    write_json(
        output_dir / "canonical_transfer_gaps.json", transfer_data.to_dict(orient="records")
    )
    pd.DataFrame(replication["per_seed"]).to_csv(
        output_dir / "replication_per_seed.csv", index=False
    )
    write_json(output_dir / "replication_summary.json", replication)
    plot_run_histories(paths, output_dir)
    plot_consolidated_profiles(paths, output_dir)
    plot_consolidated_heatmaps(paths, output_dir)
    if export_report:
        export_generated_results(data, phase, replication)
    logger.info("Comparison artifacts written to %s", output_dir)
    return output_dir, data


def plot_comparisons(suite_data, transfer_data, output_dir):
    """Plot suite-separated seed aggregates and explicitly paired transfer gaps."""
    plt = get_pyplot()
    report_figures = Path("reports/figures")
    report_figures.mkdir(parents=True, exist_ok=True)
    if not suite_data.empty:
        evaluations = sorted(suite_data["evaluation_name"].astype(str).unique())
        figure, axes = plt.subplots(
            len(evaluations),
            1,
            figsize=(9.0, max(4.2, 3.2 * len(evaluations))),
            squeeze=False,
        )
        for axis, evaluation_name in zip(axes[:, 0], evaluations, strict=True):
            group = suite_data[suite_data["evaluation_name"] == evaluation_name].sort_values(
                "method"
            )
            error = group["success_rate_std"].fillna(0.0).to_numpy()
            axis.bar(group["method"], group["success_rate_mean"], yerr=error, capsize=3)
            axis.set_title(evaluation_name.replace("_", " "))
            axis.set_ylabel("Success rate")
            axis.set_ylim(0.0, 1.0)
            axis.tick_params(axis="x", rotation=20)
        figure.suptitle("Suite-separated success across independent training seeds")
        figure.tight_layout()
        figure.savefig(Path(output_dir) / "method_success_by_suite.png", dpi=180)
        figure.savefig(report_figures / "method_success_by_suite.png", dpi=180)
        plt.close(figure)
    if not transfer_data.empty:
        grouped = transfer_data.groupby(["protocol", "method"])["transfer_gap_success"].agg(
            ["mean", "std"]
        ).reset_index()
        protocols = sorted(grouped["protocol"].unique())
        figure, axes = plt.subplots(
            1, len(protocols), figsize=(6.2 * len(protocols), 4.3), squeeze=False
        )
        for axis, protocol in zip(axes[0], protocols, strict=True):
            group = grouped[grouped["protocol"] == protocol].sort_values("mean")
            axis.barh(group["method"], group["mean"], xerr=group["std"].fillna(0.0))
            axis.axvline(0.0, color="black", linewidth=0.8)
            axis.set_title(protocol.title())
            axis.set_xlabel("Abstract-intercept minus Pymunk-transfer success")
        figure.suptitle("Canonical zero-shot transfer gaps")
        figure.tight_layout()
        figure.savefig(Path(output_dir) / "canonical_transfer_gap.png", dpi=180)
        figure.savefig(report_figures / "canonical_transfer_gap.png", dpi=180)
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
    preferred_name = "confirmatory_pymunk_profiles"
    fallback_name = "pymunk_profiles"
    has_confirmatory = any(
        (Path(run_path) / "eval" / preferred_name / "episodes.csv").is_file()
        for run_path in run_paths
    )
    evaluation_name = preferred_name if has_confirmatory else fallback_name
    for run_path in run_paths:
        run_path = Path(run_path)
        metadata_path = run_path / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        method = metadata.get("experiment_name", run_path.name)
        episodes_path = run_path / "eval" / evaluation_name / "episodes.csv"
        if not episodes_path.is_file():
            continue
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
    axis.set_title("Robustness success by perturbation profile (" + evaluation_name + ")")
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
    preferred_name = "confirmatory_pymunk_robustness"
    fallback_name = "pymunk_robustness"
    has_confirmatory = any(
        (Path(run_path) / "eval" / preferred_name / "robustness_success_pivot.csv").is_file()
        for run_path in run_paths
    )
    evaluation_name = preferred_name if has_confirmatory else fallback_name
    for run_path in run_paths:
        run_path = Path(run_path)
        metadata_path = run_path / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        method = metadata.get("experiment_name", run_path.name)
        pivot_path = run_path / "eval" / evaluation_name / "robustness_success_pivot.csv"
        if pivot_path.is_file():
            entries.append((method, pd.read_csv(pivot_path, index_col=0)))
    if not entries:
        return
    aggregated_entries = []
    for method in sorted({method for method, _ in entries}):
        pivots = [pivot for entry_method, pivot in entries if entry_method == method]
        stacked = np.stack([pivot.to_numpy(dtype=float) for pivot in pivots])
        aggregate = pivots[0].copy()
        aggregate.iloc[:, :] = np.mean(stacked, axis=0)
        aggregated_entries.append((method, aggregate))
    plt = get_pyplot()
    figure, axes = plt.subplots(
        1,
        len(aggregated_entries),
        figsize=(5.3 * len(aggregated_entries), 4.5),
        squeeze=False,
    )
    image = None
    for axis, (method, pivot) in zip(axes[0], aggregated_entries, strict=True):
        image = axis.imshow(pivot.values, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        axis.set_yticks(range(len(pivot.index)), labels=[f"{float(value):.2f}" for value in pivot.index])
        axis.set_title(method)
        axis.set_xlabel("Action delay (macro steps)")
        axis.set_ylabel("Localization noise std. dev. (m)")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Goal success rate", shrink=0.82)
    figure.suptitle("Seed-mean cross-simulator robustness grids (" + evaluation_name + ")")
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.14, top=0.84, wspace=0.32)
    figure.savefig(Path(output_dir) / "consolidated_robustness_heatmaps.png", dpi=180)
    figure.savefig(Path("reports/figures/consolidated_robustness_heatmaps.png"), dpi=180)
    plt.close(figure)
