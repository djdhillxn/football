"""Concentrated correctness tests for the mandatory research pipeline."""

import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pettingzoo.test import parallel_api_test

from robosoccer.artifacts import (
    prune_local_training_artifacts,
    resolve_run_pointer,
    sync_artifacts_from_drive,
    sync_run_to_drive,
)
from robosoccer.config import apply_overrides, load_config
from robosoccer.diagnostics import audit_action_delay, expected_delayed_actions
from robosoccer.environment import (
    AGENTS,
    AbstractSoccerEnv,
    PymunkSoccerTransferEnv,
    baseline_actions,
    sample_profile_parameters,
)
from robosoccer.evaluation import (
    add_group_success_statistics,
    build_replication_summary,
    canonical_transfer_rows,
    cooperation_probe_initial_state,
    evaluate_baselines,
    export_generated_results,
    flatten_episode_metrics,
    phase1_readiness_audit,
    record_videos,
    suite_aggregate,
    summarize_episodes,
)
from robosoccer.training import (
    FailureDirectedCurriculum,
    PPOTrainer,
    SharedActor,
    ValueNetwork,
    compute_gae,
    run_training,
)


@pytest.fixture(scope="module")
def base_config():
    return load_config("configs/base.yaml")


def small_config(base_config, output_dir=None):
    config = copy.deepcopy(base_config)
    config["experiment"]["name"] = "test_smoke"
    config["experiment"]["tensorboard"] = False
    if output_dir is not None:
        config["experiment"]["output_dir"] = str(output_dir)
    config["randomization"]["mode"] = "none"
    config["environment"]["max_episode_steps"] = 5
    config["environment"]["stationary_truncation_steps"] = 20
    config["train"].update(
        {
            "total_steps": 8,
            "num_envs": 1,
            "rollout_steps": 4,
            "checkpoint_frequency_steps": 4,
            "validation_frequency_steps": 4,
            "validation_episodes": 1,
            "progress_bar": False,
            "device": "cpu",
        }
    )
    config["ppo"].update(
        {
            "update_epochs": 1,
            "actor_minibatch_size": 8,
            "critic_minibatch_size": 4,
        }
    )
    config["video"].update({"width": 320, "height": 180, "fps": 8, "episodes": 1})
    config["evaluation"].update({"bootstrap_samples": 10, "episodes": 1})
    return config


def test_same_seed_identical_initial_abstract_state(base_config):
    first = AbstractSoccerEnv(base_config)
    second = AbstractSoccerEnv(base_config)
    try:
        first.reset(seed=123)
        second.reset(seed=123)
        np.testing.assert_allclose(first.state(), second.state())
        assert first.sampled_parameters == second.sampled_parameters
    finally:
        first.close()
        second.close()


def test_different_seeds_change_initial_state(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        env.reset(seed=1)
        first = env.state().copy()
        env.reset(seed=2)
        second = env.state().copy()
        assert not np.allclose(first, second)
    finally:
        env.close()


def test_abstract_parallel_api(base_config):
    env = AbstractSoccerEnv(base_config)
    parallel_api_test(env, num_cycles=30)


def test_pymunk_parallel_api(base_config):
    env = PymunkSoccerTransferEnv(base_config)
    parallel_api_test(env, num_cycles=30)


def test_pymunk_ball_crosses_sideline_and_terminates(base_config):
    config = copy.deepcopy(base_config)
    config["environment"]["stationary_truncation_steps"] = 100
    env = PymunkSoccerTransferEnv(config)
    try:
        env.reset(seed=17)
        half_width = config["environment"]["field_width"] / 2.0
        env.ball_body.position = (0.0, half_width - 0.02)
        env.ball_body.velocity = (0.0, 4.0)
        _, _, terminations, _, infos = env.step({agent: 6 for agent in AGENTS})
        assert all(terminations.values())
        assert infos[AGENTS[0]]["termination_reason"] == "out_of_bounds"
    finally:
        env.close()


def test_pymunk_players_are_constrained_to_playable_field(base_config):
    env = PymunkSoccerTransferEnv(base_config)
    try:
        env.reset(seed=18)
        half_length = base_config["environment"]["field_length"] / 2.0
        half_width = base_config["environment"]["field_width"] / 2.0
        radius = base_config["environment"]["player_radius"]
        for body in [*env.player_bodies.values(), env.defender_body]:
            body.position = (half_length + 1.0, half_width + 1.0)
            body.velocity = (2.0, 2.0)
        env._constrain_player_bodies()
        env._sync_from_bodies()
        positions = [player["position"] for player in env.players.values()]
        positions.append(env.defender["position"])
        assert all(abs(position[0]) <= half_length - radius for position in positions)
        assert all(abs(position[1]) <= half_width - radius for position in positions)
    finally:
        env.close()


def test_pymunk_kick_impulse_reaches_commanded_velocity(base_config):
    env = PymunkSoccerTransferEnv(base_config)
    try:
        env.reset(seed=19)
        env.ball_body.velocity = (2.0, -1.0)
        env._deliver_kick(np.array([1.0, 0.0]), 4.5)
        np.testing.assert_allclose(
            [env.ball_body.velocity.x, env.ball_body.velocity.y], [4.5, 0.0], atol=1e-8
        )
        env._apply_ball_drag(0.1)
        expected_drag = (
            base_config["environment"]["ball_drag"]
            * base_config["transfer_environment"]["ball_drag_multiplier"]
        )
        expected = 4.5 * (1.0 - expected_drag * 0.1)
        np.testing.assert_allclose(
            [env.ball_body.velocity.x, env.ball_body.velocity.y], [expected, 0.0], atol=1e-8
        )
        assert env.space.damping == pytest.approx(1.0)
    finally:
        env.close()


@pytest.mark.parametrize("environment_class", [AbstractSoccerEnv, PymunkSoccerTransferEnv])
def test_observation_and_state_shapes_are_finite(base_config, environment_class):
    env = environment_class(base_config)
    try:
        observations, _ = env.reset(seed=8)
        assert observations[AGENTS[0]].shape == (env.observation_dimension,)
        assert env.state().shape == (env.state_dimension,)
        assert all(np.isfinite(observation).all() for observation in observations.values())
        assert np.isfinite(env.state()).all()
    finally:
        env.close()


def test_agents_receive_same_shared_reward(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        env.reset(seed=4)
        _, rewards, _, _, _ = env.step({agent: 6 for agent in AGENTS})
        assert rewards[AGENTS[0]] == rewards[AGENTS[1]]
    finally:
        env.close()


def test_goal_detection(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        env.reset(seed=0)
        env.ball["position"] = np.array([base_config["environment"]["field_length"] / 2.0 + 0.01, 0.0])
        _, _, terminations, _, infos = env.step({agent: 6 for agent in AGENTS})
        assert all(terminations.values())
        assert infos[AGENTS[0]]["termination_reason"] == "goal"
    finally:
        env.close()


def test_timeout_truncation(base_config):
    config = copy.deepcopy(base_config)
    config["environment"]["max_episode_steps"] = 1
    config["environment"]["stationary_truncation_steps"] = 20
    env = AbstractSoccerEnv(config)
    try:
        env.reset(seed=5)
        _, _, _, truncations, infos = env.step({agent: 6 for agent in AGENTS})
        assert all(truncations.values())
        assert infos[AGENTS[0]]["termination_reason"] == "timeout"
    finally:
        env.close()


def test_action_delay_queue(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        env.reset(seed=9, options={"sampled_parameters": {"action_latency": 2}})
        applied = []
        for _ in range(3):
            _, _, _, _, infos = env.step({agent: 0 for agent in AGENTS})
            applied.append(infos[AGENTS[0]]["applied_action"])
        assert applied == [6, 6, 0]
    finally:
        env.close()


def test_action_delay_expected_fifo_and_production_audit(base_config, tmp_path):
    assert expected_delayed_actions([0, 1, 2, 3], 2) == [6, 6, 0, 1]
    result, trace = audit_action_delay(base_config, tmp_path, maximum_latency=2)
    assert result["passed"] is True
    assert result["case_count"] == 6
    assert trace and all(row["passed"] for row in trace)
    assert (tmp_path / "action_delay_audit.json").is_file()
    assert (tmp_path / "action_delay_trace.csv").is_file()


@pytest.mark.parametrize("environment_class", [AbstractSoccerEnv, PymunkSoccerTransferEnv])
def test_cooperation_probe_is_pass_needed_and_deterministic(
    base_config, environment_class
):
    first, first_metadata = cooperation_probe_initial_state(base_config, 250000)
    second, second_metadata = cooperation_probe_initial_state(base_config, 250000)
    assert first == second
    assert first_metadata == second_metadata
    env = environment_class(base_config)
    try:
        env.reset(
            seed=250000,
            options={
                "initial_state": first,
                "sampled_parameters": {"defender_speed_multiplier": 0.0},
                "defender_mode": "intercept",
            },
        )
        carrier = first_metadata["probe_carrier"]
        assert env.ball["possession"] == carrier
        assert env._pass_opportunity_carrier() == carrier
        actions = {agent: 6 for agent in AGENTS}
        actions[carrier] = 4
        env.step(actions)
        assert env.metrics["pass_opportunities"] == 1
        assert env.metrics["pass_attempts_on_opportunity"] == 1
        assert env.metrics["pass_attempts"] == 1
    finally:
        env.close()


def test_observation_delay_queue(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        observations, _ = env.reset(seed=11, options={"sampled_parameters": {"observation_latency": 2}})
        initial_ball = observations[AGENTS[0]][13:15].copy()
        env.ball["position"] += np.array([1.0, 0.0])
        observations, _, _, _, _ = env.step({agent: 6 for agent in AGENTS})
        np.testing.assert_allclose(observations[AGENTS[0]][13:15], initial_ball, atol=1e-6)
    finally:
        env.close()


def test_packet_loss_probability_boundaries(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        env.reset(seed=2)
        env.players[AGENTS[1]]["current_action"] = 3
        env.sampled_parameters["packet_loss"] = 0.0
        env._update_communication()
        assert env.delivered_messages[AGENTS[0]]["action"] == 3
        env.players[AGENTS[1]]["current_action"] = 4
        env.sampled_parameters["packet_loss"] = 1.0
        env._update_communication()
        assert env.delivered_messages[AGENTS[0]]["action"] == 3
    finally:
        env.close()


def test_randomized_parameters_stay_in_profile_ranges(base_config):
    rng = np.random.default_rng(7)
    profile = base_config["randomization"]["profiles"]["combined_severe"]
    for _ in range(20):
        sampled = sample_profile_parameters(profile, rng)
        for key, bounds in profile["parameters"].items():
            assert bounds[0] <= sampled[key] <= bounds[1]


def test_parameter_ablation_neutralizes_delay_inside_combined_profile(base_config):
    config = copy.deepcopy(base_config)
    config["randomization"]["mode"] = "uniform"
    config["randomization"]["disabled_parameters"] = ["action_latency"]
    env = AbstractSoccerEnv(config, profile_name="combined_severe")
    try:
        for seed in range(3):
            env.reset(seed=seed)
            assert env.selected_profile == "combined_severe"
            assert env.sampled_parameters["action_latency"] == 0
            assert env.sampled_parameters["localization_noise"] > 0.0
    finally:
        env.close()


def test_failure_directed_probabilities_constraints(base_config):
    config = copy.deepcopy(base_config["curriculum"])
    names = ["a", "b", "c", "d"]
    curriculum = FailureDirectedCurriculum(names, config)
    probabilities = curriculum.update({"a": 0.0, "b": 0.8, "c": 0.9, "d": 1.0}, 1)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    floor = config["uniform_mixture_beta"] / len(names)
    assert min(probabilities.values()) >= floor - 1e-9
    assert max(probabilities.values()) <= config["maximum_profile_probability"] + 1e-9
    assert probabilities["a"] >= probabilities["d"]


def test_curriculum_moves_toward_failing_profile(base_config):
    config = copy.deepcopy(base_config["curriculum"])
    config["maximum_profile_probability"] = 0.8
    curriculum = FailureDirectedCurriculum(["easy", "hard"], config)
    before = curriculum.probabilities["hard"]
    curriculum.update({"easy": 1.0, "hard": 0.0}, 2)
    assert curriculum.probabilities["hard"] > before


def test_role_based_baseline_valid_actions(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        env.reset(seed=42)
        actions = baseline_actions(env, "role_based", {})
        assert set(actions) == set(AGENTS)
        assert all(env.action_space(agent).contains(action) for agent, action in actions.items())
    finally:
        env.close()


def test_role_based_baseline_transfers_carrier_role(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        env.reset(seed=43)
        receiver = AGENTS[1]
        env.ball["position"] = env.players[receiver]["position"].copy()
        env._update_possession()
        memory = {"striker": AGENTS[0]}
        baseline_actions(env, "role_based", memory)
        assert memory["roles"][receiver] == "striker"
    finally:
        env.close()


def test_role_based_baseline_assigns_narrow_loose_ball_recovery(base_config):
    env = AbstractSoccerEnv(base_config)
    try:
        env.reset(seed=45)
        env.ball["position"] = np.array([0.0, 2.2])
        env.ball["velocity"] = np.array([0.0, 2.0])
        env.ball["possession"] = None
        env.players[AGENTS[0]]["position"] = np.array([-1.0, 0.0])
        env.players[AGENTS[1]]["position"] = np.array([1.0, 0.0])
        memory = {}
        actions = baseline_actions(env, "role_based", memory)
        recovery = next(agent for agent, role in memory["roles"].items() if role == "recovery")
        assert actions[recovery] == 0
    finally:
        env.close()


@pytest.mark.parametrize("environment_class", [AbstractSoccerEnv, PymunkSoccerTransferEnv])
def test_episode_fraction_metrics_are_bounded(base_config, environment_class):
    config = small_config(base_config)
    env = environment_class(config)
    try:
        env.reset(seed=44)
        team_return = 0.0
        while env.agents:
            _, rewards, _, _, infos = env.step({agent: 6 for agent in AGENTS})
            team_return += rewards[AGENTS[0]]
        metrics = infos[AGENTS[0]]["episode_metrics"]
        row = flatten_episode_metrics(
            metrics,
            team_return,
            44,
            "hold",
            "abstract" if environment_class is AbstractSoccerEnv else "pymunk",
            env.selected_profile,
            env.defender["mode"],
        )
        assert 0.0 <= row["possession_fraction"] <= 1.0
        assert 0.0 <= row["redundant_chase_fraction"] <= 1.0
        assert 0.0 <= row["invalid_action_fraction"] <= 1.0
    finally:
        env.close()


def test_success_aggregation_distinguishes_profiles_and_defenders():
    rows = []
    for defender_mode, successes in [("stationary_goalie", [True, False]), ("intercept", [True, True])]:
        for success in successes:
            rows.append(
                {
                    "team_return": 1.0 if success else -1.0,
                    "success": success,
                    "time_to_score": 1.0 if success else None,
                    "possession_fraction": 0.5,
                    "redundant_chase_fraction": 0.0,
                    "invalid_action_fraction": 0.0,
                    "attacker_collisions": 0,
                    "termination_reason": "goal" if success else "timeout",
                    "action_switches": 0,
                    "profile": "nominal",
                    "defender_mode": defender_mode,
                }
            )
    summary = summarize_episodes(rows, bootstrap_samples=10, seed=0)
    assert "minimum_profile_success_rate" not in summary
    add_group_success_statistics(summary, pd.DataFrame(rows), "defender_mode", "defender_mode")
    assert summary["minimum_defender_mode_success_rate"] == pytest.approx(0.5)
    assert summary["mean_defender_mode_success_rate"] == pytest.approx(0.75)

    rows[0]["profile"] = "delay"
    rows[1]["profile"] = "delay"
    profile_summary = summarize_episodes(rows, bootstrap_samples=10, seed=0)
    assert profile_summary["minimum_profile_success_rate"] == pytest.approx(0.5)


def test_cooperation_summary_reports_counts_and_opportunity_denominator():
    rows = []
    for attempted, completed, cooperative in [(1, 1, True), (0, 0, False)]:
        rows.append(
            {
                "team_return": 1.0 if cooperative else 0.0,
                "success": cooperative,
                "time_to_score": 2.0 if cooperative else None,
                "pass_attempts": attempted,
                "completed_passes": completed,
                "intercepted_passes": 0,
                "pass_opportunities": 4,
                "pass_attempts_on_opportunity": attempted,
                "receiver_possessions_after_pass": completed,
                "goals_after_completed_pass": int(cooperative),
                "cooperative_probe_success": cooperative,
                "possession_fraction": 0.5,
                "redundant_chase_fraction": 0.0,
                "invalid_action_fraction": 0.0,
                "attacker_collisions": 0,
                "termination_reason": "goal" if cooperative else "timeout",
                "action_switches": 0,
                "profile": "nominal",
                "defender_mode": "intercept",
            }
        )
    summary = summarize_episodes(rows, bootstrap_samples=10, seed=0)
    assert summary["pass_opportunity_count"] == 8
    assert summary["pass_attempts_on_opportunity_count"] == 1
    assert summary["pass_opportunity_action_rate"] == pytest.approx(0.125)
    assert summary["receiver_possession_after_pass_count"] == 1
    assert summary["post_pass_goal_count"] == 1
    assert summary["cooperative_success_rate"] == pytest.approx(0.5)


def _synthetic_confirmation_rows(fdr_grid=0.48):
    values = {
        "mappo_nominal": (0.70, 0.50, 0.50, 0.10),
        "mappo_uniform_dr": (0.60, 0.40, 0.42, 0.20),
        "mappo_failure_dr": (0.65, 0.70, fdr_grid, 0.35),
    }
    rows = []
    for method, metrics in values.items():
        for training_seed in range(3):
            jitter = training_seed * 0.005
            for evaluation_name, simulator, suite, column, value in [
                (
                    "confirmatory_abstract_standard",
                    "abstract",
                    "standard",
                    "canonical_success_rate",
                    metrics[0] + jitter,
                ),
                (
                    "confirmatory_pymunk_profiles",
                    "pymunk",
                    "profiles",
                    "mean_profile_success_rate",
                    metrics[1] + jitter,
                ),
                (
                    "confirmatory_pymunk_robustness",
                    "pymunk",
                    "robustness",
                    "normalized_area_under_robustness_curve",
                    metrics[2] + jitter,
                ),
                (
                    "confirmatory_pymunk_cooperation",
                    "pymunk",
                    "cooperation",
                    "cooperative_success_rate",
                    metrics[3] + jitter,
                ),
            ]:
                row = {
                    "run_dir": f"/{method}/{training_seed}",
                    "method": method,
                    "training_seed": training_seed,
                    "evaluation_name": evaluation_name,
                    "simulator": simulator,
                    "suite": suite,
                    "success_rate": value,
                    "mean_return": value,
                    "canonical_success_rate": value,
                }
                row[column] = value
                rows.append(row)
    return pd.DataFrame(rows)


def test_suite_aggregation_never_pools_different_evaluations():
    data = pd.DataFrame(
        [
            {
                "method": "mappo_nominal",
                "simulator": "pymunk",
                "suite": "transfer",
                "evaluation_name": "pymunk_transfer",
                "training_seed": 0,
                "success_rate": 0.2,
                "mean_return": 0.1,
            },
            {
                "method": "mappo_nominal",
                "simulator": "pymunk",
                "suite": "profiles",
                "evaluation_name": "pymunk_profiles",
                "training_seed": 0,
                "success_rate": 0.8,
                "mean_return": 0.7,
            },
        ]
    )
    aggregate = suite_aggregate(data)
    assert len(aggregate) == 2
    assert set(aggregate["success_rate_mean"]) == {0.2, 0.8}


def test_canonical_transfer_gap_pairs_only_declared_suites():
    data = pd.DataFrame(
        [
            {
                "run_dir": "/run",
                "method": "mappo_nominal",
                "training_seed": 0,
                "evaluation_name": "confirmatory_abstract_standard",
                "canonical_success_rate": 0.7,
            },
            {
                "run_dir": "/run",
                "method": "mappo_nominal",
                "training_seed": 0,
                "evaluation_name": "confirmatory_pymunk_transfer",
                "canonical_success_rate": 0.4,
            },
            {
                "run_dir": "/run",
                "method": "mappo_nominal",
                "training_seed": 0,
                "evaluation_name": "confirmatory_pymunk_profiles",
                "canonical_success_rate": 0.9,
            },
        ]
    )
    paired = canonical_transfer_rows(data)
    assert len(paired) == 1
    assert paired.iloc[0]["transfer_gap_success"] == pytest.approx(0.3)


def test_replication_gate_is_seed_aware_and_controls_delay_ablation(base_config):
    passing = build_replication_summary(_synthetic_confirmation_rows(), base_config)
    assert passing["complete_training_seed_counts"] == {
        "mappo_nominal": 3,
        "mappo_uniform_dr": 3,
        "mappo_failure_dr": 3,
    }
    assert passing["confirmation_passed"] is True
    assert passing["delay_ablation_authorized"] is False

    delay_failure = build_replication_summary(
        _synthetic_confirmation_rows(fdr_grid=0.35), base_config
    )
    assert delay_failure["confirmation_passed"] is False
    assert delay_failure["delay_failure_replicated"] is True
    assert delay_failure["delay_ablation_authorized"] is True


def test_generated_results_are_seed_aware_and_suite_specific(
    base_config, tmp_path, monkeypatch
):
    data = _synthetic_confirmation_rows()
    replication = build_replication_summary(data, base_config)
    monkeypatch.chdir(tmp_path)
    destination = export_generated_results(data, "final", replication)
    contents = destination.read_text(encoding="utf-8")
    assert "Training seeds" in contents
    assert "Cooperative success" in contents
    assert "Suite-specific seed-level summaries" in contents
    assert "\\label{tab:phase2-suite-results}" in contents
    assert "\\label{tab:phase2-paired-results}" in contents


def test_baseline_methods_use_paired_seeds(base_config, tmp_path):
    config = small_config(base_config, tmp_path / "runs")
    run_dir, summary = evaluate_baselines(
        config,
        episodes=2,
        methods=["random", "role_based"],
        source_config="configs/base.yaml",
    )
    with (run_dir / "eval" / "baseline_episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for simulator in ["abstract", "pymunk"]:
        random_seeds = {
            row["seed"] for row in rows if row["method"] == "random" and row["simulator"] == simulator
        }
        role_seeds = {
            row["seed"]
            for row in rows
            if row["method"] == "role_based" and row["simulator"] == simulator
        }
        assert random_seeds == role_seeds
    assert all("minimum_profile_success_rate" not in item for item in summary.values())
    assert all("minimum_defender_mode_success_rate" in item for item in summary.values())


def test_phase1_readiness_audit_has_explicit_baseline_and_learned_gates(base_config):
    baseline = {}
    for method, abstract_success, pymunk_success, redundant in [
        ("random", 0.1, 0.2, 0.3),
        ("double_chase", 0.2, 0.4, 0.7),
        ("role_based", 0.3, 0.6, 0.1),
    ]:
        for simulator, success in [("abstract", abstract_success), ("pymunk", pymunk_success)]:
            baseline[method + "__" + simulator] = {
                "episode_count": base_config["evaluation"]["episodes"],
                "success_rate": success,
                "possession_fraction": 0.2,
                "redundant_chase_fraction": redundant,
                "invalid_action_rate": 0.1,
            }
    # Raise the configured random ceiling for this synthetic example while keeping a wide spread.
    config = copy.deepcopy(base_config)
    config["evaluation"]["phase1_gate"]["maximum_random_pymunk_success"] = 0.25
    abstract = {
        "episode_count": 30,
        "success_rate": 0.5,
        "possession_fraction": 0.2,
        "redundant_chase_fraction": 0.2,
        "invalid_action_rate": 0.1,
        "minimum_defender_mode_success_rate": 0.1,
        "by_defender_mode": {
            mode: {
                "episode_count": base_config["evaluation"]["episodes"],
                "success_rate": 0.8 if mode == "intercept" else 0.2,
            }
            for mode in ["stationary_goalie", "chase_ball", "intercept"]
        },
    }
    transfer = {
        "episode_count": base_config["evaluation"]["episodes"],
        "success_rate": 0.3,
        "possession_fraction": 0.1,
        "redundant_chase_fraction": 0.1,
        "invalid_action_rate": 0.1,
    }
    audit = phase1_readiness_audit(config, baseline, abstract, transfer)
    assert audit["baseline_ready"] is True
    assert audit["phase2_ready"] is True
    baseline["random__pymunk"]["success_rate"] = 0.9
    assert phase1_readiness_audit(config, baseline)["baseline_ready"] is False


def test_actor_and_central_critic_shapes(base_config):
    actor = SharedActor(61, 7, base_config["model"])
    critic = ValueNetwork(66, base_config["model"]["central_critic_hidden_sizes"], base_config["model"])
    assert actor(torch.zeros(3, 61)).shape == (3, 7)
    assert critic(torch.zeros(3, 66)).shape == (3,)


def test_gae_hand_calculation():
    rewards = np.array([[1.0], [1.0]], dtype=np.float32)
    zeros = np.zeros_like(rewards)
    advantages, returns = compute_gae(rewards, zeros, zeros, zeros, zeros, 1.0, 1.0)
    np.testing.assert_allclose(advantages[:, 0], [2.0, 1.0])
    np.testing.assert_allclose(returns[:, 0], [2.0, 1.0])


def test_checkpoint_round_trip_preserves_actor_output(base_config, tmp_path):
    config = small_config(base_config, tmp_path / "runs")
    run_dir = tmp_path / "trainer"
    run_dir.mkdir()
    trainer = PPOTrainer(config, run_dir)
    try:
        observation = torch.randn(2, trainer.observation_size)
        with torch.no_grad():
            expected = trainer.actor(observation).clone()
        checkpoint = trainer.save_checkpoint(run_dir / "checkpoint.pt")
        with torch.no_grad():
            for parameter in trainer.actor.parameters():
                parameter.add_(1.0)
        trainer.load_checkpoint(checkpoint)
        with torch.no_grad():
            actual = trainer.actor(observation)
        torch.testing.assert_close(actual, expected)
    finally:
        trainer.close()


def test_torchscript_export_matches_actor(base_config, tmp_path):
    config = small_config(base_config, tmp_path / "runs")
    run_dir = tmp_path / "trainer"
    run_dir.mkdir()
    trainer = PPOTrainer(config, run_dir)
    try:
        path = trainer.export_actor(run_dir / "actor.ts")
        scripted = torch.jit.load(str(path))
        observation = torch.randn(4, trainer.observation_size)
        with torch.no_grad():
            torch.testing.assert_close(scripted(observation), trainer.actor(observation))
    finally:
        trainer.close()


def test_one_tiny_ppo_update_has_finite_metrics(base_config, tmp_path):
    config = small_config(base_config, tmp_path / "runs")
    run_dir = tmp_path / "trainer"
    run_dir.mkdir()
    trainer = PPOTrainer(config, run_dir)
    try:
        rollout = trainer.collect_rollout()
        metrics = trainer.ppo_update(rollout)
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        trainer.close()


def test_smoke_training_creates_required_artifacts(base_config, tmp_path):
    config = small_config(base_config, tmp_path / "runs")
    run_dir, metadata = run_training(config, source_config="configs/base.yaml")
    assert metadata["status"] == "complete"
    required = [
        "resolved_config.yaml",
        "run_metadata.json",
        "models/final_checkpoint.pt",
        "models/best_checkpoint.pt",
        "models/final_actor.ts",
        "models/best_actor.ts",
        "logs/metrics.csv",
        "videos/render_check.png",
    ]
    assert all((run_dir / relative).is_file() for relative in required)


def test_render_rgb_uint8(base_config):
    config = small_config(base_config)
    env = AbstractSoccerEnv(config)
    try:
        env.reset(seed=1)
        frame = env.render()
        assert frame.dtype == np.uint8
        assert frame.shape == (180, 320, 3)
    finally:
        env.close()


def test_short_video_encoding(base_config, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    config = small_config(base_config)
    output = tmp_path / "video_run"
    output.mkdir()
    paths = record_videos(config, output, "abstract", 1, baseline="random", seed=3)
    assert len(paths) == 1
    assert paths[0].is_file() and paths[0].stat().st_size > 0


def test_configuration_inheritance_and_yaml_overrides():
    config = load_config(
        "configs/mappo_failure_dr.yaml",
        ["train.total_steps=123", "randomization.disabled_families=[action_delay]", "video.episodes=null"],
    )
    assert config["train"]["total_steps"] == 123
    assert config["randomization"]["disabled_families"] == ["action_delay"]
    assert config["video"]["episodes"] is None


def test_apply_override_parses_boolean_and_float(base_config):
    config = apply_overrides(base_config, ["experiment.tensorboard=false", "ppo.gamma=0.9"])
    assert config["experiment"]["tensorboard"] is False
    assert config["ppo"]["gamma"] == pytest.approx(0.9)


def test_circular_configuration_inheritance_is_clear(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("inherits: second.yaml\n", encoding="utf-8")
    second.write_text("inherits: first.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular configuration inheritance"):
        load_config(first)


def test_metadata_is_valid_json_after_smoke(base_config, tmp_path):
    config = small_config(base_config, tmp_path / "runs")
    run_dir, _ = run_training(config, source_config="configs/base.yaml")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["output_artifact_paths"]["environment_steps"] >= 8


def test_global_state_optional_perturbations(base_config):
    config = copy.deepcopy(base_config)
    config["observations"]["expose_perturbations_to_critic"] = True
    env = AbstractSoccerEnv(config)
    try:
        env.reset(seed=1)
        assert env.state_dimension > env.base_state_dimension
        assert env.state().shape == (env.state_dimension,)
    finally:
        env.close()


def test_run_pointer_created(base_config, tmp_path):
    config = small_config(base_config, tmp_path / "runs")
    run_dir, _ = run_training(config, source_config="configs/base.yaml")
    pointer = Path(config["experiment"]["output_dir"]) / "latest_test_smoke.txt"
    assert pointer.is_file()
    assert Path(pointer.read_text(encoding="utf-8").strip()) == run_dir


def write_test_run(run_dir, experiment_name, status):
    run_dir.mkdir(parents=True)
    metadata = {
        "status": status,
        "experiment_name": experiment_name,
        "algorithm": "mappo",
        "seed": 0,
        "run_directory": str(run_dir),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "train.log").write_text(f"{status}\n", encoding="utf-8")
    return metadata


def test_drive_pull_merges_finished_runs_and_generated_artifacts(tmp_path):
    drive_project = tmp_path / "drive" / "RobotSoccerTransfer"
    drive_runs = drive_project / "runs"
    complete_run = drive_runs / "20260101_000001_alpha_mappo_seed0"
    write_test_run(complete_run, "alpha", "complete")
    (complete_run / "models").mkdir()
    (complete_run / "models" / "final_actor.pt").write_bytes(b"model")
    (complete_run / "checkpoints").mkdir()
    (complete_run / "checkpoints" / "checkpoint.ckpt").write_bytes(b"checkpoint")
    (complete_run / "exported_actor.pth").write_bytes(b"weights")
    (complete_run / "replay.pkl").write_bytes(b"replay")
    (complete_run / "logs" / "events.out.tfevents.test").write_bytes(b"tensorboard")
    (complete_run / "resolved_config.yaml").write_text("seed: 0\n", encoding="utf-8")
    (complete_run / "metrics.json").write_text("{}", encoding="utf-8")
    (complete_run / "plots").mkdir()
    (complete_run / "plots" / "curve.png").write_bytes(b"plot")
    (complete_run / "videos").mkdir()
    (complete_run / "videos" / "episode.mp4").write_bytes(b"video")
    write_test_run(drive_runs / "20260101_000002_failed_mappo_seed0", "failed", "failed")
    write_test_run(drive_runs / "20260101_000003_running_mappo_seed0", "running", "running")
    (drive_runs / "orphan_weights.pt").write_bytes(b"orphan")
    (drive_runs / "invalid_folder").mkdir(parents=True)
    comparison = drive_project / "comparisons" / "comparison_a"
    comparison.mkdir(parents=True)
    (comparison / "summary.json").write_text("{}", encoding="utf-8")
    reports = drive_project / "reports"
    reports.mkdir()
    (reports / "main.tex").write_text("stale source", encoding="utf-8")
    (reports / "generated_results.tex").write_text("stale generated", encoding="utf-8")
    os.utime(reports / "generated_results.tex", (1, 1))

    repository = tmp_path / "repository"
    (repository / "reports").mkdir(parents=True)
    (repository / "reports" / "main.tex").write_text("authoritative", encoding="utf-8")
    (repository / "reports" / "generated_results.tex").write_text(
        "newer generated", encoding="utf-8"
    )
    os.utime(repository / "reports" / "generated_results.tex", (2, 2))
    result = sync_artifacts_from_drive(drive_project, repository)

    local_runs = repository / "runs"
    assert (local_runs / "20260101_000001_alpha_mappo_seed0" / "logs" / "train.log").is_file()
    local_complete = local_runs / "20260101_000001_alpha_mappo_seed0"
    assert not (local_complete / "models").exists()
    assert not (local_complete / "checkpoints").exists()
    assert not (local_complete / "exported_actor.pth").exists()
    assert not (local_complete / "replay.pkl").exists()
    assert (local_complete / "logs" / "events.out.tfevents.test").is_file()
    assert (local_complete / "resolved_config.yaml").is_file()
    assert (local_complete / "metrics.json").is_file()
    assert (local_complete / "plots" / "curve.png").is_file()
    assert (local_complete / "videos" / "episode.mp4").is_file()
    assert (local_runs / "20260101_000002_failed_mappo_seed0" / "logs" / "train.log").is_file()
    assert not (local_runs / "20260101_000003_running_mappo_seed0").exists()
    assert not (local_runs / "orphan_weights.pt").exists()
    assert (local_runs / "comparisons" / "comparison_a" / "summary.json").is_file()
    assert (repository / "reports" / "main.tex").read_text(encoding="utf-8") == "authoritative"
    assert (repository / "reports" / "generated_results.tex").read_text(
        encoding="utf-8"
    ) == "newer generated"
    pointer_text = (local_runs / "latest_alpha.txt").read_text(encoding="utf-8").strip()
    assert pointer_text == "runs/20260101_000001_alpha_mappo_seed0"
    assert resolve_run_pointer(
        local_runs / "latest_alpha.txt", repository
    ) == (local_runs / "20260101_000001_alpha_mappo_seed0").resolve()
    manifest = (local_runs / "experiment_manifest.jsonl").read_text(encoding="utf-8")
    assert '"status": "complete"' in manifest
    assert '"status": "failed"' in manifest
    assert result["skipped_running"] == ["20260101_000003_running_mappo_seed0"]
    assert result["skipped_invalid"] == ["invalid_folder"]
    assert result["training_artifacts_included"] is False

    cached = sync_artifacts_from_drive(drive_project, repository)
    assert "20260101_000001_alpha_mappo_seed0" in cached["skipped_unchanged"]


def test_drive_pull_can_restore_full_training_artifacts_for_colab(tmp_path):
    drive_project = tmp_path / "drive" / "RobotSoccerTransfer"
    run_dir = drive_project / "runs" / "20260101_000001_alpha_mappo_seed0"
    write_test_run(run_dir, "alpha", "complete")
    (run_dir / "models").mkdir()
    (run_dir / "models" / "final_actor.pt").write_bytes(b"model")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "checkpoint.ckpt").write_bytes(b"checkpoint")
    (run_dir / "replay.pkl").write_bytes(b"replay")

    repository = tmp_path / "repository"
    result = sync_artifacts_from_drive(
        drive_project,
        repository,
        include_training_artifacts=True,
    )

    local_run = repository / "runs" / run_dir.name
    assert (local_run / "models" / "final_actor.pt").is_file()
    assert (local_run / "checkpoints" / "checkpoint.ckpt").is_file()
    assert (local_run / "replay.pkl").is_file()
    assert result["training_artifacts_included"] is True

    prune_local_training_artifacts(repository / "runs")
    restored = sync_artifacts_from_drive(
        drive_project,
        repository,
        include_training_artifacts=True,
    )
    assert run_dir.name not in restored["skipped_unchanged"]
    assert (local_run / "models" / "final_actor.pt").is_file()
    assert (local_run / "checkpoints" / "checkpoint.ckpt").is_file()


def test_prune_local_training_artifacts_preserves_analysis_files(tmp_path):
    local_runs = tmp_path / "runs"
    run_dir = local_runs / "20260101_000001_alpha_mappo_seed0"
    write_test_run(run_dir, "alpha", "complete")
    (run_dir / "models").mkdir()
    (run_dir / "models" / "final_actor.pt").write_bytes(b"model")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "checkpoint.pth").write_bytes(b"checkpoint")
    (run_dir / "replay.pkl").write_bytes(b"replay")
    (local_runs / "orphan_weights.pt").write_bytes(b"orphan")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "videos").mkdir()
    (run_dir / "videos" / "episode.mp4").write_bytes(b"video")

    preview = prune_local_training_artifacts(local_runs, dry_run=True)
    assert preview == {
        "removed_directories": 2,
        "removed_files": 4,
        "reclaimed_bytes": 27,
    }
    assert (run_dir / "models" / "final_actor.pt").is_file()

    result = prune_local_training_artifacts(local_runs)
    assert result == preview
    assert not (run_dir / "models").exists()
    assert not (run_dir / "checkpoints").exists()
    assert not (run_dir / "replay.pkl").exists()
    assert not (local_runs / "orphan_weights.pt").exists()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "videos" / "episode.mp4").is_file()


def test_drive_push_writes_portable_pointer_and_refreshes_same_size_metadata(tmp_path):
    drive_project = tmp_path / "drive" / "RobotSoccerTransfer"
    drive_project.mkdir(parents=True)
    run_dir = tmp_path / "runs" / "20260101_000001_alpha_mappo_seed0"
    metadata = write_test_run(run_dir, "alpha", "complete")

    sync_run_to_drive(run_dir, drive_project)
    destination = drive_project / "runs" / run_dir.name
    assert (drive_project / "runs" / "latest_alpha.txt").read_text(encoding="utf-8") == (
        f"runs/{run_dir.name}\n"
    )

    metadata["experiment_name"] = "bravo"
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    sync_run_to_drive(run_dir, drive_project)
    persisted = json.loads((destination / "run_metadata.json").read_text(encoding="utf-8"))
    assert persisted["experiment_name"] == "bravo"
    assert (drive_project / "runs" / "latest_bravo.txt").is_file()


def test_drive_push_rejects_running_run(tmp_path):
    drive_project = tmp_path / "drive" / "RobotSoccerTransfer"
    drive_project.mkdir(parents=True)
    run_dir = tmp_path / "runs" / "20260101_000001_running_mappo_seed0"
    write_test_run(run_dir, "running", "running")
    with pytest.raises(RuntimeError, match="unfinished run"):
        sync_run_to_drive(run_dir, drive_project)
