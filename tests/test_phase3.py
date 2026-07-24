"""Correctness tests for the gated recurrent Phase 3 extension."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from robosoccer.artifacts import resolve_run_pointer
from robosoccer.config import load_config
from robosoccer.phase3 import (
    PHASE3_AGENTS,
    Phase3EnvironmentBatch,
    make_phase3_environment,
    run_stage_r_reward_invariants,
)
from robosoccer.recurrent import (
    CompetenceConstrainedCurriculum,
    RecurrentCentralCritic,
    RecurrentMAPPOTrainer,
    RecurrentSharedActor,
    _cpu_rng_state,
)
from robosoccer.training import compute_gae
from robosoccer.utils import RunningMeanStd
from scripts.evaluate_phase3 import evaluate
from scripts.evaluate_phase3_gates import compact_result_for_console, gate_b_r
from scripts.record_phase3_video import (
    manifest_record_key,
    merge_manifest_records,
    record_phase3_videos,
)


@pytest.fixture(scope="module")
def phase3_config():
    return load_config("configs/phase3_smoke.yaml")


def stage_r_test_config(base, warm_start_run_id="synthetic_stage_d"):
    config = copy.deepcopy(base)
    config["experiment"]["name"] = "phase3_stage_r_test"
    config["phase3"]["active_stage"] = "stage_r"
    config["phase3"]["mode"] = "nominal"
    config["phase3"]["require_calibration"] = False
    config["phase3"]["reward_schema_version"] = 2
    config["phase3"]["stages"]["stage_r"] = {
        "target_steps": 128,
        "match_mode": True,
        "scenarios": [
            "phase3_2v2_open",
            "phase3_2v2_pass_required",
            "phase3_3v2_open",
            "phase3_3v2_press",
        ],
        "probabilities": [0.50, 0.20, 0.15, 0.15],
        "defender_styles": ["lane_block", "predictive", "zonal", "press"],
        "defender_probabilities": [0.35, 0.35, 0.20, 0.10],
    }
    config["phase3"]["stage_r"] = {
        "warm_start_run_id": warm_start_run_id,
        "warm_start_checkpoint": "best_nominal",
        "training_episode_seed_base": 1000000000,
        "training_steps": 128,
        "checkpoint_selection": {
            "pass_required_cooperation_floor": 0.50,
            "3v2_open_success_floor": 0.65,
        },
        "gate_b_r": {
            "minimum_pass_required_cooperation": 0.55,
            "minimum_3v2_open_success": 0.70,
            "minimum_worst_style_success": 0.25,
            "minimum_lane_predictive_mean": 0.40,
            "minimum_success_failure_return_gap": 5.0,
        },
    }
    config["phase3_reward"]["controlled_reception"] = 0.0
    config["ppo"]["actor_learning_rate"] = 1e-4
    config["ppo"]["critic_learning_rate"] = 1e-4
    config["ppo"]["actor_min_learning_rate"] = 3e-5
    config["ppo"]["critic_min_learning_rate"] = 3e-5
    config["train"]["total_steps"] = 128
    config["evaluation"]["seed_bases"].update(
        {
            "phase3_stage_r_r0_audit": 360000,
            "phase3_stage_r_validation": 365000,
            "phase3_gate_b_r": 370000,
        }
    )
    return config


def stage_d_checkpoint(base, root):
    config = copy.deepcopy(base)
    config["phase3"]["active_stage"] = "stage_d"
    config["phase3"]["mode"] = "nominal"
    config["phase3"]["require_calibration"] = False
    _trainer_directories(root)
    trainer = RecurrentMAPPOTrainer(config, root)
    try:
        trainer.observation_rms.mean.fill(1.25)
        trainer.state_rms.mean.fill(-0.75)
        return trainer.save_checkpoint(root / "models" / "best_nominal_checkpoint.pt")
    finally:
        trainer.close()


def test_phase3_gate_console_summary_preserves_saved_episode_rows():
    result = {
        "gate": "C",
        "nominal": {"episode_rows": {"nominal": [{"success": 1}]}},
        "cc_fdr": {"episode_rows": {"combined": [{"success": 0}, {"success": 1}]}},
        "passed": False,
    }
    compact = compact_result_for_console(result)
    assert "episode_rows" in result["nominal"]
    assert "episode_rows" not in compact["nominal"]
    assert compact["nominal"]["episode_counts"] == {"nominal": 1}
    assert compact["cc_fdr"]["episode_counts"] == {"combined": 2}


@pytest.mark.parametrize("simulator", ["abstract", "pymunk"])
@pytest.mark.parametrize(
    "scenario",
    [
        "phase3_2v2_open",
        "phase3_2v2_pass_required",
        "phase3_3v2_open",
        "phase3_3v2_press",
    ],
)
def test_phase3_resets_are_finite(phase3_config, simulator, scenario):
    env = make_phase3_environment(
        phase3_config, simulator=simulator, scenario=scenario
    )
    try:
        observations, infos = env.reset(seed=9)
        assert len(observations) == int(env.scenario["attackers"])
        assert all(np.isfinite(value).all() for value in observations.values())
        assert np.isfinite(env.state()).all()
        assert all(info["scenario"] == scenario for info in infos.values())
    finally:
        env.close()


def test_phase3_fixed_padding_and_presence_masks(phase3_config):
    shapes = []
    masks = []
    for scenario in ["phase3_2v2_open", "phase3_3v2_open"]:
        env = make_phase3_environment(phase3_config, scenario=scenario)
        try:
            observations, infos = env.reset(seed=10)
            shapes.append(next(iter(observations.values())).shape)
            masks.append(next(iter(infos.values()))["active_agent_mask"])
        finally:
            env.close()
    assert shapes == [(112,), (112,)]
    assert masks == [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]


def test_phase3_deterministic_entity_ordering(phase3_config):
    first = make_phase3_environment(phase3_config, scenario="phase3_3v2_open")
    second = make_phase3_environment(phase3_config, scenario="phase3_3v2_open")
    try:
        first.reset(seed=11)
        second.reset(seed=11)
        assert list(first.players) == PHASE3_AGENTS
        np.testing.assert_allclose(first.state(), second.state())
    finally:
        first.close()
        second.close()


def test_phase3_action_masks_invalid_shoot_and_pass(phase3_config):
    env = make_phase3_environment(phase3_config, scenario="phase3_2v2_open")
    try:
        env.reset(seed=12)
        carrier = env.ball["possession"]
        noncarrier = next(agent for agent in env.active_agents if agent != carrier)
        assert env.action_mask(carrier)[3] == 1
        assert env.action_mask(carrier)[4] == 1
        assert env.action_mask(noncarrier)[3] == 0
        assert env.action_mask(noncarrier)[4] == 0
        actions = {agent: 6 for agent in env.active_agents}
        actions[noncarrier] = 4
        _, _, _, _, _ = env.step(actions)
        assert env.metrics["invalid_action_requests"] == 1
        carrier = env.ball["possession"]
        if carrier is not None:
            receiver = next(agent for agent in env.active_agents if agent != carrier)
            env.players[receiver]["position"][1] = (
                phase3_config["environment"]["field_width"] / 2 - 0.01
            )
            env.players[receiver]["velocity"][1] = 1.0
            assert env.action_mask(carrier)[4] == 0
    finally:
        env.close()


def test_phase3_action_latency_fifo(phase3_config):
    env = make_phase3_environment(
        phase3_config, scenario="phase3_2v2_open", profile_name="delay_high"
    )
    try:
        env.reset(seed=13, options={"sampled_parameters": {"action_latency": 2}})
        carrier = env.ball["possession"]
        actions = {agent: 0 for agent in env.active_agents}
        actions[carrier] = 3
        env.step(actions)
        assert env.players[carrier]["executed_action"] == 6
        env.step(actions)
        assert env.players[carrier]["executed_action"] == 6
        env.step(actions)
        assert env.players[carrier]["executed_action"] == 3
    finally:
        env.close()


def test_phase3_observation_and_communication_delay(phase3_config):
    env = make_phase3_environment(
        phase3_config, scenario="phase3_2v2_open"
    )
    try:
        observations, _ = env.reset(
            seed=131,
            options={
                "sampled_parameters": {
                    "observation_latency": 1,
                    "communication_latency": 1,
                    "packet_loss": 0.0,
                }
            },
        )
        agent = env.active_agents[0]
        initial = observations[agent].copy()
        env.players[agent]["position"][1] += 0.5
        delayed = env._observations()[agent]
        np.testing.assert_allclose(delayed, initial)
        env.step({name: 6 for name in env.active_agents})
        assert env.players[agent]["message"][3] == 0.0
        env.step({name: 6 for name in env.active_agents})
        assert env.players[agent]["message"][3] == 1.0
    finally:
        env.close()


def test_recurrent_network_hidden_shapes_and_masks(phase3_config):
    recurrent = phase3_config["phase3"]["recurrent"]
    actor = RecurrentSharedActor(112, 7, phase3_config["model"], recurrent)
    critic = RecurrentCentralCritic(102, phase3_config["model"], recurrent)
    actor_hidden = torch.zeros(1, 6, recurrent["hidden_size"])
    critic_hidden = torch.zeros(1, 2, recurrent["hidden_size"])
    logits, next_actor = actor(torch.zeros(6, 112), actor_hidden)
    values, next_critic = critic(torch.zeros(2, 102), critic_hidden)
    assert logits.shape == (6, 7)
    assert values.shape == (2,)
    assert next_actor.shape == actor_hidden.shape
    assert next_critic.shape == critic_hidden.shape


def _trainer_directories(path):
    for relative in ["models", "checkpoints", "logs", "videos"]:
        (path / relative).mkdir(parents=True, exist_ok=True)


def test_recurrent_rollout_sequence_masks_and_finite_update(phase3_config, tmp_path):
    _trainer_directories(tmp_path)
    trainer = RecurrentMAPPOTrainer(phase3_config, tmp_path)
    try:
        rollout = trainer.collect_rollout()
        assert rollout["observations"].shape == (8, 2, 3, 112)
        assert rollout["actor_hidden"].shape == (8, 2, 3, 64)
        assert rollout["continuations"].shape == (8, 2)
        inactive = rollout["valid_agents"] < 0.5
        assert np.all(rollout["actor_hidden"][inactive] == 0.0)
        actor_entry = next(
            entry for entry in trainer._chunk_indices(rollout) if entry[2] > 0
        )
        actor_batch = trainer._actor_minibatch(rollout, [actor_entry])
        burn_in = phase3_config["phase3"]["recurrent"]["burn_in_steps"]
        assert actor_batch["valid"][:burn_in].sum() == 0.0
        assert actor_batch["valid"][burn_in:].sum() > 0.0
        critic_entry = next(
            entry
            for entry in trainer._chunk_indices(rollout, actor=False)
            if entry[1] > 0
        )
        critic_batch = trainer._critic_minibatch(rollout, [critic_entry])
        assert critic_batch["valid"][:burn_in].sum() == 0.0
        assert critic_batch["valid"][burn_in:].sum() > 0.0
        metrics = trainer.ppo_update(rollout)
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        trainer.close()


def test_controlled_reception_and_no_repeated_reward_farming(phase3_config):
    env = make_phase3_environment(
        phase3_config, scenario="phase3_2v2_pass_required"
    )
    try:
        env.reset(seed=14)
        passer, receiver = env.active_agents
        env.ball["possession"] = None
        env.last_possessor = passer
        env.pending_pass = {
            "passer": passer,
            "receiver": receiver,
            "start_x": float(env.players[receiver]["position"][0] - 0.5),
            "valid": True,
            "steps": 1,
        }
        env.ball["position"] = env.players[receiver]["position"].copy()
        env.ball["velocity"] = np.zeros(2)
        env._event_rewards = {}
        env._update_possession()
        assert env.metrics["completed_receptions"] == 1
        assert env.metrics["pass_and_advance"] == 1
        assert env._event_rewards["pass_advance"] > 0.0
        assert env.cooperative_sequence is True
        assert env._event_rewards["controlled_reception"] == 1.0
        env._update_possession()
        assert env.metrics["completed_receptions"] == 1
    finally:
        env.close()


def test_selecting_pass_has_no_direct_reward_component(phase3_config):
    env = make_phase3_environment(phase3_config, scenario="phase3_2v2_open")
    try:
        env.reset(seed=15)
        carrier = env.ball["possession"]
        actions = {agent: 5 for agent in env.active_agents}
        actions[carrier] = 4
        env.step(actions)
        assert "pass" in env._event_rewards
        assert "controlled_reception" not in env._event_rewards
        assert phase3_config["phase3_reward"].get("pass", 0.0) == 0.0
    finally:
        env.close()


def test_stage_r_reward_frontier_and_circulation_invariants(phase3_config):
    result = run_stage_r_reward_invariants(stage_r_test_config(phase3_config))
    assert result["passed"] is True
    checks = result["checks"]
    assert checks["controlled_reception_reward_zero"]["observed"] == 0.0
    assert checks["first_forward_frontier_positive"]["observed"] > 0.0
    assert checks["backward_reception_no_progress"]["observed"] == 0.0
    assert checks["aba_no_repeated_progress"]["passed"] is True
    assert checks["revisited_frontier_no_progress"]["observed"] == 0.0
    assert checks["new_frontier_incremental_only"]["passed"] is True
    assert checks["turnover_resets_chain"]["passed"] is True
    assert checks["restart_resets_chain"]["passed"] is True
    assert checks["pass_to_goal_fires_once"]["observed"] == 1
    assert checks["pass_without_goal_has_no_pass_to_goal"]["observed"] == 0
    assert checks["circulation_reward_does_not_scale"]["passed"] is True


def test_stage_r_config_preserves_historical_reward_schema():
    historical = load_config("configs/phase3_recurrent_nominal.yaml")
    stage_r = load_config("configs/phase3_stage_r.yaml")
    assert historical["phase3"].get("reward_schema_version", 1) == 1
    assert historical["phase3_reward"]["controlled_reception"] == pytest.approx(0.35)
    assert stage_r["phase3"]["reward_schema_version"] == 2
    assert stage_r["phase3_reward"]["controlled_reception"] == 0.0
    assert stage_r["phase3"]["stages"]["stage_r"]["probabilities"] == [
        0.50,
        0.20,
        0.15,
        0.15,
    ]
    assert stage_r["phase3"]["stages"]["stage_r"]["defender_probabilities"] == [
        0.35,
        0.35,
        0.20,
        0.10,
    ]


@pytest.mark.parametrize("simulator", ["abstract", "pymunk"])
@pytest.mark.parametrize(
    "style", ["lane_block", "predictive", "zonal", "press"]
)
def test_forced_defender_style_is_exact(phase3_config, simulator, style):
    env = make_phase3_environment(
        phase3_config,
        simulator=simulator,
        scenario="phase3_2v2_open",
        defender_style=style,
    )
    try:
        _, infos = env.reset(seed=220)
        assert env.defender_style == style
        assert all(info["defender_style"] == style for info in infos.values())
    finally:
        env.close()


def test_evaluation_row_records_forced_style(phase3_config):
    env = make_phase3_environment(phase3_config)
    actor = RecurrentSharedActor(
        env.observation_dimension,
        env.action_size,
        phase3_config["model"],
        phase3_config["phase3"]["recurrent"],
    )
    normalizer = RunningMeanStd((env.observation_dimension,))
    env.close()
    rows = evaluate(
        phase3_config,
        actor,
        normalizer,
        torch.device("cpu"),
        "abstract",
        "phase3_2v2_open",
        1,
        361900,
        "nominal",
        defender_style="predictive",
        seed_category="audit",
        policy_run_id="synthetic",
        checkpoint="synthetic.pt",
    )
    assert rows[0]["defender_style"] == "predictive"
    assert rows[0]["requested_defender_style"] == "predictive"
    assert rows[0]["seed_category"] == "audit"


def test_gate_b_r_cannot_silently_use_mixed_style(
    phase3_config, tmp_path, monkeypatch
):
    config = stage_r_test_config(phase3_config)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"synthetic")

    def fake_load_policy(run_dir, checkpoint_name, device):
        return config, object(), object(), checkpoint

    def fake_evaluate(
        config_value,
        actor,
        normalizer,
        device,
        simulator,
        scenario,
        episodes,
        seed_base,
        profile,
        defender_style="mixed",
        **kwargs,
    ):
        if defender_style != "mixed":
            return [{"defender_style": "mixed"}]
        return [
            {
                "defender_style": "lane_block",
                "success": 1,
                "cooperative_success": 1,
                "team_return": 10.0,
                "completed_receptions": 1,
                "valid_pass_attempts": 1,
                "meaningful_action_count": 3,
                "time_to_score": 10.0,
            }
        ]

    monkeypatch.setattr(
        "scripts.evaluate_phase3_gates.load_policy", fake_load_policy
    )
    monkeypatch.setattr("scripts.evaluate_phase3_gates.evaluate", fake_evaluate)
    with pytest.raises(RuntimeError, match="fixed-style"):
        gate_b_r(
            tmp_path,
            "best_stage_r",
            1,
            370000,
            torch.device("cpu"),
            tmp_path / "unused.json",
        )


def test_gate_b_r_uses_declared_nonoverlapping_seed_blocks(
    phase3_config, tmp_path, monkeypatch
):
    config = stage_r_test_config(phase3_config)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"synthetic")
    observed = []

    def fake_load_policy(run_dir, checkpoint_name, device):
        return config, object(), object(), checkpoint

    def fake_evaluate(
        config_value,
        actor,
        normalizer,
        device,
        simulator,
        scenario,
        episodes,
        seed_base,
        profile,
        defender_style="mixed",
        **kwargs,
    ):
        observed.append((seed_base, simulator, scenario, defender_style))
        return [
            {
                "defender_style": (
                    "lane_block" if defender_style == "mixed" else defender_style
                ),
                "success": 1,
                "cooperative_success": 1,
                "team_return": 10.0,
                "completed_receptions": 1,
                "valid_pass_attempts": 1,
                "meaningful_action_count": 3,
                "time_to_score": 10.0,
            }
        ]

    invariants = tmp_path / "reward_invariants.json"
    invariants.write_text(
        json.dumps({"reward_schema_version": 2, "passed": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.evaluate_phase3_gates.load_policy", fake_load_policy
    )
    monkeypatch.setattr("scripts.evaluate_phase3_gates.evaluate", fake_evaluate)
    result = gate_b_r(
        tmp_path,
        "best_stage_r",
        1,
        370000,
        torch.device("cpu"),
        invariants,
    )
    assert [entry[0] for entry in observed] == [
        370000 + 100 * index for index in range(14)
    ]
    assert len({entry[0] for entry in observed}) == 14
    assert result["seed_block_size"] == 100


def test_collision_metric_documents_overlap_step_semantics(phase3_config):
    env = make_phase3_environment(phase3_config, scenario="phase3_2v2_open")
    try:
        env.reset(seed=221)
        snapshot = env.metrics_snapshot()
        assert (
            snapshot["collision_counting_semantics"]
            == "macro_action_decisions_with_attacker_overlap"
        )
        assert phase3_config["phase3_reward"]["collision_penalty"] == pytest.approx(
            -0.02
        )
    finally:
        env.close()


@pytest.mark.parametrize("simulator", ["abstract", "pymunk"])
def test_phase3_defenders_stay_legal_and_speed_bounded(phase3_config, simulator):
    env = make_phase3_environment(
        phase3_config, simulator=simulator, scenario="phase3_3v2_press"
    )
    try:
        env.reset(seed=16)
        headings = {
            name: record["heading"] for name, record in env.defenders.items()
        }
        env.step({agent: 6 for agent in env.active_agents})
        maximum_turn = (
            phase3_config["phase3"]["defender_turn_rate"]
            * phase3_config["environment"]["dt"]
            * phase3_config["environment"]["macro_action_repeat"]
        )
        for name, record in env.defenders.items():
            difference = abs(
                (record["heading"] - headings[name] + np.pi) % (2 * np.pi) - np.pi
            )
            assert difference <= maximum_turn + 1e-6
        for _ in range(5):
            env.step({agent: 6 for agent in env.active_agents})
        half_length = phase3_config["environment"]["field_length"] / 2
        half_width = phase3_config["environment"]["field_width"] / 2
        radius = phase3_config["environment"]["player_radius"]
        for record in env.defenders.values():
            assert abs(record["position"][0]) <= half_length - radius + 1e-6
            assert abs(record["position"][1]) <= half_width - radius + 1e-6
            assert np.linalg.norm(record["velocity"]) <= 1.15 + 1e-5
        goalie = env.defenders["goalie"]
        assert goalie["position"][0] >= half_length - 1.20 - 1e-6
        assert abs(goalie["position"][1]) <= phase3_config["environment"]["goal_width"]
    finally:
        env.close()


def test_match_clock_score_restart_and_hidden_boundary_semantics(phase3_config):
    config = copy.deepcopy(phase3_config)
    config["phase3"]["match_mode"] = True
    config["phase3"]["match_seconds"] = 12.0
    config["phase3"]["minimum_goal_seconds"] = 0.0
    env = make_phase3_environment(config, scenario="phase3_2v2_open")
    try:
        env.reset(seed=17)
        env.cooperative_sequence = True
        env.ball["possession"] = None
        env.ball["position"] = np.array(
            [config["environment"]["field_length"] / 2 + 0.01, 0.0]
        )
        env.ball["shot_in_flight"] = True
        for defender in env.defenders.values():
            defender["position"] = np.array([-4.0, 2.5])
        _, _, terminations, truncations, _ = env.step(
            {agent: 6 for agent in env.active_agents}
        )
        assert env.agents
        assert env.match_score == 1
        assert env.match_restart_count == 1
        assert not any(terminations.values())
        assert not any(truncations.values())
    finally:
        env.close()


def test_match_clock_truncation_and_pass_to_goal(phase3_config):
    config = copy.deepcopy(phase3_config)
    config["phase3"]["match_mode"] = False
    config["phase3"]["minimum_goal_seconds"] = 0.0
    env = make_phase3_environment(
        config, scenario="phase3_2v2_pass_required"
    )
    try:
        env.reset(seed=170)
        env.cooperative_sequence = True
        env.ball["possession"] = None
        env.ball["shot_in_flight"] = True
        env.ball["cooperative_shot"] = True
        env.ball["position"] = np.array(
            [config["environment"]["field_length"] / 2 + 0.01, 0.0]
        )
        for defender in env.defenders.values():
            defender["position"] = np.array([-4.0, 2.5])
        _, _, terminations, _, infos = env.step(
            {agent: 6 for agent in env.active_agents}
        )
        assert all(terminations.values())
        metrics = next(iter(infos.values()))["episode_metrics"]
        assert metrics["pass_to_goal"] == 1
        assert metrics["cooperative_success"] == 1
        assert metrics["pass_and_goal_rate"] == 1.0
    finally:
        env.close()

    timeout_config = copy.deepcopy(phase3_config)
    timeout_config["phase3"]["match_mode"] = True
    timeout_config["phase3"]["match_seconds"] = 0.2
    timeout = make_phase3_environment(
        timeout_config, scenario="phase3_2v2_open"
    )
    try:
        timeout.reset(seed=171)
        _, _, _, truncations, _ = timeout.step(
            {agent: 6 for agent in timeout.active_agents}
        )
        assert all(truncations.values())
    finally:
        timeout.close()


def test_gae_bootstraps_match_truncation_without_crossing_reset():
    advantages, returns = compute_gae(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([[0.5], [0.4]], dtype=np.float32),
        np.array([[0.4], [0.3]], dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        np.array([[0.0], [1.0]], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95,
    )
    assert advantages[1, 0] == pytest.approx(2.0 + 0.9 * 0.3 - 0.4)
    assert returns[1, 0] == pytest.approx(2.0 + 0.9 * 0.3)
    assert advantages[0, 0] > 0.0


def test_phase3_batch_one_lane_equivalence_and_finite_state(phase3_config):
    direct = make_phase3_environment(
        phase3_config, simulator="abstract", scenario="phase3_2v2_open"
    )
    batch = Phase3EnvironmentBatch(
        phase3_config,
        1,
        simulator="abstract",
        scenario="phase3_2v2_open",
        seed_base=18,
    )
    try:
        direct_observations, _ = direct.reset(seed=18)
        batch_observations, _ = batch.reset()
        for agent in direct.active_agents:
            np.testing.assert_allclose(
                direct_observations[agent], batch_observations[0][agent]
            )
        actions = {agent: 6 for agent in direct.active_agents}
        direct_result = direct.step(actions)
        batch_result = batch.step([actions])[0]
        for agent in direct.active_agents:
            np.testing.assert_allclose(direct_result[0][agent], batch_result[0][agent])
        assert np.isfinite(batch.environments[0].state()).all()
    finally:
        direct.close()
        batch.close()


def test_competence_guard_and_probability_constraints(phase3_config):
    curriculum = CompetenceConstrainedCurriculum(
        ["delay", "noise", "movement", "kick"], phase3_config["phase3"]["cc_fdr"]
    )
    curriculum.nominal_reference = 0.8
    before = curriculum.nominal_probability
    probabilities = curriculum.update(
        {"delay": 0.0, "noise": 0.9, "movement": 0.8, "kick": 0.7},
        nominal_score=0.6,
        update_number=1,
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert curriculum.nominal_probability >= before
    assert (
        curriculum.nominal_probability
        >= phase3_config["phase3"]["cc_fdr"]["nominal_rehearsal_minimum"]
    )
    assert curriculum.guard_events == 1


def test_cc_fdr_composite_checkpoint_score(phase3_config, tmp_path):
    _trainer_directories(tmp_path)
    trainer = RecurrentMAPPOTrainer(phase3_config, tmp_path)
    trainer.curriculum = CompetenceConstrainedCurriculum(
        ["delay", "noise", "movement", "kick"],
        phase3_config["phase3"]["cc_fdr"],
    )
    trainer.curriculum.nominal_reference = 0.8
    validation = {
        "nominal": {"success_rate": 0.65},
        "cooperation": {"cooperative_success_rate": 0.4},
        "robustness": {"profile_mean": 0.5, "grid_auc": 0.6},
    }
    try:
        score, feasible = trainer._composite(validation)
        expected = 0.40 * 0.5 + 0.25 * 0.6 + 0.20 * 0.4 + 0.15 * 0.65 - 0.50 * 0.05
        assert score == pytest.approx(expected)
        assert feasible is False
    finally:
        trainer.close()


def test_relative_and_historical_pointer_resolution(tmp_path):
    repository = tmp_path / "repository"
    run = repository / "runs" / "20260722_example_seed3"
    run.mkdir(parents=True)
    relative = repository / "runs" / "latest_relative.txt"
    relative.write_text("runs/20260722_example_seed3\n", encoding="utf-8")
    assert resolve_run_pointer(relative, repository) == run.resolve()
    historical = repository / "runs" / "latest_historical.txt"
    historical.write_text(
        "/content/robot-soccer-transfer/runs/20260722_example_seed3\n",
        encoding="utf-8",
    )
    assert resolve_run_pointer(historical, repository) == run.resolve()


def test_phase3_render_is_rgb(phase3_config):
    env = make_phase3_environment(
        phase3_config, simulator="pymunk", scenario="phase3_3v2_open"
    )
    try:
        env.reset(seed=19)
        frame = env.render()
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_recurrent_checkpoint_round_trip_and_export(phase3_config, tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _trainer_directories(first_root)
    _trainer_directories(second_root)
    first = RecurrentMAPPOTrainer(phase3_config, first_root)
    try:
        checkpoint = first.save_checkpoint(first_root / "models" / "round_trip.pt")
        actor_path = first.export_actor(first_root / "models" / "actor.ts")
        assert actor_path.is_file()
        deployment = torch.jit.load(str(actor_path))
        masks = torch.ones(1, 7)
        masks[0, 3] = 0.0
        logits, hidden = deployment(
            torch.zeros(1, 112),
            torch.zeros(1, 1, 64),
            masks,
        )
        assert logits[0, 3] < -1e8
        assert hidden.shape == (1, 1, 64)
    finally:
        first.close()
    second = RecurrentMAPPOTrainer(
        phase3_config, second_root, resume_path=checkpoint
    )
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert second.environment_steps == saved["environment_steps"]
        assert second.current_update == saved["current_update"]
    finally:
        second.close()


def test_stage_r_warm_start_restores_weights_and_normalizers_but_resets_optimizers(
    phase3_config, tmp_path
):
    source_root = tmp_path / "synthetic_stage_d"
    target_root = tmp_path / "stage_r"
    checkpoint = stage_d_checkpoint(phase3_config, source_root)
    _trainer_directories(target_root)
    config = stage_r_test_config(phase3_config)
    trainer = RecurrentMAPPOTrainer(
        config,
        target_root,
        warm_start_path=checkpoint,
    )
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for name, value in trainer.actor.state_dict().items():
            torch.testing.assert_close(value.cpu(), saved["actor_weights"][name])
        for name, value in trainer.critic.state_dict().items():
            torch.testing.assert_close(value.cpu(), saved["critic_weights"][name])
        np.testing.assert_allclose(trainer.observation_rms.mean, 1.25)
        np.testing.assert_allclose(trainer.state_rms.mean, -0.75)
        assert not trainer.actor_optimizer.state
        assert not trainer.critic_optimizer.state
        assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert trainer.current_update == 0
        assert trainer.environment_steps == 0
        protocol = json.loads(
            (target_root / "logs" / "stage_r_protocol.json").read_text()
        )
        assert protocol["reset_actor_optimizer"] is True
        assert protocol["warm_start_sha256"]
    finally:
        trainer.close()


def test_stage_r_resume_refuses_historical_stage_d_checkpoint(
    phase3_config, tmp_path
):
    source_root = tmp_path / "synthetic_stage_d_resume"
    target_root = tmp_path / "stage_r_resume"
    checkpoint = stage_d_checkpoint(phase3_config, source_root)
    _trainer_directories(target_root)
    with pytest.raises(ValueError, match="Stage R resume accepts only"):
        RecurrentMAPPOTrainer(
            stage_r_test_config(phase3_config),
            target_root,
            resume_path=checkpoint,
        )


def test_stage_r_checkpoint_selection_uses_only_abstract_validation(
    phase3_config, tmp_path, monkeypatch
):
    source_root = tmp_path / "synthetic_stage_d_validation"
    target_root = tmp_path / "stage_r_validation"
    checkpoint = stage_d_checkpoint(phase3_config, source_root)
    _trainer_directories(target_root)
    trainer = RecurrentMAPPOTrainer(
        stage_r_test_config(phase3_config, source_root.name),
        target_root,
        warm_start_path=checkpoint,
    )
    simulators = []

    def fake_evaluate(
        scenario,
        episodes,
        seed_base,
        simulator="abstract",
        profile="nominal",
        defender_style="mixed",
    ):
        simulators.append(simulator)
        return {
            "success_rate": 0.8,
            "cooperative_success_rate": 0.8,
            "mean_return": 10.0,
            "mean_successful_return": 12.0,
            "mean_failed_return": 4.0,
            "success_failure_return_gap": 8.0,
        }

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    try:
        validation = trainer._validation()
        trainer._save_best(validation)
        assert simulators and set(simulators) == {"abstract"}
        assert validation["checkpoint_selection_uses_pymunk"] is False
        assert (target_root / "models" / "best_stage_r_checkpoint.pt").is_file()
        assert (target_root / "models" / "best_nominal_checkpoint.pt").is_file()
    finally:
        trainer.close()


def test_recurrent_checkpoint_rng_states_are_normalized_to_cpu_byte_tensors():
    state = torch.get_rng_state()
    if torch.cuda.is_available():
        state = state.to("cuda")
    normalized = _cpu_rng_state(state, "test RNG state")
    assert normalized.device.type == "cpu"
    assert normalized.dtype == torch.uint8
    assert normalized.is_contiguous()
    with pytest.raises(TypeError, match="must be a torch byte tensor"):
        _cpu_rng_state(torch.ones(4), "invalid RNG state")


def test_recurrent_checkpoint_payload_is_loaded_on_cpu(
    phase3_config, tmp_path, monkeypatch
):
    first_root = tmp_path / "first_cpu_load"
    second_root = tmp_path / "second_cpu_load"
    _trainer_directories(first_root)
    _trainer_directories(second_root)
    first = RecurrentMAPPOTrainer(phase3_config, first_root)
    try:
        checkpoint = first.save_checkpoint(first_root / "models" / "cpu_load.pt")
    finally:
        first.close()
    original_load = torch.load
    observed = []

    def recording_load(*args, **kwargs):
        observed.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    second = RecurrentMAPPOTrainer(
        phase3_config, second_root, resume_path=checkpoint
    )
    try:
        assert observed == ["cpu"]
    finally:
        second.close()


def test_video_manifest_append_and_deduplicate():
    first = {
        "checkpoint": "best.pt",
        "simulator": "pymunk",
        "scenario": "phase3_2v2_open",
        "defender_style": "predictive",
        "profile": "nominal",
        "seed": 360001,
        "recording_mode": "fixed_duration",
        "terminal": False,
    }
    replacement = dict(first)
    replacement["terminal"] = True
    second = dict(first)
    second["seed"] = 360002
    merged = merge_manifest_records([first], [replacement, second])
    assert len(merged) == 2
    assert merged[0]["terminal"] is True
    assert manifest_record_key(first) == manifest_record_key(replacement)
    assert manifest_record_key(first) != manifest_record_key(second)


def test_partial_and_terminal_video_metrics_and_fixed_style_replay(
    phase3_config, tmp_path
):
    run_dir = tmp_path / "video_run"
    _trainer_directories(run_dir)
    config = copy.deepcopy(phase3_config)
    config["phase3"]["max_episode_steps"] = 200
    config["phase3"]["minimum_goal_seconds"] = 1000.0
    config["phase3"]["defender_speed"] = 0.0
    config["video"]["width"] = 160
    config["video"]["height"] = 96
    config["video"]["fps"] = 1
    trainer = RecurrentMAPPOTrainer(config, run_dir)
    try:
        with torch.no_grad():
            trainer.actor.policy.bias[6] = 10.0
        trainer.save_checkpoint(run_dir / "models" / "best_nominal_checkpoint.pt")
    finally:
        trainer.close()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    partial_args = SimpleNamespace(
        run_dir=str(run_dir),
        checkpoint="best_nominal",
        simulator="abstract",
        scenario="phase3_2v2_open",
        profile="nominal",
        defender_style="predictive",
        episodes=1,
        seed_base=340000,
        seed=362000,
        seed_category="audit",
        seconds=15.0,
        until_terminal=False,
        full_match=False,
        device="cpu",
    )
    partial, manifest = record_phase3_videos(partial_args)
    assert partial[0]["terminal"] is False
    assert partial[0]["clip_end_reason"] == "video_time_limit"
    assert partial[0]["terminal_metrics"] is None
    assert partial[0]["clip_end_metrics"]["episode_steps"] > 0
    assert partial[0]["actual_defender_style"] == "predictive"

    config["phase3"]["max_episode_steps"] = 2
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    terminal_args = copy.copy(partial_args)
    terminal_args.seed = 362001
    terminal_args.until_terminal = True
    terminal, _ = record_phase3_videos(terminal_args)
    assert terminal[0]["terminal"] is True
    assert terminal[0]["terminal_metrics"]["terminal_reason"] == "timeout"
    saved = json.loads(manifest.read_text())
    assert len(saved) == 2
    assert {record["seed"] for record in saved} == {362000, 362001}


def test_cc_fdr_requires_passing_gate_b_r_and_stage_r_checkpoint(
    phase3_config, tmp_path
):
    stage_d_root = tmp_path / "cc_stage_d"
    stage_r_root = tmp_path / "cc_stage_r"
    cc_root = tmp_path / "cc_run"
    stage_d = stage_d_checkpoint(phase3_config, stage_d_root)
    _trainer_directories(stage_r_root)
    stage_r = RecurrentMAPPOTrainer(
        stage_r_test_config(phase3_config, stage_d_root.name),
        stage_r_root,
        warm_start_path=stage_d,
    )
    try:
        stage_r.last_validation = {"nominal": {"success_rate": 0.8}}
        stage_r_checkpoint = stage_r.save_checkpoint(
            stage_r_root / "models" / "best_stage_r_checkpoint.pt"
        )
    finally:
        stage_r.close()

    historical_gate = tmp_path / "phase3_gate_b.json"
    historical_gate.write_text(
        json.dumps({"gate": "B", "passed": True}), encoding="utf-8"
    )
    cc_config = stage_r_test_config(phase3_config)
    cc_config["phase3"]["mode"] = "cc_fdr"
    cc_config["phase3"]["active_stage"] = "stage_d"
    cc_config["phase3"]["cc_fdr"]["authorization_artifact"] = str(
        historical_gate
    )
    _trainer_directories(cc_root)
    with pytest.raises(ValueError, match="CC-FDR remains unauthorized"):
        RecurrentMAPPOTrainer(
            cc_config,
            cc_root,
            warm_start_path=stage_r_checkpoint,
        )

    passing_gate = tmp_path / "phase3_gate_b_r.json"
    passing_gate.write_text(
        json.dumps(
            {
                "gate": "B-R",
                "passed": True,
                "cc_fdr_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    cc_config["phase3"]["cc_fdr"]["authorization_artifact"] = str(passing_gate)
    accepted_root = tmp_path / "cc_accepted"
    _trainer_directories(accepted_root)
    accepted = RecurrentMAPPOTrainer(
        cc_config,
        accepted_root,
        warm_start_path=stage_r_checkpoint,
    )
    try:
        assert accepted.curriculum is not None
        assert accepted.current_update == 0
        assert not accepted.actor_optimizer.state
    finally:
        accepted.close()


def test_stage_r_seed_ranges_do_not_reuse_historical_protocols():
    config = load_config("configs/phase3_stage_r.yaml")
    seeds = config["evaluation"]["seed_bases"]
    assert seeds["phase3_stage_r_r0_audit"] == 360000
    assert seeds["phase3_stage_r_validation"] == 365000
    assert seeds["phase3_gate_b_r"] == 370000
    assert config["phase3"]["stage_r"]["training_episode_seed_base"] == 1000000000
    assert len(
        {
            310000,
            330000,
            340000,
            350000,
            seeds["phase3_stage_r_r0_audit"],
            seeds["phase3_stage_r_validation"],
            seeds["phase3_gate_b_r"],
            config["phase3"]["stage_r"]["training_episode_seed_base"],
        }
    ) == 8
