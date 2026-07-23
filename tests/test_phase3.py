"""Correctness tests for the gated recurrent Phase 3 extension."""

import copy

import numpy as np
import pytest
import torch

from robosoccer.artifacts import resolve_run_pointer
from robosoccer.config import load_config
from robosoccer.phase3 import (
    PHASE3_AGENTS,
    Phase3EnvironmentBatch,
    make_phase3_environment,
)
from robosoccer.recurrent import (
    CompetenceConstrainedCurriculum,
    RecurrentCentralCritic,
    RecurrentMAPPOTrainer,
    RecurrentSharedActor,
    _cpu_rng_state,
)
from robosoccer.training import compute_gae


@pytest.fixture(scope="module")
def phase3_config():
    return load_config("configs/phase3_smoke.yaml")


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
