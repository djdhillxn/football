"""Concentrated correctness tests for the mandatory research pipeline."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pettingzoo.test import parallel_api_test

from robosoccer.config import apply_overrides, load_config
from robosoccer.environment import (
    AGENTS,
    AbstractSoccerEnv,
    PymunkSoccerTransferEnv,
    baseline_actions,
    sample_profile_parameters,
)
from robosoccer.evaluation import record_videos
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
