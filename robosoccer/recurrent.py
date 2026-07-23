"""Recurrent shared MAPPO and competence-constrained Phase 3 fine-tuning."""

import copy
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from robosoccer.phase3 import make_phase3_environment
from robosoccer.training import (
    activation_class,
    capped_distribution,
    compute_gae,
    explained_variance,
    finite_gradients,
)
from robosoccer.utils import (
    MetricsWriter,
    RunningMeanStd,
    get_pyplot,
    select_device,
    write_json,
)

logger = logging.getLogger(__name__)


def plot_phase3_diagnostics(run_dir, curriculum=None):
    path = Path(run_dir) / "logs" / "metrics.csv"
    if not path.is_file():
        return
    data = pd.read_csv(path)
    if data.empty:
        return
    plt = get_pyplot()
    specifications = [
        (
            ["success_rate", "cooperative_success_rate", "pass_completion_rate"],
            "Phase 3 team-play learning",
            "Rate",
            "phase3_teamplay_learning.png",
        ),
        (
            [
                "validation_nominal_success",
                "validation_cooperative_success",
                "validation_profile_mean",
                "validation_grid_auc",
            ],
            "Nominal--cooperation--robustness validation",
            "Success",
            "phase3_validation_tradeoff.png",
        ),
        (
            [
                "environment_stepping_seconds",
                "actor_inference_seconds",
                "actor_optimization_seconds",
                "critic_optimization_seconds",
                "validation_seconds",
                "checkpoint_io_seconds",
            ],
            "Phase 3 update wall-time breakdown",
            "Seconds",
            "phase3_throughput_breakdown.png",
        ),
    ]
    for columns, title, ylabel, filename in specifications:
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        plotted = False
        for column in columns:
            if column not in data:
                continue
            values = pd.to_numeric(data[column], errors="coerce")
            if values.notna().any():
                axis.plot(data["environment_steps"], values, label=column.replace("_", " "))
                plotted = True
        if plotted:
            axis.set(
                title=title,
                xlabel="Environment steps",
                ylabel=ylabel,
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
            figure.tight_layout()
            figure.savefig(Path(run_dir) / "plots" / filename, dpi=180)
        plt.close(figure)
    if curriculum is None or not curriculum.history:
        return
    history = curriculum.history
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    updates = [entry["update"] for entry in history]
    for profile in curriculum.profile_names:
        axes[0].plot(
            updates,
            [entry["profile_probabilities"][profile] for entry in history],
            label=profile,
        )
    axes[0].set(ylabel="Profile probability", title="CC-FDR profile allocation")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot(
        updates,
        [entry["nominal_probability"] for entry in history],
        label="nominal rehearsal",
    )
    axes[1].step(
        updates,
        [float(entry["guard_active"]) for entry in history],
        where="post",
        label="guard active",
    )
    axes[1].set(xlabel="Update", ylabel="Probability / indicator")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(
        Path(run_dir) / "plots" / "phase3_cc_fdr_curriculum.png", dpi=180
    )
    plt.close(figure)


def _encoder(input_size, hidden_sizes, activation, orthogonal):
    layers = []
    previous = int(input_size)
    activation_type = activation_class(activation)
    for size in hidden_sizes:
        layer = nn.Linear(previous, int(size))
        if orthogonal:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        layers.extend([layer, activation_type()])
        previous = int(size)
    return nn.Sequential(*layers), previous


class RecurrentSharedActor(nn.Module):
    """Parameter-shared GRU actor with explicit hidden-state reset masks."""

    def __init__(self, observation_size, action_size, model_config, recurrent_config):
        super().__init__()
        self.hidden_size = int(recurrent_config["hidden_size"])
        self.encoder, encoded_size = _encoder(
            observation_size,
            model_config["actor_hidden_sizes"],
            model_config["activation"],
            model_config.get("orthogonal_init", True),
        )
        self.gru = nn.GRU(encoded_size, self.hidden_size)
        self.policy = nn.Linear(self.hidden_size, action_size)
        if model_config.get("orthogonal_init", True):
            for name, parameter in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.zeros_(parameter)
            nn.init.orthogonal_(self.policy.weight, gain=0.01)
            nn.init.zeros_(self.policy.bias)

    def forward(self, observations, hidden, continuation=None):
        single_step = observations.ndim == 2
        if single_step:
            observations = observations.unsqueeze(0)
        encoded = self.encoder(observations)
        outputs = []
        current = hidden
        for index in range(encoded.shape[0]):
            if continuation is not None and index > 0:
                current = current * continuation[index].reshape(1, -1, 1)
            output, current = self.gru(encoded[index : index + 1], current)
            outputs.append(output)
        logits = self.policy(torch.cat(outputs, dim=0))
        if single_step:
            logits = logits.squeeze(0)
        return logits, current


class MaskedRecurrentDeploymentActor(nn.Module):
    """Deployment wrapper preserving the legal-action contract."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observations, hidden, action_masks):
        logits, next_hidden = self.actor(observations, hidden)
        return logits.masked_fill(action_masks < 0.5, -1e9), next_hidden


class RecurrentCentralCritic(nn.Module):
    """Recurrent centralized value estimator over the fixed global state."""

    def __init__(self, state_size, model_config, recurrent_config):
        super().__init__()
        self.hidden_size = int(recurrent_config["hidden_size"])
        self.encoder, encoded_size = _encoder(
            state_size,
            model_config["central_critic_hidden_sizes"],
            model_config["activation"],
            model_config.get("orthogonal_init", True),
        )
        self.gru = nn.GRU(encoded_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, 1)
        if model_config.get("orthogonal_init", True):
            for name, parameter in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.zeros_(parameter)
            nn.init.orthogonal_(self.value.weight, gain=1.0)
            nn.init.zeros_(self.value.bias)

    def forward(self, states, hidden, continuation=None):
        single_step = states.ndim == 2
        if single_step:
            states = states.unsqueeze(0)
        encoded = self.encoder(states)
        outputs = []
        current = hidden
        for index in range(encoded.shape[0]):
            if continuation is not None and index > 0:
                current = current * continuation[index].reshape(1, -1, 1)
            output, current = self.gru(encoded[index : index + 1], current)
            outputs.append(output)
        values = self.value(torch.cat(outputs, dim=0)).squeeze(-1)
        if single_step:
            values = values.squeeze(0)
        return values, current


class CompetenceConstrainedCurriculum:
    """Failure-directed sampling with a hard nominal-rehearsal guard."""

    def __init__(self, profile_names, config):
        self.profile_names = list(profile_names)
        self.config = copy.deepcopy(config)
        self.nominal_probability = max(
            float(config["nominal_rehearsal_minimum"]),
            float(config["nominal_rehearsal_initial"]),
        )
        self.failure_exponent = float(config["failure_exponent"])
        self.smoothed_failures = {name: 0.5 for name in self.profile_names}
        uniform = 1.0 / max(1, len(self.profile_names))
        self.probabilities = {name: uniform for name in self.profile_names}
        self.nominal_reference = None
        self.guard_events = 0
        self.history = []

    def sample(self, rng):
        return self.sample_group(rng)[1]

    def sample_group(self, rng):
        draw = rng.random()
        if draw < self.nominal_probability or not self.profile_names:
            return "nominal", "nominal"
        cooperation = min(
            float(self.config.get("cooperation_probability", 0.20)),
            max(0.0, 1.0 - self.nominal_probability),
        )
        if draw < self.nominal_probability + cooperation:
            return "cooperation", "nominal"
        weights = np.asarray(
            [self.probabilities[name] for name in self.profile_names], dtype=np.float64
        )
        weights /= weights.sum()
        return "randomized", str(rng.choice(self.profile_names, p=weights))

    def update(self, profile_success, nominal_score, update_number):
        decay = float(self.config["ema_decay"])
        for name in self.profile_names:
            failure = 1.0 - float(profile_success.get(name, 0.0))
            self.smoothed_failures[name] = (
                decay * self.smoothed_failures[name] + (1.0 - decay) * failure
            )
        raw = np.asarray(
            [
                max(1e-5, self.smoothed_failures[name]) ** self.failure_exponent
                for name in self.profile_names
            ],
            dtype=np.float64,
        )
        raw /= raw.sum()
        mixture = float(self.config["uniform_mixture"])
        raw = (1.0 - mixture) * raw + mixture / max(1, len(raw))
        raw = capped_distribution(raw, float(self.config["maximum_profile_probability"]))
        self.probabilities = dict(zip(self.profile_names, raw.tolist(), strict=True))
        margin = float(self.config["nominal_regression_margin"])
        guard_active = (
            self.nominal_reference is not None
            and float(nominal_score) < float(self.nominal_reference) - margin
        )
        if guard_active:
            self.guard_events += 1
            self.nominal_probability = min(
                float(self.config["nominal_rehearsal_maximum"]),
                self.nominal_probability
                + float(self.config["nominal_rehearsal_increment"]),
            )
            self.failure_exponent = max(
                float(self.config["failure_exponent_minimum"]),
                self.failure_exponent
                - float(self.config["failure_exponent_decrement"]),
            )
        self.history.append(
            {
                "update": int(update_number),
                "nominal_score": float(nominal_score),
                "nominal_reference": self.nominal_reference,
                "constraint_margin": None
                if self.nominal_reference is None
                else float(nominal_score)
                - (float(self.nominal_reference) - margin),
                "guard_active": bool(guard_active),
                "nominal_probability": self.nominal_probability,
                "failure_exponent": self.failure_exponent,
                "profile_probabilities": copy.deepcopy(self.probabilities),
            }
        )
        return copy.deepcopy(self.probabilities)

    def state_dict(self):
        return {
            "profile_names": self.profile_names,
            "nominal_probability": self.nominal_probability,
            "cooperation_probability": float(
                self.config.get("cooperation_probability", 0.20)
            ),
            "failure_exponent": self.failure_exponent,
            "smoothed_failures": self.smoothed_failures,
            "probabilities": self.probabilities,
            "nominal_reference": self.nominal_reference,
            "guard_events": self.guard_events,
            "history": self.history,
        }

    def load_state_dict(self, state):
        if list(state["profile_names"]) != self.profile_names:
            raise ValueError("CC-FDR profile set differs from checkpoint")
        self.nominal_probability = float(state["nominal_probability"])
        self.failure_exponent = float(state["failure_exponent"])
        self.smoothed_failures = copy.deepcopy(state["smoothed_failures"])
        self.probabilities = copy.deepcopy(state["probabilities"])
        self.nominal_reference = state.get("nominal_reference")
        self.guard_events = int(state.get("guard_events", 0))
        self.history = copy.deepcopy(state.get("history", []))


class RecurrentMAPPOTrainer:
    """Synchronous recurrent MAPPO with padded roster masks and TBPTT PPO."""

    def __init__(self, config, run_dir, resume_path=None, warm_start_path=None):
        self.config = copy.deepcopy(config)
        self.run_dir = Path(run_dir)
        self.device = select_device(config["train"]["device"])
        self.num_envs = int(config["train"]["num_envs"])
        self.rollout_steps = int(config["train"]["rollout_steps"])
        self.phase3 = self.config["phase3"]
        self.recurrent = self.phase3["recurrent"]
        self.hidden_size = int(self.recurrent["hidden_size"])
        self.mode = self.phase3.get("mode", "nominal")
        self.stage_name = self.phase3.get("active_stage", "stage_a")
        self.stage = copy.deepcopy(self.phase3["stages"][self.stage_name])
        self.config["phase3"]["match_mode"] = bool(
            self.stage.get("match_mode", self.phase3.get("match_mode", False))
        )
        config = self.config
        self.seed = int(config["experiment"]["seed"])
        self.rng = np.random.default_rng(self.seed + 9000)
        probe = make_phase3_environment(config, simulator="abstract", scenario=self.stage["scenarios"][0])
        self.agent_names = probe.possible_agents[:]
        self.agent_count = len(self.agent_names)
        self.observation_size = probe.observation_dimension
        self.state_size = probe.state_dimension
        self.action_size = probe.action_size
        probe.close()
        self.actor = RecurrentSharedActor(
            self.observation_size,
            self.action_size,
            config["model"],
            self.recurrent,
        ).to(self.device)
        self.critic = RecurrentCentralCritic(
            self.state_size, config["model"], self.recurrent
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(config["ppo"]["actor_learning_rate"]), eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(config["ppo"]["critic_learning_rate"]), eps=1e-5
        )
        self.observation_rms = RunningMeanStd((self.observation_size,))
        self.state_rms = RunningMeanStd((self.state_size,))
        self.environments = [
            make_phase3_environment(config, simulator="abstract")
            for _ in range(self.num_envs)
        ]
        self.observations = []
        self.states = []
        self.valid_agents = np.zeros((self.num_envs, self.agent_count), dtype=np.float32)
        self.actor_hidden = np.zeros(
            (self.num_envs, self.agent_count, self.hidden_size), dtype=np.float32
        )
        self.critic_hidden = np.zeros((self.num_envs, self.hidden_size), dtype=np.float32)
        self.continuation = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.completed_episodes = []
        self.current_update = 0
        self.environment_steps = 0
        self.stage_start_steps = 0
        self.best_scores = {"nominal": None, "cooperation": None, "composite": None}
        self.last_validation = None
        self.last_rollout_timing = {}
        self.last_logging_io_seconds = 0.0
        self.low_update_signal_count = 0
        self.curriculum = None
        if self.mode == "cc_fdr":
            profiles = [name for name in self.phase3["profiles"] if name != "nominal"]
            self.curriculum = CompetenceConstrainedCurriculum(
                profiles, self.phase3["cc_fdr"]
            )
            if resume_path is None and warm_start_path is None:
                raise ValueError("Phase 3 CC-FDR requires --warm-start or --resume")
        if resume_path is not None:
            self.load_checkpoint(resume_path, warm_start=False)
        elif warm_start_path is not None:
            self.load_checkpoint(warm_start_path, warm_start=True)
        self._reset_all()

    def _sample_reset_options(self):
        if self.curriculum is None:
            scenario = str(
                self.rng.choice(
                    self.stage["scenarios"], p=self.stage.get("probabilities")
                )
            )
            return {
                "scenario": scenario,
                "profile": "nominal",
                "scenario_group": "nominal_curriculum",
            }
        group, profile = self.curriculum.sample_group(self.rng)
        curriculum = self.phase3["cc_fdr"]
        if group == "nominal":
            scenario = str(
                self.rng.choice(
                    curriculum["nominal_scenarios"],
                    p=curriculum["nominal_scenario_probabilities"],
                )
            )
        elif group == "cooperation":
            scenario = "phase3_2v2_pass_required"
        else:
            scenario = str(
                self.rng.choice(
                    curriculum["robustness_scenarios"],
                    p=curriculum["robustness_scenario_probabilities"],
                )
            )
        return {
            "scenario": scenario,
            "profile": profile,
            "scenario_group": group,
        }

    def _reset_lane(self, index, initial=False):
        options = self._sample_reset_options()
        seed = self.seed + index if initial else int(self.rng.integers(0, 2**31 - 1))
        observations, _ = self.environments[index].reset(seed=seed, options=options)
        self.observations[index] = observations
        self.states[index] = self.environments[index].state()
        self.valid_agents[index] = [
            float(agent in self.environments[index].active_agents)
            for agent in self.agent_names
        ]
        self.actor_hidden[index] = 0.0
        self.critic_hidden[index] = 0.0
        self.continuation[index] = 0.0

    def _reset_all(self):
        self.observations = [{} for _ in range(self.num_envs)]
        self.states = [None for _ in range(self.num_envs)]
        for index in range(self.num_envs):
            self._reset_lane(index, initial=True)

    def _raw_observation_batch(self):
        result = np.zeros(
            (self.num_envs, self.agent_count, self.observation_size), dtype=np.float32
        )
        masks = np.zeros(
            (self.num_envs, self.agent_count, self.action_size), dtype=np.float32
        )
        for env_index, env in enumerate(self.environments):
            for agent_index, agent in enumerate(self.agent_names):
                if agent in env.active_agents:
                    result[env_index, agent_index] = self.observations[env_index][agent]
                    masks[env_index, agent_index] = env.action_mask(agent)
        return result, masks

    def _normalize_observations(self, values, valid, update=True):
        values = np.asarray(values, dtype=np.float32)
        if not self.config["observations"].get("normalize", True):
            return values
        flat = values.reshape(-1, self.observation_size)
        active = np.asarray(valid).reshape(-1) > 0.5
        if update and np.any(active):
            self.observation_rms.update(flat[active])
        return self.observation_rms.normalize(
            values, self.config["observations"]["clip"]
        )

    def _normalize_states(self, values, update=True):
        values = np.asarray(values, dtype=np.float32)
        if not self.config["observations"].get("normalize", True):
            return values
        if update:
            self.state_rms.update(values)
        return self.state_rms.normalize(values, self.config["observations"]["clip"])

    def collect_rollout(self):
        steps = self.rollout_steps
        shape = (steps, self.num_envs, self.agent_count)
        observations = np.zeros((*shape, self.observation_size), dtype=np.float32)
        states = np.zeros((steps, self.num_envs, self.state_size), dtype=np.float32)
        action_masks = np.zeros((*shape, self.action_size), dtype=np.float32)
        valid_agents = np.zeros(shape, dtype=np.float32)
        actions = np.zeros(shape, dtype=np.int64)
        old_logs = np.zeros(shape, dtype=np.float32)
        rewards = np.zeros((steps, self.num_envs), dtype=np.float32)
        values = np.zeros((steps, self.num_envs), dtype=np.float32)
        next_values = np.zeros_like(values)
        terminations = np.zeros_like(values)
        truncations = np.zeros_like(values)
        continuations = np.zeros((steps, self.num_envs), dtype=np.float32)
        actor_hidden = np.zeros((*shape, self.hidden_size), dtype=np.float32)
        critic_hidden = np.zeros((steps, self.num_envs, self.hidden_size), dtype=np.float32)
        self.completed_episodes = []
        self.actor.eval()
        self.critic.eval()
        transfer_seconds = 0.0
        inference_seconds = 0.0
        environment_seconds = 0.0
        for step in range(steps):
            raw_observations, raw_masks = self._raw_observation_batch()
            raw_states = np.asarray(self.states, dtype=np.float32)
            normalized_observations = self._normalize_observations(
                raw_observations, self.valid_agents, update=True
            )
            normalized_states = self._normalize_states(raw_states, update=True)
            observations[step] = normalized_observations
            states[step] = normalized_states
            action_masks[step] = raw_masks
            valid_agents[step] = self.valid_agents
            continuations[step] = self.continuation
            actor_hidden[step] = self.actor_hidden
            critic_hidden[step] = self.critic_hidden
            transfer_started = time.perf_counter()
            flat_observations = torch.as_tensor(
                normalized_observations.reshape(-1, self.observation_size),
                device=self.device,
            )
            flat_hidden = torch.as_tensor(
                self.actor_hidden.reshape(1, -1, self.hidden_size), device=self.device
            )
            mask_tensor = torch.as_tensor(
                raw_masks.reshape(-1, self.action_size), device=self.device
            )
            state_tensor = torch.as_tensor(normalized_states, device=self.device)
            critic_hidden_tensor = torch.as_tensor(
                self.critic_hidden.reshape(1, self.num_envs, self.hidden_size),
                device=self.device,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            transfer_seconds += time.perf_counter() - transfer_started
            inference_started = time.perf_counter()
            with torch.no_grad():
                logits, next_actor_hidden = self.actor(flat_observations, flat_hidden)
                logits = logits.masked_fill(mask_tensor < 0.5, -1e9)
                distribution = Categorical(logits=logits)
                flat_actions = distribution.sample()
                flat_logs = distribution.log_prob(flat_actions)
                critic_values, next_critic_hidden = self.critic(
                    state_tensor, critic_hidden_tensor
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            actions[step] = flat_actions.cpu().numpy().reshape(
                self.num_envs, self.agent_count
            )
            old_logs[step] = flat_logs.cpu().numpy().reshape(
                self.num_envs, self.agent_count
            )
            values[step] = critic_values.cpu().numpy()
            self.actor_hidden = next_actor_hidden.cpu().numpy().reshape(
                self.num_envs, self.agent_count, self.hidden_size
            )
            self.actor_hidden *= self.valid_agents[:, :, None]
            self.critic_hidden = next_critic_hidden.cpu().numpy().reshape(
                self.num_envs, self.hidden_size
            )
            inference_seconds += time.perf_counter() - inference_started
            environment_started = time.perf_counter()
            for env_index, env in enumerate(self.environments):
                action_dict = {
                    agent: int(actions[step, env_index, agent_index])
                    for agent_index, agent in enumerate(self.agent_names)
                    if agent in env.active_agents
                }
                next_observations, reward_dict, terminated_dict, truncated_dict, infos = env.step(
                    action_dict
                )
                reward = float(next(iter(reward_dict.values())))
                rewards[step, env_index] = reward
                self.episode_returns[env_index] += reward
                self.episode_lengths[env_index] += 1
                terminated = bool(next(iter(terminated_dict.values())))
                truncated = bool(next(iter(truncated_dict.values())))
                terminations[step, env_index] = float(terminated)
                truncations[step, env_index] = float(truncated)
                final_state = env.state()
                normalized_final = self._normalize_states(final_state, update=False)
                with torch.no_grad():
                    final_value, _ = self.critic(
                        torch.as_tensor(normalized_final[None, :], device=self.device),
                        torch.as_tensor(
                            self.critic_hidden[env_index].reshape(
                                1, 1, self.hidden_size
                            ),
                            device=self.device,
                        ),
                    )
                next_values[step, env_index] = float(final_value.cpu().numpy()[0])
                if terminated or truncated:
                    metrics = copy.deepcopy(
                        next(iter(infos.values())).get("episode_metrics", {})
                    )
                    metrics["team_return"] = float(self.episode_returns[env_index])
                    self.completed_episodes.append(metrics)
                    self.episode_returns[env_index] = 0.0
                    self.episode_lengths[env_index] = 0
                    self._reset_lane(env_index)
                else:
                    self.observations[env_index] = next_observations
                    self.states[env_index] = final_state
                    self.continuation[env_index] = 1.0
            environment_seconds += time.perf_counter() - environment_started
        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            self.config["ppo"]["gamma"],
            self.config["ppo"]["gae_lambda"],
        )
        actor_advantages = np.repeat(advantages[:, :, None], self.agent_count, axis=2)
        self.environment_steps += steps * self.num_envs
        self.last_rollout_timing = {
            "rollout_transfer_seconds": transfer_seconds,
            "actor_inference_seconds": inference_seconds,
            "environment_stepping_seconds": environment_seconds,
        }
        return {
            "observations": observations,
            "states": states,
            "action_masks": action_masks,
            "valid_agents": valid_agents,
            "actions": actions,
            "old_log_probabilities": old_logs,
            "old_values": values,
            "advantages": actor_advantages,
            "returns": returns,
            "continuations": continuations,
            "actor_hidden": actor_hidden,
            "critic_hidden": critic_hidden,
            "raw_rewards": rewards,
        }

    def _chunk_indices(self, rollout, actor=True):
        length = int(self.recurrent["sequence_length"])
        entries = []
        if actor:
            for env_index in range(self.num_envs):
                for agent_index in range(self.agent_count):
                    for start in range(0, self.rollout_steps, length):
                        end = min(self.rollout_steps, start + length)
                        if np.any(
                            rollout["valid_agents"][start:end, env_index, agent_index] > 0.5
                        ):
                            entries.append((env_index, agent_index, start, end))
        else:
            for env_index in range(self.num_envs):
                for start in range(0, self.rollout_steps, length):
                    entries.append((env_index, start, min(self.rollout_steps, start + length)))
        return entries

    def _actor_minibatch(self, rollout, entries):
        sequence_length = int(self.recurrent["sequence_length"])
        burn_in = int(self.recurrent["burn_in_steps"])
        length = sequence_length + burn_in
        batch = len(entries)
        observations = np.zeros(
            (length, batch, self.observation_size), dtype=np.float32
        )
        action_masks = np.zeros((length, batch, self.action_size), dtype=np.float32)
        actions = np.zeros((length, batch), dtype=np.int64)
        old_logs = np.zeros((length, batch), dtype=np.float32)
        advantages = np.zeros((length, batch), dtype=np.float32)
        valid = np.zeros((length, batch), dtype=np.float32)
        continuation = np.ones((length, batch), dtype=np.float32)
        hidden = np.zeros((1, batch, self.hidden_size), dtype=np.float32)
        for column, (env_index, agent_index, start, end) in enumerate(entries):
            context_start = max(0, start - burn_in)
            count = end - context_start
            loss_start = start - context_start
            observations[:count, column] = rollout["observations"][
                context_start:end, env_index, agent_index
            ]
            action_masks[:count, column] = rollout["action_masks"][
                context_start:end, env_index, agent_index
            ]
            actions[:count, column] = rollout["actions"][
                context_start:end, env_index, agent_index
            ]
            old_logs[:count, column] = rollout["old_log_probabilities"][
                context_start:end, env_index, agent_index
            ]
            advantages[:count, column] = rollout["advantages"][
                context_start:end, env_index, agent_index
            ]
            valid[loss_start:count, column] = rollout["valid_agents"][
                start:end, env_index, agent_index
            ]
            continuation[:count, column] = rollout["continuations"][
                context_start:end, env_index
            ]
            continuation[0, column] = 1.0
            hidden[0, column] = rollout["actor_hidden"][
                context_start, env_index, agent_index
            ]
        return {
            "observations": observations,
            "action_masks": action_masks,
            "actions": actions,
            "old_logs": old_logs,
            "advantages": advantages,
            "valid": valid,
            "continuation": continuation,
            "hidden": hidden,
        }

    def _critic_minibatch(self, rollout, entries):
        sequence_length = int(self.recurrent["sequence_length"])
        burn_in = int(self.recurrent["burn_in_steps"])
        length = sequence_length + burn_in
        batch = len(entries)
        states = np.zeros((length, batch, self.state_size), dtype=np.float32)
        old_values = np.zeros((length, batch), dtype=np.float32)
        returns = np.zeros((length, batch), dtype=np.float32)
        valid = np.zeros((length, batch), dtype=np.float32)
        continuation = np.ones((length, batch), dtype=np.float32)
        hidden = np.zeros((1, batch, self.hidden_size), dtype=np.float32)
        for column, (env_index, start, end) in enumerate(entries):
            context_start = max(0, start - burn_in)
            count = end - context_start
            loss_start = start - context_start
            states[:count, column] = rollout["states"][context_start:end, env_index]
            old_values[:count, column] = rollout["old_values"][
                context_start:end, env_index
            ]
            returns[:count, column] = rollout["returns"][context_start:end, env_index]
            valid[loss_start:count, column] = 1.0
            continuation[:count, column] = rollout["continuations"][
                context_start:end, env_index
            ]
            continuation[0, column] = 1.0
            hidden[0, column] = rollout["critic_hidden"][context_start, env_index]
        return {
            "states": states,
            "old_values": old_values,
            "returns": returns,
            "valid": valid,
            "continuation": continuation,
            "hidden": hidden,
        }

    def ppo_update(self, rollout):
        self.actor.train()
        self.critic.train()
        ppo = self.config["ppo"]
        actor_entries = self._chunk_indices(rollout, actor=True)
        critic_entries = self._chunk_indices(rollout, actor=False)
        sequence_length = int(self.recurrent["sequence_length"])
        actor_chunks_per_batch = max(
            1, int(ppo["actor_minibatch_size"]) // sequence_length
        )
        critic_chunks_per_batch = max(
            1, int(ppo["critic_minibatch_size"]) // sequence_length
        )
        policy_losses = []
        entropies = []
        approximate_kls = []
        clip_fractions = []
        actor_norms = []
        value_losses = []
        critic_norms = []
        active_advantages = rollout["advantages"][rollout["valid_agents"] > 0.5]
        advantage_mean = float(active_advantages.mean())
        advantage_std = float(active_advantages.std() + 1e-8)
        actor_started = time.perf_counter()
        for _ in range(int(ppo["update_epochs"])):
            self.rng.shuffle(actor_entries)
            stop_actor = False
            for start in range(0, len(actor_entries), actor_chunks_per_batch):
                data = self._actor_minibatch(
                    rollout, actor_entries[start : start + actor_chunks_per_batch]
                )
                observations = torch.as_tensor(data["observations"], device=self.device)
                mask = torch.as_tensor(data["action_masks"], device=self.device)
                actions = torch.as_tensor(data["actions"], device=self.device)
                old_logs = torch.as_tensor(data["old_logs"], device=self.device)
                advantages = torch.as_tensor(
                    (data["advantages"] - advantage_mean) / advantage_std,
                    device=self.device,
                )
                valid = torch.as_tensor(data["valid"], device=self.device)
                continuation = torch.as_tensor(data["continuation"], device=self.device)
                hidden = torch.as_tensor(data["hidden"], device=self.device)
                logits, _ = self.actor(observations, hidden, continuation)
                logits = logits.masked_fill(mask < 0.5, -1e9)
                distribution = Categorical(logits=logits)
                new_logs = distribution.log_prob(actions)
                entropy_values = distribution.entropy()
                ratio = torch.exp(new_logs - old_logs)
                unclipped = ratio * advantages
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - float(ppo["clip_range"]),
                        1.0 + float(ppo["clip_range"]),
                    )
                    * advantages
                )
                denominator = valid.sum().clamp_min(1.0)
                policy_loss = -(torch.minimum(unclipped, clipped) * valid).sum() / denominator
                entropy = (entropy_values * valid).sum() / denominator
                actor_loss = policy_loss - float(ppo["entropy_coefficient"]) * entropy
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Recurrent actor loss became non-finite")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if not finite_gradients(self.actor):
                    raise FloatingPointError("Recurrent actor gradients became non-finite")
                norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), float(ppo["max_gradient_norm"])
                )
                self.actor_optimizer.step()
                with torch.no_grad():
                    log_ratio = new_logs - old_logs
                    kl = (((ratio - 1.0) - log_ratio) * valid).sum() / denominator
                    clipped_fraction = (
                        (torch.abs(ratio - 1.0) > float(ppo["clip_range"])).float()
                        * valid
                    ).sum() / denominator
                policy_losses.append(float(policy_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                approximate_kls.append(float(kl.cpu()))
                clip_fractions.append(float(clipped_fraction.cpu()))
                actor_norms.append(float(norm.detach().cpu()))
                if approximate_kls[-1] > float(ppo["target_kl"]):
                    logger.warning(
                        "Recurrent PPO actor stopped early: approximate KL %.4f > %.4f",
                        approximate_kls[-1],
                        float(ppo["target_kl"]),
                    )
                    stop_actor = True
                    break
            if stop_actor:
                break
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        actor_seconds = time.perf_counter() - actor_started
        critic_started = time.perf_counter()
        for _ in range(int(ppo["update_epochs"])):
            self.rng.shuffle(critic_entries)
            for start in range(0, len(critic_entries), critic_chunks_per_batch):
                data = self._critic_minibatch(
                    rollout, critic_entries[start : start + critic_chunks_per_batch]
                )
                states = torch.as_tensor(data["states"], device=self.device)
                old_values = torch.as_tensor(data["old_values"], device=self.device)
                returns = torch.as_tensor(data["returns"], device=self.device)
                valid = torch.as_tensor(data["valid"], device=self.device)
                continuation = torch.as_tensor(data["continuation"], device=self.device)
                hidden = torch.as_tensor(data["hidden"], device=self.device)
                predicted, _ = self.critic(states, hidden, continuation)
                clipped = old_values + torch.clamp(
                    predicted - old_values,
                    -float(ppo["value_clip_range"]),
                    float(ppo["value_clip_range"]),
                )
                losses = torch.maximum(
                    torch.square(predicted - returns), torch.square(clipped - returns)
                )
                value_loss = 0.5 * (losses * valid).sum() / valid.sum().clamp_min(1.0)
                critic_loss = float(ppo["value_coefficient"]) * value_loss
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError("Recurrent critic loss became non-finite")
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                if not finite_gradients(self.critic):
                    raise FloatingPointError("Recurrent critic gradients became non-finite")
                norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(ppo["max_gradient_norm"])
                )
                self.critic_optimizer.step()
                value_losses.append(float(value_loss.detach().cpu()))
                critic_norms.append(float(norm.detach().cpu()))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        critic_seconds = time.perf_counter() - critic_started
        clip_mean = float(np.mean(clip_fractions))
        if clip_mean > float(ppo.get("clipping_warning_fraction", 0.80)):
            logger.warning("PPO clipping fraction is excessive: %.3f", clip_mean)
        kl_mean = float(np.mean(approximate_kls))
        entropy_mean = float(np.mean(entropies))
        if clip_mean < 1e-4 and kl_mean < 1e-5:
            self.low_update_signal_count += 1
        else:
            self.low_update_signal_count = 0
        if self.low_update_signal_count >= int(ppo.get("stalled_update_warning_count", 20)):
            logger.warning(
                "PPO update signal has remained negligible for %d updates",
                self.low_update_signal_count,
            )
        if entropy_mean < float(ppo.get("entropy_collapse_warning", 0.05)):
            logger.warning("PPO action entropy is low: %.4f", entropy_mean)
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": entropy_mean,
            "approximate_kl": kl_mean,
            "clip_fraction": clip_mean,
            "explained_variance": explained_variance(
                rollout["old_values"].reshape(-1), rollout["returns"].reshape(-1)
            ),
            "actor_gradient_norm": float(np.mean(actor_norms)),
            "critic_gradient_norm": float(np.mean(critic_norms)),
            "actor_optimization_seconds": actor_seconds,
            "critic_optimization_seconds": critic_seconds,
        }

    def _episode_summary(self):
        if not self.completed_episodes:
            return {
                "mean_return": 0.0,
                "success_rate": 0.0,
                "cooperative_success_rate": 0.0,
                "pass_completion_rate": 0.0,
            }
        attempts = sum(item.get("valid_pass_attempts", 0) for item in self.completed_episodes)
        completions = sum(item.get("completed_receptions", 0) for item in self.completed_episodes)
        return {
            "mean_return": float(
                np.mean([item.get("team_return", 0.0) for item in self.completed_episodes])
            ),
            "success_rate": float(
                np.mean([item.get("success", 0.0) for item in self.completed_episodes])
            ),
            "cooperative_success_rate": float(
                np.mean(
                    [item.get("cooperative_success", 0.0) for item in self.completed_episodes]
                )
            ),
            "pass_completion_rate": completions / max(1, attempts),
        }

    def evaluate(self, scenario, episodes, seed_base, simulator="abstract", profile="nominal"):
        env = make_phase3_environment(
            self.config, simulator=simulator, scenario=scenario, profile_name=profile
        )
        successes = []
        cooperative = []
        returns = []
        self.actor.eval()
        try:
            for episode in range(int(episodes)):
                observations, _ = env.reset(seed=int(seed_base) + episode)
                hidden = np.zeros(
                    (1, self.agent_count, self.hidden_size), dtype=np.float32
                )
                team_return = 0.0
                while env.agents:
                    raw = np.zeros(
                        (self.agent_count, self.observation_size), dtype=np.float32
                    )
                    masks = np.zeros(
                        (self.agent_count, self.action_size), dtype=np.float32
                    )
                    for index, agent in enumerate(self.agent_names):
                        if agent in env.active_agents:
                            raw[index] = observations[agent]
                            masks[index] = env.action_mask(agent)
                    normalized = self._normalize_observations(
                        raw[None, ...],
                        np.asarray(
                            [[float(agent in env.active_agents) for agent in self.agent_names]]
                        ),
                        update=False,
                    )[0]
                    with torch.no_grad():
                        logits, next_hidden = self.actor(
                            torch.as_tensor(normalized, device=self.device),
                            torch.as_tensor(hidden, device=self.device),
                        )
                        logits = logits.masked_fill(
                            torch.as_tensor(masks, device=self.device) < 0.5, -1e9
                        )
                        selected = torch.argmax(logits, dim=-1).cpu().numpy()
                    hidden = next_hidden.cpu().numpy()
                    actions = {
                        agent: int(selected[index])
                        for index, agent in enumerate(self.agent_names)
                        if agent in env.active_agents
                    }
                    observations, rewards, _, _, infos = env.step(actions)
                    team_return += float(next(iter(rewards.values())))
                metrics = next(iter(infos.values()))["episode_metrics"]
                successes.append(float(metrics["success"]))
                cooperative.append(float(metrics["cooperative_success"]))
                returns.append(team_return)
        finally:
            env.close()
        return {
            "success_rate": float(np.mean(successes)),
            "cooperative_success_rate": float(np.mean(cooperative)),
            "mean_return": float(np.mean(returns)),
        }

    def _validation(self):
        episodes = int(self.config["train"]["validation_episodes"])
        seed_base = int(self.config["evaluation"]["seed_bases"].get("phase3_validation", 320000))
        nominal = self.evaluate(
            "phase3_2v2_open", episodes, seed_base, simulator="abstract"
        )
        cooperation = self.evaluate(
            "phase3_2v2_pass_required",
            episodes,
            seed_base + 10000,
            simulator="abstract",
        )
        robustness = None
        if self.curriculum is not None:
            profiles = {}
            profile_episodes = max(2, episodes // max(1, len(self.curriculum.profile_names)))
            for index, profile in enumerate(self.curriculum.profile_names):
                profiles[profile] = self.evaluate(
                    "phase3_2v2_open",
                    profile_episodes,
                    seed_base + 20000 + index * 1000,
                    simulator="pymunk",
                    profile=profile,
                )["success_rate"]
            profile_mean = float(np.mean(list(profiles.values())))
            grid_names = [
                name
                for name in ["delay_low", "delay_high", "localization", "combined"]
                if name in profiles
            ]
            grid_auc = float(np.mean([profiles[name] for name in grid_names]))
            robustness = {
                "profile_mean": profile_mean,
                "grid_auc": grid_auc,
                "profiles": profiles,
            }
            self.curriculum.update(
                profiles, nominal["success_rate"], self.current_update
            )
            self._write_curriculum()
        result = {
            "nominal": nominal,
            "cooperation": cooperation,
            "robustness": robustness,
        }
        self.last_validation = result
        return result

    def _composite(self, validation):
        weights = self.phase3["cc_fdr"]["composite_weights"]
        nominal = validation["nominal"]["success_rate"]
        cooperation = validation["cooperation"]["cooperative_success_rate"]
        robustness = validation["robustness"] or {
            "profile_mean": 0.0,
            "grid_auc": 0.0,
        }
        penalty = 0.0
        feasible = True
        if self.curriculum is not None and self.curriculum.nominal_reference is not None:
            floor = self.curriculum.nominal_reference - float(
                self.phase3["cc_fdr"]["nominal_regression_margin"]
            )
            penalty = max(0.0, floor - nominal)
            feasible = penalty <= 0.0
        score = (
            float(weights["nominal"]) * nominal
            + float(weights["profile_mean"]) * robustness["profile_mean"]
            + float(weights["grid_auc"]) * robustness["grid_auc"]
            + float(weights["cooperation"]) * cooperation
            - float(weights["constraint_penalty"]) * penalty
        )
        return score, feasible

    def _save_best(self, validation):
        nominal = validation["nominal"]["success_rate"]
        cooperation = validation["cooperation"]["cooperative_success_rate"]
        composite, feasible = self._composite(validation)
        candidates = [
            ("nominal", nominal, True),
            ("cooperation", cooperation, True),
            ("composite", composite, feasible),
        ]
        for name, score, allowed in candidates:
            if not allowed:
                continue
            if self.best_scores[name] is None or score > self.best_scores[name]:
                self.best_scores[name] = float(score)
                self.save_checkpoint(self.run_dir / "models" / f"best_{name}_checkpoint.pt")

    def _write_curriculum(self):
        if self.curriculum is None:
            return
        write_json(
            self.run_dir / "logs" / "cc_fdr_history.json",
            {
                "state": self.curriculum.state_dict(),
                "history": self.curriculum.history,
            },
        )

    def _decay_learning_rates(self, total_updates):
        if not self.config["ppo"].get("linear_learning_rate_decay", True):
            return
        fraction = max(0.0, 1.0 - self.current_update / max(1, total_updates))
        actor_min = float(self.config["ppo"].get("actor_min_learning_rate", 0.0))
        critic_min = float(self.config["ppo"].get("critic_min_learning_rate", 0.0))
        actor_rate = max(
            actor_min, float(self.config["ppo"]["actor_learning_rate"]) * fraction
        )
        critic_rate = max(
            critic_min, float(self.config["ppo"]["critic_learning_rate"]) * fraction
        )
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_rate
        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_rate
        if actor_rate <= actor_min or critic_rate <= critic_min:
            logger.warning(
                "Learning-rate floor active: actor %.2e critic %.2e",
                actor_rate,
                critic_rate,
            )

    def checkpoint_payload(self):
        return {
            "checkpoint_schema": 1,
            "phase3_observation_schema": 1,
            "phase3_state_schema": 1,
            "observation_size": self.observation_size,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "agent_count": self.agent_count,
            "scenario_schema": int(self.phase3.get("schema_version", 1)),
            "actor_weights": self.actor.state_dict(),
            "critic_weights": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "observation_normalization": self.observation_rms.state_dict(),
            "global_state_normalization": self.state_rms.state_dict(),
            "current_update": self.current_update,
            "environment_steps": self.environment_steps,
            "stage_start_steps": self.stage_start_steps,
            "active_stage": self.stage_name,
            "mode": self.mode,
            "recurrent_hidden_size": self.hidden_size,
            "best_scores": self.best_scores,
            "last_validation": self.last_validation,
            "curriculum": self.curriculum.state_dict() if self.curriculum is not None else None,
            "numpy_random_state": np.random.get_state(),
            "generator_state": self.rng.bit_generator.state,
            "python_random_state": random.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "resolved_configuration": self.config,
        }

    def save_checkpoint(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)
        return destination

    def load_checkpoint(self, path, warm_start=False):
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError("Checkpoint does not exist: " + str(source))
        checkpoint = torch.load(source, map_location=self.device, weights_only=False)
        if int(checkpoint.get("recurrent_hidden_size", -1)) != self.hidden_size:
            raise ValueError("Recurrent hidden size differs from checkpoint")
        if int(checkpoint.get("phase3_observation_schema", -1)) != 1:
            raise ValueError("Phase 3 observation schema differs from checkpoint")
        if int(checkpoint.get("phase3_state_schema", -1)) != 1:
            raise ValueError("Phase 3 state schema differs from checkpoint")
        expected_dimensions = {
            "observation_size": self.observation_size,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "agent_count": self.agent_count,
            "scenario_schema": int(self.phase3.get("schema_version", 1)),
        }
        for name, expected in expected_dimensions.items():
            if int(checkpoint.get(name, -1)) != int(expected):
                raise ValueError(
                    "Phase 3 " + name + " differs from checkpoint"
                )
        self.actor.load_state_dict(checkpoint["actor_weights"])
        self.critic.load_state_dict(checkpoint["critic_weights"])
        self.observation_rms.load_state_dict(checkpoint["observation_normalization"])
        self.state_rms.load_state_dict(checkpoint["global_state_normalization"])
        if warm_start:
            if self.mode == "cc_fdr":
                reference = checkpoint.get("last_validation", {})
                self.curriculum.nominal_reference = (
                    reference.get("nominal", {}).get("success_rate")
                )
                if self.curriculum.nominal_reference is None:
                    raise ValueError(
                        "CC-FDR warm start checkpoint lacks nominal validation reference"
                    )
            self.stage_start_steps = 0
            logger.info("Warm-started Phase 3 weights from %s", source)
            return
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.current_update = int(checkpoint["current_update"])
        self.environment_steps = int(checkpoint["environment_steps"])
        self.stage_start_steps = int(checkpoint.get("stage_start_steps", 0))
        self.best_scores = copy.deepcopy(checkpoint.get("best_scores", self.best_scores))
        self.last_validation = copy.deepcopy(checkpoint.get("last_validation"))
        if self.curriculum is not None and checkpoint.get("curriculum") is not None:
            self.curriculum.load_state_dict(checkpoint["curriculum"])
        np.random.set_state(checkpoint["numpy_random_state"])
        self.rng.bit_generator.state = checkpoint["generator_state"]
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
        logger.info(
            "Resumed Phase 3 update %d at %d environment steps from %s",
            self.current_update,
            self.environment_steps,
            source,
        )

    def export_actor(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        actor = copy.deepcopy(self.actor).to("cpu").eval()
        deployment = MaskedRecurrentDeploymentActor(actor).eval()
        observations = torch.zeros(1, self.observation_size)
        hidden = torch.zeros(1, 1, self.hidden_size)
        action_masks = torch.ones(1, self.action_size)
        traced = torch.jit.trace(
            deployment,
            (observations, hidden, action_masks),
            check_trace=False,
            strict=False,
        )
        traced.save(str(destination))
        return destination

    def train(self):
        if self.mode == "cc_fdr":
            target_steps = self.environment_steps + int(self.config["train"]["total_steps"])
        else:
            target_steps = int(self.stage["target_steps"])
        steps_per_update = self.num_envs * self.rollout_steps
        remaining = max(0, target_steps - self.environment_steps)
        total_updates = self.current_update + math.ceil(remaining / steps_per_update)
        metrics_writer = MetricsWriter(
            self.run_dir, enabled=self.config["experiment"].get("tensorboard", True)
        )
        progress = tqdm(
            total=max(target_steps, self.environment_steps),
            initial=min(self.environment_steps, target_steps),
            desc=f"Phase3 {self.stage_name}",
            disable=not self.config["train"].get("progress_bar", True),
            file=sys.stdout,
            dynamic_ncols=True,
            unit="env-step",
            mininterval=float(self.config["train"].get("progress_interval_seconds", 1.0)),
        )
        started = time.perf_counter()
        try:
            while self.current_update < total_updates:
                self.current_update += 1
                self._decay_learning_rates(total_updates)
                rollout_started = time.perf_counter()
                rollout = self.collect_rollout()
                rollout_seconds = time.perf_counter() - rollout_started
                update_started = time.perf_counter()
                update_metrics = self.ppo_update(rollout)
                update_seconds = time.perf_counter() - update_started
                episode = self._episode_summary()
                validation = None
                validation_seconds = 0.0
                frequency = int(self.config["train"]["validation_frequency_steps"])
                if (
                    self.environment_steps % max(steps_per_update, frequency) < steps_per_update
                    or self.current_update == total_updates
                ):
                    validation_started = time.perf_counter()
                    validation = self._validation()
                    self._save_best(validation)
                    validation_seconds = time.perf_counter() - validation_started
                checkpoint_seconds = 0.0
                checkpoint_frequency = int(
                    self.config["train"]["checkpoint_frequency_steps"]
                )
                if (
                    self.environment_steps % max(steps_per_update, checkpoint_frequency)
                    < steps_per_update
                    or self.current_update == total_updates
                ):
                    checkpoint_started = time.perf_counter()
                    self.save_checkpoint(
                        self.run_dir
                        / "checkpoints"
                        / f"checkpoint_step_{self.environment_steps}.pt"
                    )
                    checkpoint_seconds = time.perf_counter() - checkpoint_started
                cuda_allocated = 0
                cuda_reserved = 0
                cuda_maximum = 0
                if self.device.type == "cuda":
                    cuda_allocated = int(torch.cuda.memory_allocated(self.device))
                    cuda_reserved = int(torch.cuda.memory_reserved(self.device))
                    cuda_maximum = int(torch.cuda.max_memory_allocated(self.device))
                actor_learning_rate = self.actor_optimizer.param_groups[0]["lr"]
                critic_learning_rate = self.critic_optimizer.param_groups[0]["lr"]
                row = {
                    "update": self.current_update,
                    "environment_steps": self.environment_steps,
                    "stage": self.stage_name,
                    "mode": self.mode,
                    "mean_episodic_return": episode["mean_return"],
                    "success_rate": episode["success_rate"],
                    "cooperative_success_rate": episode["cooperative_success_rate"],
                    "pass_completion_rate": episode["pass_completion_rate"],
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": update_seconds,
                    **self.last_rollout_timing,
                    "validation_seconds": validation_seconds,
                    "checkpoint_io_seconds": checkpoint_seconds,
                    "logging_io_seconds_previous_update": self.last_logging_io_seconds,
                    "transitions_per_second": steps_per_update
                    / max(1e-9, rollout_seconds + update_seconds),
                    "agent_steps_per_second": steps_per_update
                    * self.agent_count
                    / max(1e-9, rollout_seconds + update_seconds),
                    "cpu_count": os.cpu_count(),
                    "num_envs": self.num_envs,
                    "actor_minibatch_size": int(self.config["ppo"]["actor_minibatch_size"]),
                    "critic_minibatch_size": int(
                        self.config["ppo"]["critic_minibatch_size"]
                    ),
                    "recurrent_sequence_length": int(
                        self.recurrent["sequence_length"]
                    ),
                    "actor_learning_rate": actor_learning_rate,
                    "critic_learning_rate": critic_learning_rate,
                    "cuda_allocated_bytes": cuda_allocated,
                    "cuda_reserved_bytes": cuda_reserved,
                    "cuda_maximum_bytes": cuda_maximum,
                    **update_metrics,
                    "validation_nominal_success": ""
                    if validation is None
                    else validation["nominal"]["success_rate"],
                    "validation_cooperative_success": ""
                    if validation is None
                    else validation["cooperation"]["cooperative_success_rate"],
                    "validation_profile_mean": ""
                    if validation is None
                    or validation["robustness"] is None
                    else validation["robustness"]["profile_mean"],
                    "validation_grid_auc": ""
                    if validation is None
                    or validation["robustness"] is None
                    else validation["robustness"]["grid_auc"],
                    "nominal_rehearsal_probability": ""
                    if self.curriculum is None
                    else self.curriculum.nominal_probability,
                    "constraint_guard_events": ""
                    if self.curriculum is None
                    else self.curriculum.guard_events,
                }
                logging_started = time.perf_counter()
                metrics_writer.write(row)
                self.last_logging_io_seconds = time.perf_counter() - logging_started
                validation_display = (
                    "-"
                    if validation is None
                    else f"{validation['nominal']['success_rate']:.2f}"
                )
                gpu_display = f"{cuda_allocated / (1024**2):.0f}M"
                progress.set_postfix(
                    step=f"{self.environment_steps}/{target_steps}",
                    sps=f"{row['transitions_per_second']:.0f}",
                    success=f"{episode['success_rate']:.2f}",
                    ret=f"{episode['mean_return']:.2f}",
                    val=validation_display,
                    coop=f"{episode['cooperative_success_rate']:.2f}",
                    kl=f"{update_metrics['approximate_kl']:.3f}",
                    ent=f"{update_metrics['entropy']:.2f}",
                    lr=f"{actor_learning_rate:.1e}",
                    gpu=gpu_display,
                )
                progress.update(min(steps_per_update, target_steps - progress.n))
        finally:
            progress.close()
            metrics_writer.close()
        plot_phase3_diagnostics(self.run_dir, self.curriculum)
        final_checkpoint = self.save_checkpoint(
            self.run_dir / "models" / "final_checkpoint.pt"
        )
        if not (self.run_dir / "models" / "best_nominal_checkpoint.pt").is_file():
            self.save_checkpoint(self.run_dir / "models" / "best_nominal_checkpoint.pt")
        actor_path = self.export_actor(self.run_dir / "models" / "final_actor.ts")
        frame = self.environments[0].render()
        from PIL import Image

        Image.fromarray(frame).save(self.run_dir / "videos" / "render_check.png")
        write_json(
            self.run_dir / "logs" / "phase3_training_summary.json",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "environment_steps": self.environment_steps,
                "updates": self.current_update,
                "stage": self.stage_name,
                "mode": self.mode,
                "best_scores": self.best_scores,
                "last_validation": self.last_validation,
                "full_scientific_run": self.environment_steps >= 500000,
            },
        )
        return {
            "final_checkpoint": str(final_checkpoint),
            "best_checkpoint": str(
                self.run_dir / "models" / "best_nominal_checkpoint.pt"
            ),
            "final_actor": str(actor_path),
            "metrics_csv": str(self.run_dir / "logs" / "metrics.csv"),
            "render_check": str(self.run_dir / "videos" / "render_check.png"),
            "environment_steps": self.environment_steps,
            "updates": self.current_update,
            "stage": self.stage_name,
            "mode": self.mode,
        }

    def close(self):
        for env in self.environments:
            env.close()
        self.environments = []
