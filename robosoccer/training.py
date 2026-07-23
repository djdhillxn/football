"""Compact parameter-shared IPPO/MAPPO training and adaptive randomization."""

import copy
import csv
import json
import logging
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from robosoccer.config import save_config
from robosoccer.environment import AGENTS, available_profile_names, make_environment
from robosoccer.utils import (
    MetricsWriter,
    RunningMeanStd,
    check_finite,
    create_run_directory,
    finalize_run,
    get_pyplot,
    initial_metadata,
    select_device,
    set_global_seeds,
    setup_logging,
    utc_now,
    write_json,
)

logger = logging.getLogger(__name__)


def activation_class(name):
    if str(name).lower() == "tanh":
        return nn.Tanh
    if str(name).lower() == "relu":
        return nn.ReLU
    raise ValueError("Unsupported model activation: " + str(name))


def build_mlp(input_size, hidden_sizes, output_size, activation, orthogonal, output_gain=1.0):
    layers = []
    previous = input_size
    activation_type = activation_class(activation)
    for hidden in hidden_sizes:
        layer = nn.Linear(previous, int(hidden))
        if orthogonal:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.constant_(layer.bias, 0.0)
        layers.extend([layer, activation_type()])
        previous = int(hidden)
    output = nn.Linear(previous, output_size)
    if orthogonal:
        nn.init.orthogonal_(output.weight, gain=output_gain)
        nn.init.constant_(output.bias, 0.0)
    layers.append(output)
    return nn.Sequential(*layers)


class SharedActor(nn.Module):
    """Categorical actor shared by both attackers and restricted to local observations."""

    def __init__(self, observation_size, action_size, model_config):
        super().__init__()
        self.network = build_mlp(
            observation_size,
            model_config["actor_hidden_sizes"],
            action_size,
            model_config["activation"],
            model_config.get("orthogonal_init", True),
            output_gain=0.01,
        )

    def forward(self, observation):
        return self.network(observation)


class ValueNetwork(nn.Module):
    """Local IPPO critic or global MAPPO team critic."""

    def __init__(self, input_size, hidden_sizes, model_config):
        super().__init__()
        self.network = build_mlp(
            input_size,
            hidden_sizes,
            1,
            model_config["activation"],
            model_config.get("orthogonal_init", True),
            output_gain=1.0,
        )

    def forward(self, observation):
        return self.network(observation).squeeze(-1)


def build_networks(config, observation_size, state_size):
    actor = SharedActor(observation_size, 7, config["model"])
    if config["ppo"]["algorithm"] == "ippo":
        critic = ValueNetwork(
            observation_size, config["model"]["local_critic_hidden_sizes"], config["model"]
        )
    else:
        critic = ValueNetwork(
            state_size, config["model"]["central_critic_hidden_sizes"], config["model"]
        )
    return actor, critic


def compute_gae(rewards, values, next_values, terminations, truncations, gamma, gae_lambda):
    """Compute GAE, bootstrapping at time limits but never across episode resets."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminations = np.asarray(terminations, dtype=np.float32)
    truncations = np.asarray(truncations, dtype=np.float32)
    if not (rewards.shape == values.shape == next_values.shape == terminations.shape == truncations.shape):
        raise ValueError("GAE inputs must have identical shapes")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros(rewards.shape[1:], dtype=np.float32)
    for index in reversed(range(rewards.shape[0])):
        bootstrap_mask = 1.0 - terminations[index]
        continuation_mask = 1.0 - np.maximum(terminations[index], truncations[index])
        delta = rewards[index] + gamma * next_values[index] * bootstrap_mask - values[index]
        running = delta + gamma * gae_lambda * continuation_mask * running
        advantages[index] = running
    returns = advantages + values
    check_finite("advantages", advantages)
    check_finite("returns", returns)
    return advantages, returns


def capped_distribution(probabilities, maximum):
    """Project nonnegative weights onto a probability simplex with a hard upper cap."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Probability vector must be non-empty and one-dimensional")
    if maximum * len(values) < 1.0 - 1e-10:
        raise ValueError("maximum_profile_probability is too small for the profile count")
    values = np.maximum(values, 0.0)
    if values.sum() <= 0:
        values = np.ones_like(values)
    remaining = np.ones(len(values), dtype=bool)
    result = np.zeros(len(values), dtype=np.float64)
    mass = 1.0
    while np.any(remaining):
        weights = values[remaining]
        if weights.sum() <= 0:
            proposal = np.full(weights.shape, mass / len(weights))
        else:
            proposal = weights / weights.sum() * mass
        indices = np.flatnonzero(remaining)
        over = proposal > maximum + 1e-12
        if not np.any(over):
            result[indices] = proposal
            break
        capped_indices = indices[over]
        result[capped_indices] = maximum
        remaining[capped_indices] = False
        mass = 1.0 - result.sum()
    result /= result.sum()
    return result


class FailureDirectedCurriculum:
    """Failure-rate EMA and capped sampling distribution over perturbation profiles."""

    def __init__(self, profile_names, curriculum_config):
        if not profile_names:
            raise ValueError("Failure-directed curriculum requires at least one profile")
        self.profile_names = list(profile_names)
        self.config = copy.deepcopy(curriculum_config)
        uniform = 1.0 / len(self.profile_names)
        self.probabilities = {name: uniform for name in self.profile_names}
        self.smoothed_failures = {name: 0.5 for name in self.profile_names}
        self.history = []

    def update(self, success_rates, update_number):
        decay = float(self.config["ema_decay"])
        epsilon = float(self.config["epsilon"])
        alpha = float(self.config["alpha"])
        beta = float(self.config["uniform_mixture_beta"])
        raw_failures = {}
        for name in self.profile_names:
            if name not in success_rates:
                raise ValueError("Missing curriculum success rate for profile: " + name)
            failure = 1.0 - float(success_rates[name])
            raw_failures[name] = failure
            self.smoothed_failures[name] = (
                decay * self.smoothed_failures[name] + (1.0 - decay) * failure
            )
        weights = np.array(
            [(epsilon + self.smoothed_failures[name]) ** alpha for name in self.profile_names],
            dtype=np.float64,
        )
        weights /= weights.sum()
        mixed = (1.0 - beta) * weights + beta / len(weights)
        mixed = capped_distribution(mixed, float(self.config["maximum_profile_probability"]))
        self.probabilities = {
            name: float(mixed[index]) for index, name in enumerate(self.profile_names)
        }
        entropy = -float(np.sum(mixed * np.log(mixed + 1e-12)))
        for name in self.profile_names:
            self.history.append(
                {
                    "update": int(update_number),
                    "profile": name,
                    "success_rate": float(success_rates[name]),
                    "raw_failure": raw_failures[name],
                    "smoothed_failure": self.smoothed_failures[name],
                    "sampling_probability": self.probabilities[name],
                    "sampling_entropy": entropy,
                }
            )
        return copy.deepcopy(self.probabilities)

    def state_dict(self):
        return {
            "profile_names": self.profile_names,
            "probabilities": self.probabilities,
            "smoothed_failures": self.smoothed_failures,
            "history": self.history,
        }

    def load_state_dict(self, state):
        if list(state["profile_names"]) != self.profile_names:
            raise ValueError("Checkpoint curriculum profiles do not match the resolved configuration")
        self.probabilities = copy.deepcopy(state["probabilities"])
        self.smoothed_failures = copy.deepcopy(state["smoothed_failures"])
        self.history = copy.deepcopy(state.get("history", []))

    def entropy(self):
        values = np.array(list(self.probabilities.values()), dtype=np.float64)
        return -float(np.sum(values * np.log(values + 1e-12)))


def finite_gradients(module):
    for parameter in module.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def explained_variance(predictions, targets):
    variance = np.var(targets)
    if variance < 1e-12:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / variance)


class PPOTrainer:
    """Synchronous parameter-shared PPO trainer supporting IPPO and MAPPO critics."""

    def __init__(self, config, run_dir, resume_path=None):
        self.config = copy.deepcopy(config)
        self.run_dir = Path(run_dir)
        self.device = select_device(config["train"]["device"])
        self.algorithm = config["ppo"]["algorithm"]
        self.num_envs = int(config["train"]["num_envs"])
        self.rollout_steps = int(config["train"]["rollout_steps"])
        self.environments = []
        self.observations = []
        self.states = []
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.completed_episodes = []
        self.current_update = 0
        self.environment_steps = 0
        self.best_validation_score = None
        self._last_checkpoint_step = 0
        self._last_validation_step = 0
        self.seed = int(config["experiment"]["seed"])
        self.training_seed_counter = 0

        probe = make_environment(config, "abstract")
        self.observation_size = probe.observation_dimension
        self.state_size = probe.state_dimension
        probe.close()
        self.actor, self.critic = build_networks(config, self.observation_size, self.state_size)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(config["ppo"]["actor_learning_rate"]), eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(config["ppo"]["critic_learning_rate"]), eps=1e-5
        )
        self.observation_rms = RunningMeanStd((self.observation_size,))
        self.state_rms = RunningMeanStd((self.state_size,))
        self.return_rms = RunningMeanStd(())
        self.curriculum = None
        if config["randomization"]["mode"] == "failure_directed":
            profile_names = available_profile_names(config, include_nominal=False)
            self.curriculum = FailureDirectedCurriculum(profile_names, config["curriculum"])
        self._create_environments()
        if resume_path is not None:
            self.load_checkpoint(resume_path)
        self._reset_environment_batch()

    def _create_environments(self):
        probabilities = self.curriculum.probabilities if self.curriculum is not None else None
        for _ in range(self.num_envs):
            env = make_environment(self.config, "abstract", profile_probabilities=probabilities)
            self.environments.append(env)

    def _reset_environment_batch(self):
        self.observations = []
        self.states = []
        for index, env in enumerate(self.environments):
            observations, _ = env.reset(seed=self.seed + index)
            self.observations.append(observations)
            self.states.append(env.state())

    def _normalize_observations(self, values, update=True):
        values = np.asarray(values, dtype=np.float32)
        if not self.config["observations"].get("normalize", True):
            return values
        if update:
            self.observation_rms.update(values)
        return self.observation_rms.normalize(values, self.config["observations"]["clip"])

    def _normalize_states(self, values, update=True):
        values = np.asarray(values, dtype=np.float32)
        if not self.config["observations"].get("normalize", True):
            return values
        if update:
            self.state_rms.update(values)
        return self.state_rms.normalize(values, self.config["observations"]["clip"])

    def collect_rollout(self):
        time_steps = self.rollout_steps
        agent_count = len(AGENTS)
        observations = np.zeros(
            (time_steps, self.num_envs, agent_count, self.observation_size), dtype=np.float32
        )
        states = np.zeros((time_steps, self.num_envs, self.state_size), dtype=np.float32)
        actions = np.zeros((time_steps, self.num_envs, agent_count), dtype=np.int64)
        log_probabilities = np.zeros((time_steps, self.num_envs, agent_count), dtype=np.float32)
        rewards = np.zeros((time_steps, self.num_envs), dtype=np.float32)
        terminations = np.zeros((time_steps, self.num_envs), dtype=np.float32)
        truncations = np.zeros((time_steps, self.num_envs), dtype=np.float32)
        if self.algorithm == "ippo":
            values = np.zeros((time_steps, self.num_envs, agent_count), dtype=np.float32)
            next_values = np.zeros_like(values)
        else:
            values = np.zeros((time_steps, self.num_envs), dtype=np.float32)
            next_values = np.zeros_like(values)

        self.completed_episodes = []
        self.actor.eval()
        self.critic.eval()
        for step_index in range(time_steps):
            raw_observations = np.asarray(
                [
                    [self.observations[env_index][agent] for agent in AGENTS]
                    for env_index in range(self.num_envs)
                ],
                dtype=np.float32,
            )
            raw_states = np.asarray(self.states, dtype=np.float32)
            normalized_observations = self._normalize_observations(
                raw_observations.reshape(-1, self.observation_size), update=True
            ).reshape(raw_observations.shape)
            normalized_states = self._normalize_states(raw_states, update=True)
            observations[step_index] = normalized_observations
            states[step_index] = normalized_states
            with torch.no_grad():
                observation_tensor = torch.as_tensor(
                    normalized_observations.reshape(-1, self.observation_size), device=self.device
                )
                distribution = Categorical(logits=self.actor(observation_tensor))
                sampled_actions = distribution.sample()
                sampled_logs = distribution.log_prob(sampled_actions)
                actions[step_index] = sampled_actions.cpu().numpy().reshape(self.num_envs, agent_count)
                log_probabilities[step_index] = sampled_logs.cpu().numpy().reshape(
                    self.num_envs, agent_count
                )
                if self.algorithm == "ippo":
                    critic_values = self.critic(observation_tensor).reshape(self.num_envs, agent_count)
                else:
                    state_tensor = torch.as_tensor(normalized_states, device=self.device)
                    critic_values = self.critic(state_tensor)
                values[step_index] = critic_values.cpu().numpy()

            for env_index, env in enumerate(self.environments):
                action_dict = {
                    agent: int(actions[step_index, env_index, agent_index])
                    for agent_index, agent in enumerate(AGENTS)
                }
                next_observations, reward_dict, terminated_dict, truncated_dict, infos = env.step(
                    action_dict
                )
                reward = float(reward_dict[AGENTS[0]])
                rewards[step_index, env_index] = reward
                self.episode_returns[env_index] += reward
                self.episode_lengths[env_index] += 1
                terminated = bool(terminated_dict[AGENTS[0]])
                truncated = bool(truncated_dict[AGENTS[0]])
                terminations[step_index, env_index] = float(terminated)
                truncations[step_index, env_index] = float(truncated)
                final_observations = np.asarray(
                    [env.observe(agent) for agent in AGENTS], dtype=np.float32
                )
                final_state = env.state()
                normalized_final_observations = self._normalize_observations(
                    final_observations, update=False
                )
                normalized_final_state = self._normalize_states(final_state, update=False)
                with torch.no_grad():
                    if self.algorithm == "ippo":
                        next_value = self.critic(
                            torch.as_tensor(normalized_final_observations, device=self.device)
                        ).cpu().numpy()
                    else:
                        next_value = self.critic(
                            torch.as_tensor(normalized_final_state[None, :], device=self.device)
                        ).cpu().numpy()[0]
                next_values[step_index, env_index] = next_value
                if terminated or truncated:
                    episode_metrics = copy.deepcopy(infos[AGENTS[0]].get("episode_metrics", {}))
                    episode_metrics["team_return"] = float(self.episode_returns[env_index])
                    episode_metrics["episode_steps"] = int(self.episode_lengths[env_index])
                    self.completed_episodes.append(episode_metrics)
                    reset_observations, _ = env.reset()
                    self.observations[env_index] = reset_observations
                    self.states[env_index] = env.state()
                    self.episode_returns[env_index] = 0.0
                    self.episode_lengths[env_index] = 0
                else:
                    self.observations[env_index] = next_observations
                    self.states[env_index] = final_state

        learning_rewards = rewards.copy()
        if self.config["observations"].get("reward_normalize", False):
            self.return_rms.update(learning_rewards.reshape(-1))
            learning_rewards /= float(np.sqrt(self.return_rms.var + 1e-8))
        if self.algorithm == "ippo":
            expanded_rewards = np.repeat(learning_rewards[:, :, None], agent_count, axis=2)
            expanded_terminations = np.repeat(terminations[:, :, None], agent_count, axis=2)
            expanded_truncations = np.repeat(truncations[:, :, None], agent_count, axis=2)
            advantages, returns = compute_gae(
                expanded_rewards,
                values,
                next_values,
                expanded_terminations,
                expanded_truncations,
                self.config["ppo"]["gamma"],
                self.config["ppo"]["gae_lambda"],
            )
            actor_advantages = advantages
        else:
            advantages, returns = compute_gae(
                learning_rewards,
                values,
                next_values,
                terminations,
                truncations,
                self.config["ppo"]["gamma"],
                self.config["ppo"]["gae_lambda"],
            )
            actor_advantages = np.repeat(advantages[:, :, None], agent_count, axis=2)
        self.environment_steps += time_steps * self.num_envs
        return {
            "observations": observations,
            "states": states,
            "actions": actions,
            "old_log_probabilities": log_probabilities,
            "old_values": values,
            "advantages": actor_advantages,
            "returns": returns,
            "raw_rewards": rewards,
        }

    def ppo_update(self, rollout):
        self.actor.train()
        self.critic.train()
        ppo = self.config["ppo"]
        observation_batch = rollout["observations"].reshape(-1, self.observation_size)
        action_batch = rollout["actions"].reshape(-1)
        old_log_batch = rollout["old_log_probabilities"].reshape(-1)
        advantage_batch = rollout["advantages"].reshape(-1)
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
        actor_count = len(action_batch)
        actor_minibatch = min(int(ppo["actor_minibatch_size"]), actor_count)
        if actor_minibatch < int(ppo["actor_minibatch_size"]):
            logger.debug("Adjusted actor minibatch from %s to %s", ppo["actor_minibatch_size"], actor_minibatch)
        policy_losses = []
        entropies = []
        approximate_kls = []
        clip_fractions = []
        actor_gradient_norms = []
        stop_actor = False
        for _ in range(int(ppo["update_epochs"])):
            permutation = np.random.permutation(actor_count)
            for start in range(0, actor_count, actor_minibatch):
                indices = permutation[start : start + actor_minibatch]
                observations = torch.as_tensor(observation_batch[indices], device=self.device)
                actions = torch.as_tensor(action_batch[indices], device=self.device)
                old_logs = torch.as_tensor(old_log_batch[indices], device=self.device)
                advantages = torch.as_tensor(advantage_batch[indices], device=self.device)
                distribution = Categorical(logits=self.actor(observations))
                new_logs = distribution.log_prob(actions)
                entropy = distribution.entropy().mean()
                log_ratio = new_logs - old_logs
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - ppo["clip_range"], 1.0 + ppo["clip_range"]) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                actor_loss = policy_loss - ppo["entropy_coefficient"] * entropy
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Actor loss became NaN or infinity")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if not finite_gradients(self.actor):
                    raise FloatingPointError("Actor gradients became NaN or infinity")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), float(ppo["max_gradient_norm"])
                )
                self.actor_optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > ppo["clip_range"]).float().mean()
                policy_losses.append(float(policy_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                approximate_kls.append(float(approximate_kl.cpu()))
                clip_fractions.append(float(clip_fraction.cpu()))
                actor_gradient_norms.append(float(gradient_norm.detach().cpu()))
                if approximate_kls[-1] > float(ppo["target_kl"]):
                    stop_actor = True
                    break
            if stop_actor:
                logger.debug("Actor epochs stopped early at approximate KL %.5f", approximate_kls[-1])
                break

        if self.algorithm == "ippo":
            critic_inputs = observation_batch
            old_values = rollout["old_values"].reshape(-1)
            returns = rollout["returns"].reshape(-1)
        else:
            critic_inputs = rollout["states"].reshape(-1, self.state_size)
            old_values = rollout["old_values"].reshape(-1)
            returns = rollout["returns"].reshape(-1)
        critic_count = len(returns)
        critic_minibatch = min(int(ppo["critic_minibatch_size"]), critic_count)
        value_losses = []
        critic_gradient_norms = []
        for _ in range(int(ppo["update_epochs"])):
            permutation = np.random.permutation(critic_count)
            for start in range(0, critic_count, critic_minibatch):
                indices = permutation[start : start + critic_minibatch]
                inputs = torch.as_tensor(critic_inputs[indices], device=self.device)
                old_value = torch.as_tensor(old_values[indices], device=self.device)
                target = torch.as_tensor(returns[indices], device=self.device)
                predicted = self.critic(inputs)
                if ppo.get("value_clip_range") is not None:
                    clipped_value = old_value + torch.clamp(
                        predicted - old_value,
                        -float(ppo["value_clip_range"]),
                        float(ppo["value_clip_range"]),
                    )
                    loss_unclipped = torch.square(predicted - target)
                    loss_clipped = torch.square(clipped_value - target)
                    value_loss = 0.5 * torch.maximum(loss_unclipped, loss_clipped).mean()
                else:
                    value_loss = 0.5 * torch.square(predicted - target).mean()
                critic_loss = float(ppo["value_coefficient"]) * value_loss
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError("Critic loss became NaN or infinity")
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                if not finite_gradients(self.critic):
                    raise FloatingPointError("Critic gradients became NaN or infinity")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(ppo["max_gradient_norm"])
                )
                self.critic_optimizer.step()
                value_losses.append(float(value_loss.detach().cpu()))
                critic_gradient_norms.append(float(gradient_norm.detach().cpu()))
        if entropies and np.mean(entropies) < 0.08:
            logger.warning("Policy entropy is very low (%.4f); inspect for premature collapse.", np.mean(entropies))
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approximate_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": explained_variance(old_values, returns),
            "actor_gradient_norm": float(np.mean(actor_gradient_norms)),
            "critic_gradient_norm": float(np.mean(critic_gradient_norms)),
        }

    def _episode_summary(self):
        if not self.completed_episodes:
            return {"mean_return": 0.0, "success_rate": 0.0, "mean_length": 0.0}
        returns = [episode.get("team_return", 0.0) for episode in self.completed_episodes]
        successes = [float(episode.get("success", False)) for episode in self.completed_episodes]
        lengths = [episode.get("episode_steps", 0) for episode in self.completed_episodes]
        return {
            "mean_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lengths)),
        }

    def _deterministic_evaluation(self, episodes, seed_base, profile=None, simulator="abstract"):
        successes = []
        returns = []
        times = []
        env = make_environment(self.config, simulator, profile_name=profile)
        self.actor.eval()
        try:
            for episode in range(int(episodes)):
                observations, _ = env.reset(seed=int(seed_base) + episode)
                team_return = 0.0
                while env.agents:
                    batch = np.asarray([[observations[agent] for agent in AGENTS]], dtype=np.float32)
                    normalized = self._normalize_observations(
                        batch.reshape(-1, self.observation_size), update=False
                    )
                    with torch.no_grad():
                        logits = self.actor(torch.as_tensor(normalized, device=self.device))
                        selected = torch.argmax(logits, dim=-1).cpu().numpy()
                    action_dict = {agent: int(selected[index]) for index, agent in enumerate(AGENTS)}
                    observations, rewards, _, _, infos = env.step(action_dict)
                    team_return += float(rewards[AGENTS[0]])
                metrics = infos[AGENTS[0]]["episode_metrics"]
                successes.append(float(metrics.get("success", False)))
                returns.append(team_return)
                if metrics.get("time_to_score") is not None:
                    times.append(float(metrics["time_to_score"]))
        finally:
            env.close()
        return {
            "success_rate": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "mean_time_to_score": float(np.mean(times)) if times else None,
        }

    def _curriculum_update(self):
        seed_base = self.config["evaluation"]["seed_bases"]["curriculum"]
        episodes = int(self.config["curriculum"]["episodes_per_profile"])
        success_rates = {}
        for profile_index, profile in enumerate(self.curriculum.profile_names):
            summary = self._deterministic_evaluation(
                episodes, seed_base + profile_index * 1000, profile=profile
            )
            success_rates[profile] = summary["success_rate"]
        probabilities = self.curriculum.update(success_rates, self.current_update)
        for env in self.environments:
            env.set_profile_probabilities(probabilities)
        self._write_curriculum_history()
        return probabilities

    def _write_curriculum_history(self):
        if self.curriculum is None:
            return
        path = self.run_dir / "logs" / "curriculum_history.csv"
        if not self.curriculum.history:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.curriculum.history[0]))
            writer.writeheader()
            writer.writerows(self.curriculum.history)

    def _decay_learning_rates(self, total_updates):
        if not self.config["ppo"].get("linear_learning_rate_decay", True):
            return
        fraction = max(0.0, 1.0 - (self.current_update - 1) / max(total_updates, 1))
        actor_rate = self.config["ppo"]["actor_learning_rate"] * fraction
        critic_rate = self.config["ppo"]["critic_learning_rate"] * fraction
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_rate
        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_rate

    def checkpoint_payload(self):
        return {
            "actor_weights": self.actor.state_dict(),
            "critic_weights": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "observation_normalization": self.observation_rms.state_dict(),
            "global_state_normalization": self.state_rms.state_dict(),
            "return_normalization": self.return_rms.state_dict(),
            "current_update": self.current_update,
            "environment_steps": self.environment_steps,
            "curriculum": self.curriculum.state_dict() if self.curriculum is not None else None,
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "best_validation_score": self.best_validation_score,
            "resolved_configuration": self.config,
        }

    def save_checkpoint(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)
        return destination

    def load_checkpoint(self, path):
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError("Checkpoint does not exist: " + str(source))
        checkpoint = torch.load(source, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_weights"])
        self.critic.load_state_dict(checkpoint["critic_weights"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.observation_rms.load_state_dict(checkpoint["observation_normalization"])
        self.state_rms.load_state_dict(checkpoint["global_state_normalization"])
        self.return_rms.load_state_dict(checkpoint["return_normalization"])
        self.current_update = int(checkpoint["current_update"])
        self.environment_steps = int(checkpoint["environment_steps"])
        self.best_validation_score = checkpoint.get("best_validation_score")
        if self.curriculum is not None and checkpoint.get("curriculum") is not None:
            self.curriculum.load_state_dict(checkpoint["curriculum"])
            for env in self.environments:
                env.set_profile_probabilities(self.curriculum.probabilities)
        np.random.set_state(checkpoint["numpy_random_state"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
        logger.info(
            "Resumed update %d at %s environment steps from %s.",
            self.current_update,
            f"{self.environment_steps:,}",
            source,
        )

    def export_actor(self, path, actor_state=None):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = copy.deepcopy(self.actor.state_dict())
        if actor_state is not None:
            self.actor.load_state_dict(actor_state)
        actor_copy = copy.deepcopy(self.actor).to("cpu").eval()
        scripted = torch.jit.script(actor_copy)
        scripted.save(str(destination))
        self.actor.load_state_dict(original)
        self.actor.to(self.device)
        return destination

    def train(self):
        total_steps = int(self.config["train"]["total_steps"])
        steps_per_update = self.num_envs * self.rollout_steps
        total_updates = math.ceil(total_steps / steps_per_update)
        metrics_writer = MetricsWriter(
            self.run_dir, enabled=self.config["experiment"].get("tensorboard", True)
        )
        logger.info(
            "Training %s with %d environments for %s steps.",
            self.algorithm.upper(),
            self.num_envs,
            f"{total_steps:,}",
        )
        progress = tqdm(
            total=total_updates,
            initial=self.current_update,
            disable=not self.config["train"].get("progress_bar", True),
            desc=self.algorithm.upper(),
            dynamic_ncols=True,
        )
        training_start = time.perf_counter()
        final_validation = None
        try:
            for update in range(self.current_update + 1, total_updates + 1):
                self.current_update = update
                self._decay_learning_rates(total_updates)
                rollout = self.collect_rollout()
                update_metrics = self.ppo_update(rollout)
                episode = self._episode_summary()
                curriculum_probabilities = None
                curriculum_config = self.config["curriculum"]
                if (
                    self.curriculum is not None
                    and update >= int(curriculum_config["warmup_updates"])
                    and update % int(curriculum_config["evaluation_interval_updates"]) == 0
                ):
                    curriculum_probabilities = self._curriculum_update()
                elapsed = max(time.perf_counter() - training_start, 1e-6)
                fps = self.environment_steps / elapsed
                if self.curriculum is not None:
                    entropy = self.curriculum.entropy()
                    maximum_probability = max(self.curriculum.probabilities.values())
                    most_sampled = max(self.curriculum.probabilities, key=self.curriculum.probabilities.get)
                else:
                    entropy = 0.0
                    maximum_probability = 1.0
                    most_sampled = "nominal"
                row = {
                    "environment_steps": self.environment_steps,
                    "update": update,
                    "mean_episodic_return": episode["mean_return"],
                    "success_rate": episode["success_rate"],
                    "mean_episode_length": episode["mean_length"],
                    **update_metrics,
                    "actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
                    "critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"],
                    "fps": fps,
                    "curriculum_entropy": entropy,
                    "maximum_curriculum_probability": maximum_probability,
                    "most_sampled_profile": most_sampled,
                }
                if update_metrics["approximate_kl"] > self.config["ppo"]["target_kl"] * 2.0:
                    logger.warning(
                        "Approximate KL %.4f materially exceeds target %.4f.",
                        update_metrics["approximate_kl"],
                        self.config["ppo"]["target_kl"],
                    )
                checkpoint_frequency = int(self.config["train"]["checkpoint_frequency_steps"])
                if self.environment_steps - self._last_checkpoint_step >= checkpoint_frequency:
                    self.save_checkpoint(
                        self.run_dir / "checkpoints" / f"checkpoint_step_{self.environment_steps}.pt"
                    )
                    self._last_checkpoint_step = self.environment_steps
                validation_frequency = int(self.config["train"]["validation_frequency_steps"])
                should_validate = (
                    self.environment_steps - self._last_validation_step >= validation_frequency
                    or update == total_updates
                )
                if should_validate:
                    final_validation = self._deterministic_evaluation(
                        self.config["train"]["validation_episodes"],
                        self.config["evaluation"]["seed_bases"]["validation"],
                        profile="nominal",
                    )
                    self._last_validation_step = self.environment_steps
                    mean_time = final_validation["mean_time_to_score"]
                    time_tie = -mean_time if mean_time is not None else -1e9
                    score = (
                        final_validation["success_rate"],
                        final_validation["mean_return"],
                        time_tie,
                    )
                    if self.best_validation_score is None or tuple(score) > tuple(self.best_validation_score):
                        self.best_validation_score = score
                        self.save_checkpoint(self.run_dir / "models" / "best_checkpoint.pt")
                row["validation_success_rate"] = (
                    final_validation["success_rate"] if final_validation is not None else ""
                )
                row["validation_mean_return"] = (
                    final_validation["mean_return"] if final_validation is not None else ""
                )
                metrics_writer.write(row)
                if update == 1 or update == total_updates or update % max(1, total_updates // 10) == 0:
                    logger.info(
                        "Update %d/%d | success %.2f | return %.2f | FPS %s",
                        update,
                        total_updates,
                        episode["success_rate"],
                        episode["mean_return"],
                        f"{fps:,.0f}",
                    )
                progress.set_postfix(
                    success=f"{episode['success_rate']:.2f}",
                    reward=f"{episode['mean_return']:.2f}",
                    kl=f"{update_metrics['approximate_kl']:.3f}",
                )
                progress.update(1)
                if curriculum_probabilities is not None:
                    logger.info("Updated failure-directed sampling over %d profiles.", len(curriculum_probabilities))
        finally:
            progress.close()
            metrics_writer.close()
        final_checkpoint = self.save_checkpoint(self.run_dir / "models" / "final_checkpoint.pt")
        final_actor = self.export_actor(self.run_dir / "models" / "final_actor.ts")
        best_checkpoint_path = self.run_dir / "models" / "best_checkpoint.pt"
        if not best_checkpoint_path.exists():
            self.save_checkpoint(best_checkpoint_path)
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        best_actor = self.export_actor(
            self.run_dir / "models" / "best_actor.ts", actor_state=best_checkpoint["actor_weights"]
        )
        frame = self.environments[0].render()
        from PIL import Image

        Image.fromarray(frame).save(self.run_dir / "videos" / "render_check.png")
        self._write_curriculum_history()
        plot_training_history(self.run_dir)
        plot_curriculum_history(self.run_dir)
        return {
            "final_checkpoint": str(final_checkpoint),
            "best_checkpoint": str(best_checkpoint_path),
            "final_actor": str(final_actor),
            "best_actor": str(best_actor),
            "metrics_csv": str(self.run_dir / "logs" / "metrics.csv"),
            "render_check": str(self.run_dir / "videos" / "render_check.png"),
            "final_validation": final_validation,
            "environment_steps": self.environment_steps,
            "updates": self.current_update,
        }

    def close(self):
        for env in self.environments:
            env.close()
        self.environments = []


def plot_training_history(run_dir):
    plt = get_pyplot()
    path = Path(run_dir) / "logs" / "metrics.csv"
    if not path.is_file():
        return
    data = pd.read_csv(path)
    plot_specs = [
        (["mean_episodic_return"], "Training episodic return", "Team return", "training_return.png"),
        (["success_rate"], "Observed training success", "Success rate", "training_success.png"),
        (
            ["validation_success_rate"],
            "Fixed-seed validation success",
            "Success rate",
            "validation_success.png",
        ),
        (["policy_loss", "value_loss"], "PPO optimization losses", "Loss", "training_losses.png"),
        (["approximate_kl", "clip_fraction"], "PPO trust-region diagnostics", "Value", "training_ppo_diagnostics.png"),
    ]
    for columns, title, ylabel, filename in plot_specs:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for column in columns:
            if column in data:
                axis.plot(data["environment_steps"], data[column], label=column.replace("_", " "))
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if len(columns) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(Path(run_dir) / "plots" / filename, dpi=180)
        plt.close(figure)


def plot_curriculum_history(run_dir):
    plt = get_pyplot()
    path = Path(run_dir) / "logs" / "curriculum_history.csv"
    if not path.is_file():
        return
    data = pd.read_csv(path)
    if data.empty:
        return
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for profile, group in data.groupby("profile"):
        axis.plot(group["update"], group["sampling_probability"], marker="o", label=profile)
    axis.set_title("Failure-directed perturbation sampling probabilities")
    axis.set_xlabel("PPO update")
    axis.set_ylabel("Sampling probability")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(Path(run_dir) / "plots" / "curriculum_probabilities.png", dpi=180)
    plt.close(figure)


def load_checkpoint_actor(config, checkpoint_path, device="cpu"):
    """Build an actor and observation normalizer from a complete training checkpoint."""
    probe = make_environment(config, "abstract")
    observation_size = probe.observation_dimension
    state_size = probe.state_dimension
    probe.close()
    actor, _ = build_networks(config, observation_size, state_size)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_weights"])
    actor.to(device).eval()
    normalizer = RunningMeanStd((observation_size,))
    normalizer.load_state_dict(checkpoint["observation_normalization"])
    return actor, normalizer, checkpoint


def run_training(
    config,
    source_config=None,
    parsed_args=None,
    run_name=None,
    resume_path=None,
    warm_start_path=None,
):
    """Create an artifact run, train safely, and update metadata on success or failure."""
    run_dir = create_run_directory(config, run_name=run_name)
    setup_logging(
        run_dir,
        config["logging"].get("console_level", "INFO"),
        config["logging"].get("file_level", "DEBUG"),
    )
    save_config(config, run_dir / "resolved_config.yaml")
    metadata = initial_metadata(config, run_dir, source_config, parsed_args)
    write_json(run_dir / "run_metadata.json", metadata)
    set_global_seeds(
        int(config["experiment"]["seed"]), config["experiment"].get("deterministic_torch", False)
    )
    trainer = None
    try:
        if config.get("phase3", {}).get("enabled", False):
            phase3 = config["phase3"]
            if phase3.get("require_calibration", False):
                summary_path = phase3.get("calibration_summary")
                if not summary_path:
                    raise ValueError(
                        "Phase 3 training is gated: pass --calibration-summary from a "
                        "successful non-smoke calibration run"
                    )
                calibration = json.loads(Path(summary_path).read_text(encoding="utf-8"))
                if not calibration.get("training_authorized", False):
                    raise ValueError(
                        "Phase 3 calibration did not authorize training; thresholds are not relaxed"
                    )
            from robosoccer.recurrent import RecurrentMAPPOTrainer

            trainer = RecurrentMAPPOTrainer(
                config,
                run_dir,
                resume_path=resume_path,
                warm_start_path=warm_start_path,
            )
        else:
            if warm_start_path is not None:
                raise ValueError("--warm-start is supported only for Phase 3 recurrent training")
            trainer = PPOTrainer(config, run_dir, resume_path=resume_path)
        artifacts = trainer.train()
        metadata["status"] = "complete"
        metadata["utc_completion"] = utc_now()
        metadata["training_step_counts"] = {
            "environment_steps": artifacts["environment_steps"],
            "updates": artifacts["updates"],
        }
        metadata["output_artifact_paths"] = artifacts
        write_json(run_dir / "run_metadata.json", metadata)
        finalize_run(config, run_dir, metadata)
        logger.info("Completed run: %s", run_dir)
        return run_dir, metadata
    except Exception as exc:
        if trainer is not None:
            try:
                trainer.save_checkpoint(run_dir / "checkpoints" / "emergency_checkpoint.pt")
            except Exception:
                logger.exception("Could not save emergency checkpoint")
        metadata["status"] = "failed"
        metadata["utc_completion"] = utc_now()
        metadata["failure_exception"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_json(run_dir / "run_metadata.json", metadata)
        logger.exception("Training failed: %s", exc)
        raise
    finally:
        if trainer is not None:
            trainer.close()
