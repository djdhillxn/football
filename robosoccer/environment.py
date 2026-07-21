"""Abstract and rigid-body two-attacker robot-soccer environments.

The environments share observations, rewards, rendering, and behavior semantics. Their
transition implementations are separate: the abstract simulator integrates explicit
kinematics, while the transfer simulator controls Pymunk rigid bodies with forces.
"""

import copy
import logging
import math
from collections import deque

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from PIL import Image, ImageDraw

from robosoccer.utils import check_finite

try:
    import pymunk
except ImportError:
    pymunk = None

logger = logging.getLogger(__name__)

ACTION_NAMES = [
    "approach_ball",
    "dribble_left",
    "dribble_right",
    "shoot",
    "pass_to_teammate",
    "move_to_support",
    "hold_and_face_ball",
]
AGENTS = ["attacker_0", "attacker_1"]
DEFENDER_MODES = ["stationary_goalie", "chase_ball", "intercept"]
PERTURBATION_KEYS = [
    "action_latency",
    "observation_latency",
    "communication_latency",
    "packet_loss",
    "speed_multiplier",
    "angular_speed_multiplier",
    "acceleration_multiplier",
    "kick_strength_multiplier",
    "kick_direction_noise",
    "ball_drag_multiplier",
    "ball_restitution",
    "ball_mass_multiplier",
    "localization_noise",
    "heading_noise",
    "ball_observation_noise",
    "teammate_position_noise",
    "defender_position_noise",
    "missed_ball_probability",
    "missed_teammate_probability",
    "missed_defender_probability",
    "defender_reaction_delay",
    "defender_speed_multiplier",
]


def angle_wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def unit_vector(vector):
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / length


def clip_length(vector, maximum):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= maximum or length < 1e-8:
        return vector
    return vector * maximum / length


def one_hot(index, size):
    result = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= int(index) < size:
        result[int(index)] = 1.0
    return result


def world_to_ego(vector, heading):
    cosine = math.cos(heading)
    sine = math.sin(heading)
    x_value, y_value = vector
    return np.array(
        [cosine * x_value + sine * y_value, -sine * x_value + cosine * y_value],
        dtype=np.float64,
    )


def distance_to_segment(point, start, end):
    segment = np.asarray(end) - np.asarray(start)
    denominator = float(np.dot(segment, segment))
    if denominator < 1e-10:
        return float(np.linalg.norm(np.asarray(point) - np.asarray(start)))
    fraction = float(np.dot(np.asarray(point) - np.asarray(start), segment) / denominator)
    projection = np.asarray(start) + np.clip(fraction, 0.0, 1.0) * segment
    return float(np.linalg.norm(np.asarray(point) - projection))


def available_profile_names(config, include_nominal=True):
    disabled = set(config["randomization"].get("disabled_families", []))
    configured_names = config["randomization"].get("training_profiles")
    names = []
    for name, profile in config["randomization"]["profiles"].items():
        if configured_names is not None and name not in configured_names and name != "nominal":
            continue
        if profile.get("family", "") in disabled:
            continue
        if name == "nominal" and not include_nominal:
            continue
        names.append(name)
    return names


def sample_profile_parameters(profile, rng):
    """Sample interpretable profile ranges, preserving integer latency values."""
    sampled = {}
    for key, bounds in profile.get("parameters", {}).items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            low, high = bounds
            if isinstance(low, int) and isinstance(high, int):
                sampled[key] = int(rng.integers(low, high + 1))
            else:
                sampled[key] = float(rng.uniform(float(low), float(high)))
        else:
            sampled[key] = copy.deepcopy(bounds)
    return sampled


class SoccerEnvBase(ParallelEnv):
    """Shared multi-agent interface, observations, rewards, and episode bookkeeping."""

    metadata = {"render_modes": ["rgb_array"], "name": "robosoccer_v0", "render_fps": 15}

    def __init__(self, config, render_mode=None, profile_name=None, profile_probabilities=None):
        self.config = copy.deepcopy(config)
        self.env_config = self.config["environment"]
        self.transfer_config = self.config["transfer_environment"]
        self.reward_config = self.config["reward"]
        self.opponent_config = self.config["opponent"]
        self.observation_config = self.config["observations"]
        self.randomization_config = self.config["randomization"]
        self.render_config = self.config["video"]
        self.render_mode = render_mode
        self.possible_agents = AGENTS[:]
        self.agents = []
        self.observation_dimension = 61
        self.base_state_dimension = 66
        self.perturbation_state_dimension = len(PERTURBATION_KEYS)
        self.state_dimension = self.base_state_dimension
        if self.observation_config.get("expose_perturbations_to_critic", False):
            self.state_dimension += self.perturbation_state_dimension
        self._observation_spaces = {
            agent: spaces.Box(-10.0, 10.0, (self.observation_dimension,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {agent: spaces.Discrete(len(ACTION_NAMES)) for agent in self.possible_agents}
        self.state_space = spaces.Box(-10.0, 10.0, (self.state_dimension,), dtype=np.float32)
        self.forced_profile = profile_name
        self.profile_probabilities = copy.deepcopy(profile_probabilities)
        self.rng = np.random.default_rng()
        self.seed_value = None
        self.players = {}
        self.ball = {}
        self.defender = {}
        self.sampled_parameters = {}
        self.selected_profile = "nominal"
        self.step_count = 0
        self.stationary_steps = 0
        self.action_queues = {}
        self.observation_queues = {}
        self.ball_histories = {}
        self.communication_queues = {}
        self.delivered_messages = {}
        self._last_observations = {}
        self._termination_reason = None
        self._kick_used = {}
        self._events = {}
        self._pass_pending = None
        self._defender_ball_history = deque(maxlen=16)
        self.metrics = {}
        self._last_render_info = {}

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def set_profile_probabilities(self, probabilities):
        self.profile_probabilities = copy.deepcopy(probabilities)

    def _default_parameters(self):
        return {
            "action_latency": 0,
            "observation_latency": 0,
            "communication_latency": 0,
            "packet_loss": 0.0,
            "speed_multiplier": 1.0,
            "angular_speed_multiplier": 1.0,
            "acceleration_multiplier": 1.0,
            "kick_strength_multiplier": 1.0,
            "kick_direction_noise": 0.0,
            "ball_drag_multiplier": 1.0,
            "ball_restitution": self.env_config["ball_restitution"],
            "ball_mass_multiplier": 1.0,
            "localization_noise": 0.0,
            "heading_noise": 0.0,
            "ball_observation_noise": 0.0,
            "teammate_position_noise": 0.0,
            "defender_position_noise": 0.0,
            "missed_ball_probability": 0.0,
            "missed_teammate_probability": 0.0,
            "missed_defender_probability": 0.0,
            "defender_reaction_delay": self.opponent_config["reaction_delay_steps"],
            "defender_speed_multiplier": 1.0,
        }

    def _select_profile(self, options):
        options = options or {}
        forced = options.get("profile", self.forced_profile)
        mode = self.randomization_config.get("mode", "none")
        profiles = self.randomization_config["profiles"]
        enabled = available_profile_names(self.config, include_nominal=True)
        if forced is not None:
            if forced not in profiles:
                raise ValueError("Unknown perturbation profile: " + str(forced))
            selected = forced
        elif mode == "none":
            selected = "nominal"
        elif mode == "uniform":
            if self.rng.random() < float(self.randomization_config.get("nominal_probability", 0.0)):
                selected = "nominal"
            else:
                non_nominal = [name for name in enabled if name != "nominal"]
                selected = str(self.rng.choice(non_nominal or ["nominal"]))
        else:
            nominal_probability = float(self.randomization_config.get("nominal_probability", 0.0))
            probabilities = self.profile_probabilities or {}
            candidates = [name for name in enabled if name != "nominal"]
            if self.rng.random() < nominal_probability:
                selected = "nominal"
            elif not candidates:
                selected = "nominal"
            elif probabilities:
                weights = np.array([max(0.0, probabilities.get(name, 0.0)) for name in candidates])
                if weights.sum() <= 0:
                    weights = np.ones(len(candidates), dtype=np.float64)
                weights /= weights.sum()
                selected = str(self.rng.choice(candidates, p=weights))
            else:
                selected = str(self.rng.choice(candidates))
        parameters = self._default_parameters()
        parameters.update(sample_profile_parameters(profiles[selected], self.rng))
        parameters.update(copy.deepcopy(options.get("sampled_parameters", {})))
        maximum_delay = int(self.env_config.get("maximum_delay_steps", 8))
        for key in ["action_latency", "observation_latency", "communication_latency"]:
            parameters[key] = int(np.clip(parameters[key], 0, maximum_delay))
        parameters["packet_loss"] = float(np.clip(parameters["packet_loss"], 0.0, 1.0))
        return selected, parameters

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        elif self.seed_value is None:
            self.seed_value = int(np.random.SeedSequence().entropy) % (2**31)
            self.rng = np.random.default_rng(self.seed_value)
        self.agents = self.possible_agents[:]
        self.selected_profile, self.sampled_parameters = self._select_profile(options)
        self.step_count = 0
        self.stationary_steps = 0
        self._termination_reason = None
        self._pass_pending = None
        self._defender_ball_history.clear()
        self._events = {}

        attacker_positions, ball_position, defender_position = self._sample_initial_positions()
        self.players = {}
        for index, agent in enumerate(self.possible_agents):
            heading = float(self.rng.uniform(-0.25, 0.25))
            self.players[agent] = {
                "position": attacker_positions[index],
                "heading": heading,
                "velocity": np.zeros(2, dtype=np.float64),
                "angular_velocity": 0.0,
                "radius": self.env_config["player_radius"],
                "previous_action": 6,
                "current_action": 6,
                "action_repeat_remaining": 0,
                "possesses_ball": False,
                "distance_travelled": 0.0,
            }
        self.ball = {
            "position": ball_position,
            "velocity": np.zeros(2, dtype=np.float64),
            "radius": self.env_config["ball_radius"],
            "last_touch": None,
            "possession": None,
        }
        self.defender = {
            "position": defender_position,
            "heading": math.pi,
            "velocity": np.zeros(2, dtype=np.float64),
            "angular_velocity": 0.0,
            "mode": (options or {}).get("defender_mode", self.opponent_config["mode"]),
        }
        if self.defender["mode"] not in DEFENDER_MODES:
            raise ValueError("Unknown defender mode: " + str(self.defender["mode"]))
        self._defender_ball_history.append(self.ball["position"].copy())
        self._reset_queues()
        self._reset_physics()
        self._reset_metrics()
        self._update_possession()
        initial_snapshot = self._snapshot()
        latency = self.sampled_parameters["observation_latency"]
        self.observation_queues = {
            agent: deque([copy.deepcopy(initial_snapshot) for _ in range(latency + 1)], maxlen=latency + 1)
            for agent in self.possible_agents
        }
        self.ball_histories = {
            agent: deque(
                [self.ball["position"].copy() for _ in range(self.observation_config["history_length"])],
                maxlen=self.observation_config["history_length"],
            )
            for agent in self.possible_agents
        }
        self._last_observations = self._create_observations(advance_queue=False)
        infos = {
            agent: {
                "selected_profile": self.selected_profile,
                "sampled_parameters": copy.deepcopy(self.sampled_parameters),
                "seed": self.seed_value,
            }
            for agent in self.possible_agents
        }
        return copy.deepcopy(self._last_observations), infos

    def _sample_initial_positions(self):
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        minimum_distance = 2.2 * self.env_config["player_radius"]
        for _ in range(200):
            attackers = [
                np.array(
                    [self.rng.uniform(-0.38, -0.18) * half_length, self.rng.uniform(-0.55, 0.55) * half_width]
                ),
                np.array(
                    [self.rng.uniform(-0.40, -0.15) * half_length, self.rng.uniform(-0.55, 0.55) * half_width]
                ),
            ]
            ball = np.mean(attackers, axis=0) + self.rng.uniform([-0.15, -0.35], [0.45, 0.35])
            defender = np.array(
                [self.rng.uniform(0.45, 0.72) * half_length, self.rng.uniform(-0.45, 0.45) * half_width]
            )
            objects = [*attackers, ball, defender]
            pair_distances = [
                np.linalg.norm(objects[i] - objects[j])
                for i in range(len(objects))
                for j in range(i + 1, len(objects))
            ]
            if min(pair_distances) > minimum_distance and abs(ball[1]) < half_width - 0.2:
                return attackers, ball, defender
        raise RuntimeError("Could not sample a non-overlapping initial soccer state")

    def _reset_queues(self):
        action_latency = self.sampled_parameters["action_latency"]
        self.action_queues = {agent: [6] * action_latency for agent in self.possible_agents}
        communication_latency = self.sampled_parameters["communication_latency"]
        self.communication_queues = {}
        self.delivered_messages = {}
        for index, receiver in enumerate(self.possible_agents):
            sender = self.possible_agents[1 - index]
            message = {"action": 6, "possesses_ball": False, "sender": sender}
            self.communication_queues[receiver] = [copy.deepcopy(message)] * communication_latency
            self.delivered_messages[receiver] = {**message, "age": 0}

    def _reset_metrics(self):
        self.metrics = {
            "success": False,
            "termination_reason": None,
            "time_to_score": None,
            "team_return": 0.0,
            "pass_attempts": 0,
            "completed_passes": 0,
            "intercepted_passes": 0,
            "possession_steps": 0,
            "possession_changes": 0,
            "attacker_collisions": 0,
            "defender_contacts": 0,
            "out_of_bounds_events": 0,
            "invalid_kick_attempts": 0,
            "invalid_pass_attempts": 0,
            "action_switches": 0,
            "redundant_ball_chasing_steps": 0,
            "distance_attacker_0": 0.0,
            "distance_attacker_1": 0.0,
            "separation_sum": 0.0,
            "support_quality_sum": 0.0,
            "episode_steps": 0,
            "selected_profile": self.selected_profile,
            "sampled_parameters": copy.deepcopy(self.sampled_parameters),
        }

    def _reset_physics(self):
        pass

    def _snapshot(self):
        return {
            "players": {
                agent: {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in player.items()
                }
                for agent, player in self.players.items()
            },
            "ball": {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.ball.items()
            },
            "defender": {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.defender.items()
            },
            "step_count": self.step_count,
        }

    def observe(self, agent):
        return copy.deepcopy(self._last_observations[agent])

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("step() called after the episode ended; call reset()")
        acting_agents = self.agents[:]
        if set(actions) != set(acting_agents):
            raise ValueError("Actions must be supplied for exactly the current agents")
        requested = {}
        applied = {}
        for agent in acting_agents:
            action = int(actions[agent])
            if not self.action_space(agent).contains(action):
                raise ValueError("Invalid action for " + agent + ": " + str(action))
            requested[agent] = action
            queue = self.action_queues[agent]
            queue.append(action)
            applied[agent] = int(queue.pop(0))

        previous_potential = self._potential_components()
        previous_positions = {agent: self.players[agent]["position"].copy() for agent in acting_agents}
        previous_collision_count = self.metrics["attacker_collisions"]
        for agent in acting_agents:
            player = self.players[agent]
            if player["current_action"] != applied[agent]:
                self.metrics["action_switches"] += 1
            player["previous_action"] = player["current_action"]
            player["current_action"] = applied[agent]
            player["action_repeat_remaining"] = self.env_config["macro_action_repeat"]
        self._kick_used = {agent: False for agent in acting_agents}
        self._events = {
            "invalid_action": 0,
            "pass_reward": 0.0,
            "defender_clear": False,
            "out_of_bounds": False,
        }
        repeat = int(self.env_config["macro_action_repeat"])
        for substep in range(repeat):
            for agent in acting_agents:
                self.players[agent]["action_repeat_remaining"] = repeat - substep
            self._physics_substep(applied, float(self.env_config["dt"]))
            self._update_possession()
            if self._check_immediate_terminal() is not None:
                break
        for agent in acting_agents:
            self.players[agent]["action_repeat_remaining"] = 0

        self.step_count += 1
        self._update_communication()
        self._update_metrics(applied, previous_positions)
        next_potential = self._potential_components()
        termination_reason = self._check_terminal()
        terminated = termination_reason in {"goal", "defender_clear", "out_of_bounds", "invalid_state"}
        truncated = termination_reason in {"timeout", "stationary_ball"}
        reward_components = self._reward_components(
            previous_potential,
            next_potential,
            termination_reason,
            self.metrics["attacker_collisions"] - previous_collision_count,
        )
        shared_reward = float(sum(reward_components.values()))
        check_finite("team reward", shared_reward)
        self.metrics["team_return"] += shared_reward
        self.metrics["episode_steps"] = self.step_count
        self._last_observations = self._create_observations(advance_queue=True)
        done = terminated or truncated
        if done:
            self._termination_reason = termination_reason
            self.metrics["termination_reason"] = termination_reason
            self.metrics["success"] = termination_reason == "goal"
            if termination_reason == "goal":
                self.metrics["time_to_score"] = (
                    self.step_count * self.env_config["dt"] * self.env_config["macro_action_repeat"]
                )
        rewards = {agent: shared_reward for agent in acting_agents}
        terminations = {agent: terminated for agent in acting_agents}
        truncations = {agent: truncated for agent in acting_agents}
        infos = {}
        for agent in acting_agents:
            infos[agent] = {
                "termination_reason": termination_reason,
                "reward_components": copy.deepcopy(reward_components),
                "episode_metrics": copy.deepcopy(self.metrics) if done else {},
                "selected_profile": self.selected_profile,
                "sampled_parameters": copy.deepcopy(self.sampled_parameters),
                "requested_action": requested[agent],
                "applied_action": applied[agent],
            }
        # ParallelEnv emits the final observation for agents that terminate on this call.
        # This also gives time-limit bootstrapping code an explicit terminal transition.
        observations = copy.deepcopy(self._last_observations)
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _reward_components(self, previous, current, reason, new_collisions):
        gamma = float(self.config["ppo"]["gamma"])
        reward = {
            "sparse_goal": self.reward_config["goal"] if reason == "goal" else 0.0,
            "progress_shaping": self.reward_config["progress_weight"]
            * (gamma * current["progress"] - previous["progress"]),
            "possession_shaping": self.reward_config["possession_weight"]
            * (gamma * current["possession"] - previous["possession"]),
            "support_shaping": self.reward_config["support_weight"]
            * (gamma * current["support"] - previous["support"]),
            "pass_reward": self._events["pass_reward"],
            "penalties": self.reward_config["time_penalty"],
        }
        reward["penalties"] += self.reward_config["invalid_action"] * self._events["invalid_action"]
        reward["penalties"] += self.reward_config["collision"] * new_collisions
        if reason == "defender_clear":
            reward["penalties"] += self.reward_config["defender_clear"]
        elif reason == "out_of_bounds":
            reward["penalties"] += self.reward_config["out_of_bounds"]
        elif reason in {"timeout", "stationary_ball"}:
            reward["penalties"] += self.reward_config["timeout"]
        return reward

    def _physics_substep(self, actions, dt):
        raise NotImplementedError

    def _desired_motion(self, agent, action, dt):
        player = self.players[agent]
        ball_delta = self.ball["position"] - player["position"]
        distance = float(np.linalg.norm(ball_delta))
        goal = np.array([self.env_config["field_length"] / 2.0, 0.0])
        target = player["position"].copy()
        speed_fraction = 1.0
        if action == 0:
            target = self.ball["position"] - np.array([0.32, 0.0])
            if distance < 0.65:
                speed_fraction = max(0.25, distance / 0.65)
        elif action in {1, 2}:
            side = 1.0 if action == 1 else -1.0
            lane = np.array([goal[0], side * self.env_config["goal_width"] * 0.32])
            target = self.ball["position"] + 0.6 * unit_vector(lane - self.ball["position"])
            if distance <= self.env_config["kick_radius"] * 1.35:
                impulse = unit_vector(lane - self.ball["position"]) * self.env_config["dribble_impulse"]
                self._deliver_dribble_impulse(impulse * dt / self.env_config["dt"])
                self.ball["last_touch"] = agent
            else:
                self._mark_invalid(action)
        elif action == 3:
            lane_y = float(np.clip(-0.55 * self.defender["position"][1], -0.55, 0.55))
            kick_target = np.array([goal[0] + 0.2, lane_y])
            self._attempt_kick(agent, action, kick_target, self.env_config["shoot_speed"])
            target = self.ball["position"] - np.array([0.15, 0.0])
            speed_fraction = 0.25
        elif action == 4:
            teammate_name = AGENTS[1 - AGENTS.index(agent)]
            teammate = self.players[teammate_name]
            prediction = teammate["position"] + 0.35 * teammate["velocity"]
            self._attempt_kick(agent, action, prediction, self.env_config["pass_speed"])
            target = self.ball["position"] - np.array([0.12, 0.0])
            speed_fraction = 0.25
        elif action == 5:
            target = self.compute_support_target(agent)
        else:
            target = player["position"]
            speed_fraction = 0.0
        facing_target = self.ball["position"] if action in {5, 6} else target
        if np.linalg.norm(facing_target - player["position"]) < 1e-8:
            desired_heading = player["heading"]
        else:
            desired_heading = math.atan2(
                facing_target[1] - player["position"][1], facing_target[0] - player["position"][0]
            )
        maximum_speed = (
            self.env_config["player_max_speed"] * self.sampled_parameters["speed_multiplier"]
        )
        desired_velocity = unit_vector(target - player["position"]) * maximum_speed * speed_fraction
        if action == 6:
            desired_velocity *= 0.1
        return desired_velocity, desired_heading

    def _attempt_kick(self, agent, action, target, nominal_speed):
        if self._kick_used[agent]:
            return
        self._kick_used[agent] = True
        player = self.players[agent]
        delta = self.ball["position"] - player["position"]
        distance = float(np.linalg.norm(delta))
        target_angle = math.atan2(target[1] - self.ball["position"][1], target[0] - self.ball["position"][0])
        aligned = abs(angle_wrap(target_angle - player["heading"])) <= self.env_config["kick_angle_tolerance"]
        if distance > self.env_config["kick_radius"] or not aligned:
            self._mark_invalid(action)
            return
        noisy_angle = target_angle + float(
            self.rng.normal(0.0, self.sampled_parameters["kick_direction_noise"])
        )
        direction = np.array([math.cos(noisy_angle), math.sin(noisy_angle)])
        speed = nominal_speed * self.sampled_parameters["kick_strength_multiplier"]
        self._deliver_kick(direction, speed)
        self.ball["last_touch"] = agent
        self.ball["possession"] = None
        if action == 4:
            teammate = AGENTS[1 - AGENTS.index(agent)]
            self.metrics["pass_attempts"] += 1
            self._pass_pending = {"from": agent, "to": teammate, "age": 0}

    def _deliver_kick(self, direction, speed):
        self.ball["velocity"] = direction * speed

    def _deliver_dribble_impulse(self, impulse):
        self.ball["velocity"] = clip_length(
            self.ball["velocity"] + impulse,
            self.env_config["pass_speed"] * self.sampled_parameters["kick_strength_multiplier"],
        )

    def _mark_invalid(self, action):
        self._events["invalid_action"] += 1
        if action == 3:
            self.metrics["invalid_kick_attempts"] += 1
        elif action == 4:
            self.metrics["invalid_pass_attempts"] += 1

    def _defender_target(self):
        mode = self.defender["mode"]
        half_length = self.env_config["field_length"] / 2.0
        if mode == "stationary_goalie":
            return np.array([half_length - 0.35, np.clip(self.ball["position"][1], -0.55, 0.55)])
        if mode == "chase_ball":
            return self.ball["position"].copy()
        delay = int(self.sampled_parameters["defender_reaction_delay"])
        history = list(self._defender_ball_history)
        perceived = history[max(0, len(history) - 1 - delay)] if history else self.ball["position"]
        if perceived[0] < 0.5:
            return np.array([half_length - 0.75, np.clip(perceived[1] * 0.45, -0.65, 0.65)])
        predicted = perceived + 0.7 * self.ball["velocity"]
        predicted[0] = np.clip(predicted[0], 0.5, half_length - 0.25)
        predicted[1] = np.clip(predicted[1], -1.3, 1.3)
        return predicted

    def _defender_desired_motion(self):
        target = self._defender_target()
        delta = target - self.defender["position"]
        maximum = self.opponent_config["max_speed"] * self.sampled_parameters["defender_speed_multiplier"]
        velocity = unit_vector(delta) * maximum
        heading = math.atan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-8 else self.defender["heading"]
        return velocity, heading

    def _defender_clear_if_possible(self):
        delta = self.ball["position"] - self.defender["position"]
        if np.linalg.norm(delta) <= self.opponent_config["control_radius"]:
            direction = unit_vector(np.array([-1.0, 0.15 * np.sign(self.ball["position"][1] + 1e-6)]))
            self._defender_kick(direction, self.opponent_config["clear_speed"])
            if self.ball["last_touch"] != "defender":
                self.metrics["defender_contacts"] += 1
            self.ball["last_touch"] = "defender"
            self.ball["possession"] = "defender"

    def _defender_kick(self, direction, speed):
        self.ball["velocity"] = direction * speed

    def compute_support_target(self, agent):
        teammate = AGENTS[1 - AGENTS.index(agent)]
        ball = self.ball["position"]
        teammate_position = self.players[teammate]["position"]
        defender = self.defender["position"]
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        separation = float(self.env_config.get("support_separation", 1.4))
        preferred_side = -1.0 if teammate_position[1] >= ball[1] else 1.0
        lateral = ball[1] + preferred_side * separation
        forward = ball[0] + 0.7 + 0.25 * max(0.0, defender[0] - ball[0])
        target = np.array([forward, lateral], dtype=np.float64)
        # Shift away from a blocked passing segment instead of using one fixed support point.
        if distance_to_segment(defender, ball, target) < 0.55:
            target[1] += preferred_side * 0.75
        target[0] = np.clip(target[0], -half_length + 0.35, half_length - 0.65)
        target[1] = np.clip(target[1], -half_width + 0.35, half_width - 0.35)
        return target

    def _update_possession(self):
        previous = self.ball.get("possession")
        distances = {
            agent: float(np.linalg.norm(self.ball["position"] - self.players[agent]["position"]))
            for agent in self.possible_agents
        }
        closest = min(distances, key=distances.get)
        if distances[closest] <= self.env_config["kick_radius"]:
            possession = closest
        elif np.linalg.norm(self.ball["position"] - self.defender["position"]) <= self.opponent_config["control_radius"]:
            possession = "defender"
        else:
            possession = None
        self.ball["possession"] = possession
        for agent in self.possible_agents:
            self.players[agent]["possesses_ball"] = possession == agent
        if possession != previous and previous is not None:
            self.metrics["possession_changes"] += 1
        if possession in self.possible_agents:
            self.metrics["possession_steps"] += 1
            self.ball["last_touch"] = possession
        if self._pass_pending is not None:
            self._pass_pending["age"] += 1
            if possession == self._pass_pending["to"]:
                self.metrics["completed_passes"] += 1
                self._events["pass_reward"] += self.reward_config["successful_pass"]
                self._pass_pending = None
            elif possession == "defender" or self._pass_pending["age"] > 18:
                self.metrics["intercepted_passes"] += 1
                self._events["pass_reward"] += self.reward_config["lost_pass"]
                self._pass_pending = None

    def _check_immediate_terminal(self):
        if not self._finite_state():
            return "invalid_state"
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        ball = self.ball["position"]
        if ball[0] >= half_length and abs(ball[1]) <= self.env_config["goal_width"] / 2.0:
            return "goal"
        if ball[0] <= self.env_config.get("failure_boundary", -half_length - 0.2):
            return "defender_clear"
        if abs(ball[1]) > half_width + 0.15 or ball[0] > half_length + 0.25:
            return "out_of_bounds"
        return None

    def _check_terminal(self):
        immediate = self._check_immediate_terminal()
        if immediate is not None:
            if immediate == "out_of_bounds":
                self.metrics["out_of_bounds_events"] += 1
            return immediate
        speed = float(np.linalg.norm(self.ball["velocity"]))
        if speed < 0.025 and self.ball["possession"] is None:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0
        if self.step_count >= self.env_config["max_episode_steps"]:
            return "timeout"
        if self.stationary_steps >= self.env_config.get("stationary_truncation_steps", 45):
            return "stationary_ball"
        return None

    def _finite_state(self):
        arrays = [self.ball["position"], self.ball["velocity"], self.defender["position"]]
        arrays += [self.players[agent]["position"] for agent in self.possible_agents]
        arrays += [self.players[agent]["velocity"] for agent in self.possible_agents]
        return all(np.all(np.isfinite(array)) for array in arrays)

    def _update_communication(self):
        latency = self.sampled_parameters["communication_latency"]
        loss = self.sampled_parameters["packet_loss"]
        for receiver_index, receiver in enumerate(self.possible_agents):
            sender = self.possible_agents[1 - receiver_index]
            message = {
                "action": self.players[sender]["current_action"],
                "possesses_ball": self.players[sender]["possesses_ball"],
                "sender": sender,
            }
            queue = self.communication_queues[receiver]
            queue.append(None if self.rng.random() < loss else message)
            delivered = queue.pop(0) if len(queue) > latency else None
            self.delivered_messages[receiver]["age"] += 1
            if delivered is not None:
                self.delivered_messages[receiver] = {**delivered, "age": 0}

    def _support_quality(self):
        possession = self.ball["possession"]
        if possession not in self.possible_agents:
            distances = {
                agent: np.linalg.norm(self.players[agent]["position"] - self.ball["position"])
                for agent in self.possible_agents
            }
            possession = min(distances, key=distances.get)
        supporter = AGENTS[1 - AGENTS.index(possession)]
        carrier_position = self.players[possession]["position"]
        support_position = self.players[supporter]["position"]
        separation = float(np.linalg.norm(support_position - carrier_position))
        separation_score = math.exp(-((separation - 1.5) ** 2) / 1.4)
        progress = np.clip(
            (support_position[0] + self.env_config["field_length"] / 2.0)
            / self.env_config["field_length"],
            0.0,
            1.0,
        )
        lane_distance = distance_to_segment(self.defender["position"], carrier_position, support_position)
        lane_score = np.clip(lane_distance / 0.9, 0.0, 1.0)
        defender_clearance = np.clip(
            np.linalg.norm(support_position - self.defender["position"]) / 1.2, 0.0, 1.0
        )
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        inside = float(abs(support_position[0]) < half_length and abs(support_position[1]) < half_width)
        return float(
            np.clip(
                0.35 * separation_score
                + 0.20 * progress
                + 0.25 * lane_score
                + 0.15 * defender_clearance
                + 0.05 * inside,
                0.0,
                1.0,
            )
        )

    def _potential_components(self):
        progress = np.clip(
            (self.ball["position"][0] + self.env_config["field_length"] / 2.0)
            / self.env_config["field_length"],
            0.0,
            1.0,
        )
        possession = float(self.ball["possession"] in self.possible_agents)
        return {"progress": float(progress), "possession": possession, "support": self._support_quality()}

    def _update_metrics(self, actions, previous_positions):
        for agent in self.possible_agents:
            distance = float(np.linalg.norm(self.players[agent]["position"] - previous_positions[agent]))
            self.players[agent]["distance_travelled"] += distance
            self.metrics["distance_" + agent] += distance
        separation = float(
            np.linalg.norm(self.players[AGENTS[0]]["position"] - self.players[AGENTS[1]]["position"])
        )
        self.metrics["separation_sum"] += separation
        self.metrics["support_quality_sum"] += self._support_quality()
        ball_actions = {0, 1, 2, 3, 4}
        near = all(
            np.linalg.norm(self.players[agent]["position"] - self.ball["position"])
            < self.env_config.get("redundant_chase_distance", 1.5)
            for agent in self.possible_agents
        )
        possession_conflict = self.ball["possession"] in self.possible_agents and all(
            actions[agent] in ball_actions for agent in self.possible_agents
        )
        if all(actions[agent] in ball_actions for agent in self.possible_agents) and (near or possession_conflict):
            self.metrics["redundant_ball_chasing_steps"] += 1

    def _create_observations(self, advance_queue):
        snapshot = self._snapshot()
        observations = {}
        for agent in self.possible_agents:
            queue = self.observation_queues[agent]
            if advance_queue:
                queue.append(copy.deepcopy(snapshot))
            perceived = queue[0]
            observations[agent] = self._build_local_observation(agent, perceived)
            check_finite("local observation", observations[agent])
        return observations

    def _build_local_observation(self, agent, snapshot):
        player = snapshot["players"][agent]
        teammate_name = AGENTS[1 - AGENTS.index(agent)]
        teammate = snapshot["players"][teammate_name]
        ball = snapshot["ball"]
        defender = snapshot["defender"]
        heading = player["heading"] + self.rng.normal(0.0, self.sampled_parameters["heading_noise"])
        localization = player["position"] + self.rng.normal(
            0.0, self.sampled_parameters["localization_noise"], 2
        )
        max_speed = max(self.env_config["player_max_speed"], 1e-6)
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        diagonal = math.hypot(self.env_config["field_length"], self.env_config["field_width"])
        values = [math.sin(heading), math.cos(heading)]
        values.extend((world_to_ego(player["velocity"], heading) / max_speed).tolist())
        values.append(player["angular_velocity"] / max(self.env_config["player_max_turn_rate"], 1e-6))
        values.extend(one_hot(player["previous_action"], len(ACTION_NAMES)).tolist())
        values.append(player["action_repeat_remaining"] / self.env_config["macro_action_repeat"])

        ball_visible = self.rng.random() >= self.sampled_parameters["missed_ball_probability"]
        noisy_ball = ball["position"] + self.rng.normal(
            0.0, self.sampled_parameters["ball_observation_noise"], 2
        )
        ball_relative = world_to_ego(noisy_ball - localization, heading)
        ball_velocity = world_to_ego(ball["velocity"] - player["velocity"], heading)
        ball_distance = float(np.linalg.norm(noisy_ball - localization))
        ball_bearing = math.atan2(ball_relative[1], ball_relative[0])
        if not ball_visible:
            ball_relative[:] = 0.0
            ball_velocity[:] = 0.0
            ball_distance = 0.0
            ball_bearing = 0.0
        values.extend((ball_relative / np.array([half_length, half_width])).tolist())
        values.extend((ball_velocity / max(self.env_config["shoot_speed"], 1e-6)).tolist())
        values.extend(
            [
                ball_distance / diagonal,
                math.sin(ball_bearing),
                math.cos(ball_bearing),
                float(ball_visible),
                float(ball_visible and ball_distance <= self.env_config["kick_radius"]),
            ]
        )
        history = self.ball_histories[agent]
        if ball_visible:
            history.append(noisy_ball.copy())
        else:
            history.append(history[-1].copy())
        for historical_position in history:
            relative = world_to_ego(historical_position - localization, heading)
            values.extend((relative / np.array([half_length, half_width])).tolist())

        for post_y in [-self.env_config["goal_width"] / 2.0, self.env_config["goal_width"] / 2.0]:
            relative = world_to_ego(np.array([half_length, post_y]) - localization, heading)
            values.extend((relative / np.array([half_length, half_width])).tolist())
        values.extend(
            [
                (localization[0] + half_length) / self.env_config["field_length"],
                (half_length - localization[0]) / self.env_config["field_length"],
                (localization[1] + half_width) / self.env_config["field_width"],
                (half_width - localization[1]) / self.env_config["field_width"],
            ]
        )
        goal_relative = world_to_ego(np.array([half_length, 0.0]) - localization, heading)
        goal_bearing = math.atan2(goal_relative[1], goal_relative[0])
        values.extend([math.sin(goal_bearing), math.cos(goal_bearing)])

        teammate_visible = self.rng.random() >= self.sampled_parameters["missed_teammate_probability"]
        teammate_position = teammate["position"] + self.rng.normal(
            0.0, self.sampled_parameters.get("teammate_position_noise", 0.0), 2
        )
        teammate_relative = world_to_ego(teammate_position - localization, heading)
        teammate_velocity = world_to_ego(teammate["velocity"] - player["velocity"], heading)
        relative_heading = angle_wrap(teammate["heading"] - heading)
        if not teammate_visible:
            teammate_relative[:] = 0.0
            teammate_velocity[:] = 0.0
            relative_heading = 0.0
        message = self.delivered_messages[agent]
        values.extend((teammate_relative / np.array([half_length, half_width])).tolist())
        values.extend((teammate_velocity / max_speed).tolist())
        values.extend([math.sin(relative_heading), math.cos(relative_heading), float(teammate_visible)])
        values.append(float(message["possesses_ball"]))
        values.extend(one_hot(message["action"], len(ACTION_NAMES)).tolist())
        values.append(min(message["age"], 20) / 20.0)

        defender_visible = self.rng.random() >= self.sampled_parameters["missed_defender_probability"]
        defender_position = defender["position"] + self.rng.normal(
            0.0, self.sampled_parameters.get("defender_position_noise", 0.0), 2
        )
        defender_relative = world_to_ego(defender_position - localization, heading)
        defender_velocity = world_to_ego(defender["velocity"] - player["velocity"], heading)
        if not defender_visible:
            defender_relative[:] = 0.0
            defender_velocity[:] = 0.0
        values.extend((defender_relative / np.array([half_length, half_width])).tolist())
        values.extend((defender_velocity / max_speed).tolist())
        values.append(float(defender_visible))
        values.extend(one_hot(AGENTS.index(agent), len(AGENTS)).tolist())
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.observation_dimension,):
            raise RuntimeError(
                f"Observation construction produced shape {result.shape}, expected {(self.observation_dimension,)}"
            )
        return np.clip(result, -10.0, 10.0)

    def state(self):
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        maximum_speed = max(self.env_config["player_max_speed"], 1e-6)
        values = []
        for agent in self.possible_agents:
            player = self.players[agent]
            values.extend((player["position"] / np.array([half_length, half_width])).tolist())
            values.extend([math.sin(player["heading"]), math.cos(player["heading"])])
            values.extend((player["velocity"] / maximum_speed).tolist())
            values.append(player["angular_velocity"] / self.env_config["player_max_turn_rate"])
            values.extend(one_hot(player["previous_action"], len(ACTION_NAMES)).tolist())
            values.extend(one_hot(player["current_action"], len(ACTION_NAMES)).tolist())
            values.append(player["action_repeat_remaining"] / self.env_config["macro_action_repeat"])
        values.extend((self.ball["position"] / np.array([half_length, half_width])).tolist())
        values.extend((self.ball["velocity"] / self.env_config["shoot_speed"]).tolist())
        possession_index = {None: 0, AGENTS[0]: 1, AGENTS[1]: 2, "defender": 3}.get(
            self.ball["possession"], 0
        )
        values.extend(one_hot(possession_index, 4).tolist())
        values.extend((self.defender["position"] / np.array([half_length, half_width])).tolist())
        values.extend([math.sin(self.defender["heading"]), math.cos(self.defender["heading"])])
        values.extend((self.defender["velocity"] / max(self.opponent_config["max_speed"], 1e-6)).tolist())
        values.extend(one_hot(DEFENDER_MODES.index(self.defender["mode"]), len(DEFENDER_MODES)).tolist())
        touch_index = {None: 0, AGENTS[0]: 1, AGENTS[1]: 2, "defender": 3}.get(
            self.ball["last_touch"], 0
        )
        values.extend(one_hot(touch_index, 4).tolist())
        values.append(self.step_count / self.env_config["max_episode_steps"])
        if self.observation_config.get("expose_perturbations_to_critic", False):
            for key in PERTURBATION_KEYS:
                values.append(float(self.sampled_parameters.get(key, 0.0)))
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.state_dimension,):
            raise RuntimeError(f"Global state shape {result.shape} does not match {(self.state_dimension,)}")
        check_finite("global state", result)
        return np.clip(result, -10.0, 10.0)

    def render(self):
        width = int(self.render_config.get("width", 1280))
        height = int(self.render_config.get("height", 720))
        image = Image.new("RGB", (width, height), (28, 34, 40))
        draw = ImageDraw.Draw(image)
        margin_x = int(width * 0.10)
        margin_y = int(height * 0.12)
        field_width_pixels = width - 2 * margin_x
        field_height_pixels = height - 2 * margin_y
        draw.rounded_rectangle(
            [margin_x, margin_y, margin_x + field_width_pixels, margin_y + field_height_pixels],
            radius=12,
            fill=(39, 128, 73),
            outline=(238, 238, 230),
            width=3,
        )
        draw.line(
            [width // 2, margin_y, width // 2, margin_y + field_height_pixels],
            fill=(225, 235, 225),
            width=2,
        )
        centre_radius = int(field_height_pixels * 0.13)
        draw.ellipse(
            [width // 2 - centre_radius, height // 2 - centre_radius, width // 2 + centre_radius, height // 2 + centre_radius],
            outline=(225, 235, 225),
            width=2,
        )

        def pixel(position):
            x_value = margin_x + (position[0] / self.env_config["field_length"] + 0.5) * field_width_pixels
            y_value = margin_y + (0.5 - position[1] / self.env_config["field_width"]) * field_height_pixels
            return int(x_value), int(y_value)

        goal_pixels = int(self.env_config["goal_width"] / self.env_config["field_width"] * field_height_pixels)
        draw.rectangle(
            [margin_x + field_width_pixels, height // 2 - goal_pixels // 2, margin_x + field_width_pixels + 18, height // 2 + goal_pixels // 2],
            outline=(220, 220, 220),
            width=3,
        )
        scale = field_height_pixels / self.env_config["field_width"]
        colours = {AGENTS[0]: (55, 125, 255), AGENTS[1]: (120, 190, 255)}
        for agent in self.possible_agents:
            player = self.players[agent]
            x_value, y_value = pixel(player["position"])
            radius = max(5, int(player["radius"] * scale))
            draw.ellipse(
                [x_value - radius, y_value - radius, x_value + radius, y_value + radius],
                fill=colours[agent],
                outline=(15, 35, 70),
                width=2,
            )
            heading_end = (
                x_value + int(radius * 1.4 * math.cos(player["heading"])),
                y_value - int(radius * 1.4 * math.sin(player["heading"])),
            )
            draw.line([x_value, y_value, *heading_end], fill=(245, 245, 245), width=3)
            draw.text((x_value - radius, y_value + radius + 3), agent[-1], fill=(255, 255, 255))
        defender_x, defender_y = pixel(self.defender["position"])
        defender_radius = max(5, int(self.env_config["player_radius"] * scale))
        draw.ellipse(
            [
                defender_x - defender_radius,
                defender_y - defender_radius,
                defender_x + defender_radius,
                defender_y + defender_radius,
            ],
            fill=(230, 78, 70),
            outline=(90, 15, 15),
            width=2,
        )
        ball_x, ball_y = pixel(self.ball["position"])
        ball_radius = max(4, int(self.env_config["ball_radius"] * scale))
        draw.ellipse(
            [ball_x - ball_radius, ball_y - ball_radius, ball_x + ball_radius, ball_y + ball_radius],
            fill=(248, 242, 218),
            outline=(25, 25, 25),
            width=2,
        )
        simulator_name = self.__class__.__name__.replace("Soccer", " ").replace("Env", "").strip()
        overlay = (
            f"{simulator_name} | seed {self.seed_value} | step {self.step_count} | "
            f"profile {self.selected_profile}"
        )
        draw.rectangle([margin_x, 16, width - margin_x, 52], fill=(20, 25, 30))
        draw.text((margin_x + 12, 27), overlay, fill=(245, 245, 245))
        action_text = " | ".join(
            agent + ": " + ACTION_NAMES[self.players[agent]["current_action"]]
            for agent in self.possible_agents
        )
        striker = min(
            self.possible_agents,
            key=lambda agent: np.linalg.norm(
                self.players[agent]["position"] - self.ball["position"]
            ),
        )
        supporter = AGENTS[1 - AGENTS.index(striker)]
        role_text = " | inferred roles: " + striker + " striker, " + supporter + " support"
        draw.text((margin_x + 12, height - 39), action_text + role_text, fill=(235, 235, 235))
        return np.asarray(image, dtype=np.uint8)

    def close(self):
        pass


class AbstractSoccerEnv(SoccerEnvBase):
    """Fast explicit kinematic simulator used for policy training."""

    def _physics_substep(self, actions, dt):
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        for agent in self.possible_agents:
            player = self.players[agent]
            desired_velocity, desired_heading = self._desired_motion(agent, actions[agent], dt)
            acceleration = (
                self.env_config["player_acceleration"] * self.sampled_parameters["acceleration_multiplier"]
            )
            change = clip_length(desired_velocity - player["velocity"], acceleration * dt)
            player["velocity"] += change
            maximum_speed = self.env_config["player_max_speed"] * self.sampled_parameters["speed_multiplier"]
            player["velocity"] = clip_length(player["velocity"], maximum_speed)
            heading_error = angle_wrap(desired_heading - player["heading"])
            maximum_turn = (
                self.env_config["player_max_turn_rate"]
                * self.sampled_parameters["angular_speed_multiplier"]
            )
            player["angular_velocity"] = float(np.clip(heading_error / max(dt, 1e-6), -maximum_turn, maximum_turn))
            player["heading"] = angle_wrap(player["heading"] + player["angular_velocity"] * dt)
            player["position"] += player["velocity"] * dt
            player["position"][0] = np.clip(
                player["position"][0], -half_length + player["radius"], half_length - player["radius"]
            )
            player["position"][1] = np.clip(
                player["position"][1], -half_width + player["radius"], half_width - player["radius"]
            )
        self._simulate_defender_kinematics(dt)
        self._resolve_agent_collisions()
        self._resolve_ball_contacts()
        drag = self.env_config["ball_drag"] * self.sampled_parameters["ball_drag_multiplier"]
        self.ball["velocity"] *= max(0.0, 1.0 - drag * dt)
        self.ball["velocity"] = clip_length(self.ball["velocity"], self.env_config["shoot_speed"] * 1.7)
        self.ball["position"] += self.ball["velocity"] * dt
        self._defender_ball_history.append(self.ball["position"].copy())

    def _simulate_defender_kinematics(self, dt):
        desired_velocity, desired_heading = self._defender_desired_motion()
        acceleration = self.opponent_config["acceleration"]
        change = clip_length(desired_velocity - self.defender["velocity"], acceleration * dt)
        self.defender["velocity"] += change
        maximum = self.opponent_config["max_speed"] * self.sampled_parameters["defender_speed_multiplier"]
        self.defender["velocity"] = clip_length(self.defender["velocity"], maximum)
        turn = self.opponent_config["max_turn_rate"]
        error = angle_wrap(desired_heading - self.defender["heading"])
        self.defender["angular_velocity"] = float(np.clip(error / max(dt, 1e-6), -turn, turn))
        self.defender["heading"] = angle_wrap(self.defender["heading"] + self.defender["angular_velocity"] * dt)
        self.defender["position"] += self.defender["velocity"] * dt
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        radius = self.env_config["player_radius"]
        self.defender["position"][0] = np.clip(self.defender["position"][0], -half_length + radius, half_length - radius)
        self.defender["position"][1] = np.clip(self.defender["position"][1], -half_width + radius, half_width - radius)
        self._defender_clear_if_possible()

    def _resolve_agent_collisions(self):
        entities = [(agent, self.players[agent]) for agent in self.possible_agents]
        entities.append(("defender", self.defender))
        radius = self.env_config["player_radius"]
        for first_index in range(len(entities)):
            for second_index in range(first_index + 1, len(entities)):
                first_name, first = entities[first_index]
                second_name, second = entities[second_index]
                delta = second["position"] - first["position"]
                distance = float(np.linalg.norm(delta))
                overlap = 2.0 * radius - distance
                if overlap > 0:
                    normal = unit_vector(delta) if distance > 1e-8 else np.array([1.0, 0.0])
                    first["position"] -= normal * overlap * 0.5
                    second["position"] += normal * overlap * 0.5
                    first["velocity"] -= normal * max(0.0, np.dot(first["velocity"], normal)) * 0.5
                    second["velocity"] += normal * max(0.0, -np.dot(second["velocity"], normal)) * 0.5
                    if first_name in AGENTS and second_name in AGENTS:
                        self.metrics["attacker_collisions"] += 1

    def _resolve_ball_contacts(self):
        entities = [(agent, self.players[agent]) for agent in self.possible_agents]
        entities.append(("defender", self.defender))
        for name, entity in entities:
            delta = self.ball["position"] - entity["position"]
            distance = float(np.linalg.norm(delta))
            minimum = self.env_config["player_radius"] + self.env_config["ball_radius"]
            if distance < minimum:
                normal = unit_vector(delta) if distance > 1e-8 else np.array([1.0, 0.0])
                self.ball["position"] = entity["position"] + normal * minimum
                relative = entity["velocity"] - self.ball["velocity"]
                impulse = max(0.0, float(np.dot(relative, normal)))
                self.ball["velocity"] += normal * impulse * 0.75
                self.ball["last_touch"] = name


class PymunkSoccerTransferEnv(SoccerEnvBase):
    """Independent rigid-body transfer simulator using Pymunk/Chipmunk physics."""

    def __init__(self, config, render_mode=None, profile_name=None, profile_probabilities=None):
        if pymunk is None:
            raise ImportError("Pymunk is required for PymunkSoccerTransferEnv; install project dependencies")
        self.space = None
        self.player_bodies = {}
        self.player_shapes = {}
        self.ball_body = None
        self.ball_shape = None
        self.defender_body = None
        self.defender_shape = None
        super().__init__(config, render_mode, profile_name, profile_probabilities)

    def _reset_physics(self):
        self.sampled_parameters["defender_reaction_delay"] += int(
            self.transfer_config.get("changed_reaction_delay_steps", 1)
        )
        self.space = pymunk.Space()
        drag = self.env_config["ball_drag"] * self.sampled_parameters["ball_drag_multiplier"]
        self.space.damping = math.exp(-0.35 * drag)
        self.space.gravity = (0.0, 0.0)
        radius = self.env_config["player_radius"]
        mass = self.transfer_config["player_mass"]
        moment = pymunk.moment_for_circle(mass, 0.0, radius)
        for agent in self.possible_agents:
            body = pymunk.Body(mass, moment)
            body.position = tuple(self.players[agent]["position"])
            body.angle = self.players[agent]["heading"]
            shape = pymunk.Circle(body, radius)
            shape.friction = self.transfer_config["player_friction"]
            shape.elasticity = 0.15
            self.space.add(body, shape)
            self.player_bodies[agent] = body
            self.player_shapes[agent] = shape
        self.defender_body = pymunk.Body(mass, moment)
        self.defender_body.position = tuple(self.defender["position"])
        self.defender_body.angle = self.defender["heading"]
        self.defender_shape = pymunk.Circle(self.defender_body, radius)
        self.defender_shape.friction = self.transfer_config["player_friction"]
        self.defender_shape.elasticity = 0.18
        self.space.add(self.defender_body, self.defender_shape)
        ball_mass = self.transfer_config["ball_mass"] * self.sampled_parameters["ball_mass_multiplier"]
        ball_radius = self.env_config["ball_radius"]
        ball_moment = pymunk.moment_for_circle(ball_mass, 0.0, ball_radius)
        self.ball_body = pymunk.Body(ball_mass, ball_moment)
        self.ball_body.position = tuple(self.ball["position"])
        self.ball_shape = pymunk.Circle(self.ball_body, ball_radius)
        self.ball_shape.friction = self.transfer_config["ball_friction"]
        self.ball_shape.elasticity = self.sampled_parameters["ball_restitution"]
        self.space.add(self.ball_body, self.ball_shape)
        self._add_walls()

    def _add_walls(self):
        half_length = self.env_config["field_length"] / 2.0
        half_width = self.env_config["field_width"] / 2.0
        goal_half = self.env_config["goal_width"] / 2.0
        static = self.space.static_body
        endpoints = [
            ((-half_length, -half_width), (half_length, -half_width)),
            ((-half_length, half_width), (half_length, half_width)),
            ((-half_length, -half_width), (-half_length, -goal_half)),
            ((-half_length, goal_half), (-half_length, half_width)),
            ((half_length, -half_width), (half_length, -goal_half)),
            ((half_length, goal_half), (half_length, half_width)),
        ]
        for start, end in endpoints:
            segment = pymunk.Segment(static, start, end, 0.04)
            segment.friction = 0.55
            segment.elasticity = self.transfer_config["wall_elasticity"]
            self.space.add(segment)

    def _physics_substep(self, actions, dt):
        substeps = int(self.transfer_config["substeps"])
        sub_dt = dt / substeps
        for _ in range(substeps):
            self._sync_from_bodies()
            for agent in self.possible_agents:
                desired_velocity, desired_heading = self._desired_motion(agent, actions[agent], sub_dt)
                self._apply_velocity_controller(
                    self.player_bodies[agent],
                    desired_velocity,
                    desired_heading,
                    self.env_config["player_acceleration"]
                    * self.sampled_parameters["acceleration_multiplier"],
                    self.env_config["player_max_turn_rate"]
                    * self.sampled_parameters["angular_speed_multiplier"],
                    sub_dt,
                )
            defender_velocity, defender_heading = self._defender_desired_motion()
            self._apply_velocity_controller(
                self.defender_body,
                defender_velocity,
                defender_heading,
                self.opponent_config["acceleration"],
                self.opponent_config["max_turn_rate"],
                sub_dt,
            )
            self._defender_clear_if_possible()
            self.space.step(sub_dt)
            self._bound_body_speeds()
        self._sync_from_bodies()
        self._defender_ball_history.append(self.ball["position"].copy())
        separation = np.linalg.norm(self.players[AGENTS[0]]["position"] - self.players[AGENTS[1]]["position"])
        if separation < 2.0 * self.env_config["player_radius"] * 1.01:
            self.metrics["attacker_collisions"] += 1

    def _apply_velocity_controller(self, body, desired_velocity, desired_heading, maximum_acceleration, maximum_turn, dt):
        current = np.array([body.velocity.x, body.velocity.y])
        velocity_error = desired_velocity - current
        desired_acceleration = clip_length(
            velocity_error * self.transfer_config["velocity_control_gain"], maximum_acceleration
        )
        body.apply_force_at_local_point(tuple(desired_acceleration * body.mass), (0.0, 0.0))
        angular_error = angle_wrap(desired_heading - body.angle)
        desired_rate = float(np.clip(angular_error / max(dt, 1e-6), -maximum_turn, maximum_turn))
        body.torque += (desired_rate - body.angular_velocity) * body.moment * 5.0

    def _bound_body_speeds(self):
        maximum = self.env_config["player_max_speed"] * self.sampled_parameters["speed_multiplier"]
        for body in self.player_bodies.values():
            clipped = clip_length(np.array([body.velocity.x, body.velocity.y]), maximum * 1.15)
            body.velocity = tuple(clipped)
            body.angular_velocity = float(
                np.clip(
                    body.angular_velocity,
                    -self.env_config["player_max_turn_rate"] * 1.2,
                    self.env_config["player_max_turn_rate"] * 1.2,
                )
            )
        defender_maximum = self.opponent_config["max_speed"] * self.sampled_parameters["defender_speed_multiplier"]
        self.defender_body.velocity = tuple(
            clip_length(
                np.array([self.defender_body.velocity.x, self.defender_body.velocity.y]),
                defender_maximum * 1.15,
            )
        )
        ball_maximum = self.env_config["shoot_speed"] * 1.9
        self.ball_body.velocity = tuple(
            clip_length(np.array([self.ball_body.velocity.x, self.ball_body.velocity.y]), ball_maximum)
        )

    def _sync_from_bodies(self):
        for agent in self.possible_agents:
            body = self.player_bodies[agent]
            self.players[agent]["position"] = np.array([body.position.x, body.position.y])
            self.players[agent]["velocity"] = np.array([body.velocity.x, body.velocity.y])
            self.players[agent]["heading"] = angle_wrap(float(body.angle))
            self.players[agent]["angular_velocity"] = float(body.angular_velocity)
        self.defender["position"] = np.array([self.defender_body.position.x, self.defender_body.position.y])
        self.defender["velocity"] = np.array([self.defender_body.velocity.x, self.defender_body.velocity.y])
        self.defender["heading"] = angle_wrap(float(self.defender_body.angle))
        self.defender["angular_velocity"] = float(self.defender_body.angular_velocity)
        self.ball["position"] = np.array([self.ball_body.position.x, self.ball_body.position.y])
        self.ball["velocity"] = np.array([self.ball_body.velocity.x, self.ball_body.velocity.y])

    def _deliver_kick(self, direction, speed):
        impulse = direction * speed * self.ball_body.mass
        self.ball_body.apply_impulse_at_world_point(tuple(impulse), tuple(self.ball_body.position))

    def _deliver_dribble_impulse(self, impulse):
        scaled = np.asarray(impulse) * self.ball_body.mass * 3.5
        self.ball_body.apply_impulse_at_world_point(tuple(scaled), tuple(self.ball_body.position))

    def _defender_kick(self, direction, speed):
        impulse = direction * speed * self.ball_body.mass
        self.ball_body.apply_impulse_at_world_point(tuple(impulse), tuple(self.ball_body.position))

    def close(self):
        self.space = None
        self.player_bodies = {}
        self.player_shapes = {}
        self.ball_body = None
        self.ball_shape = None
        self.defender_body = None
        self.defender_shape = None


def baseline_actions(env, method, memory=None):
    """Return valid actions for one of the three deterministic/scripted baselines."""
    if method == "random":
        return {agent: int(env.rng.integers(0, len(ACTION_NAMES))) for agent in env.possible_agents}
    if method == "double_chase":
        actions = {}
        for agent in env.possible_agents:
            distance = np.linalg.norm(env.players[agent]["position"] - env.ball["position"])
            actions[agent] = 3 if distance <= env.env_config["kick_radius"] else 0
        return actions
    if method != "role_based":
        raise ValueError("Unknown baseline method: " + str(method))
    if memory is None:
        memory = {}
    times = {}
    for agent in env.possible_agents:
        distance = np.linalg.norm(env.players[agent]["position"] - env.ball["position"])
        closing = max(0.15, env.env_config["player_max_speed"] + np.dot(
            env.players[agent]["velocity"], unit_vector(env.ball["position"] - env.players[agent]["position"])
        ))
        times[agent] = float(distance / closing)
    candidate = min(times, key=times.get)
    striker = memory.get("striker", candidate)
    if candidate != striker and times[candidate] + 0.25 < times[striker]:
        if memory.get("candidate") == candidate:
            memory["candidate_steps"] = memory.get("candidate_steps", 0) + 1
        else:
            memory["candidate"] = candidate
            memory["candidate_steps"] = 1
        if memory["candidate_steps"] >= 2:
            striker = candidate
            memory["striker"] = striker
            memory["candidate_steps"] = 0
    else:
        memory["striker"] = striker
        memory["candidate_steps"] = 0
    supporter = AGENTS[1 - AGENTS.index(striker)]
    striker_position = env.players[striker]["position"]
    teammate_position = env.players[supporter]["position"]
    ball = env.ball["position"]
    defender = env.defender["position"]
    kickable = np.linalg.norm(striker_position - ball) <= env.env_config["kick_radius"]
    goal = np.array([env.env_config["field_length"] / 2.0, 0.0])
    shooting_blocked = distance_to_segment(defender, ball, goal) < 0.48
    teammate_lane_open = distance_to_segment(defender, ball, teammate_position) > 0.55
    if kickable and shooting_blocked and teammate_lane_open and teammate_position[0] > ball[0] - 0.4:
        striker_action = 4
    elif kickable and ball[0] > 0.5:
        striker_action = 3
    elif kickable:
        striker_action = 1 if defender[1] <= ball[1] else 2
    else:
        striker_action = 0
    actions = {striker: striker_action, supporter: 5}
    if np.linalg.norm(teammate_position - striker_position) < 0.55:
        actions[supporter] = 5
    memory["roles"] = {striker: "striker", supporter: "support"}
    return actions


def make_environment(config, simulator="abstract", **kwargs):
    if simulator == "abstract":
        return AbstractSoccerEnv(config, **kwargs)
    if simulator == "pymunk":
        return PymunkSoccerTransferEnv(config, **kwargs)
    raise ValueError("Unsupported simulator: " + str(simulator))


def validate_parallel_environments(config, cycles=100):
    """Run PettingZoo's official parallel API checks for both mandatory simulators."""
    from pettingzoo.test import parallel_api_test

    for simulator in ["abstract", "pymunk"]:
        env = make_environment(config, simulator)
        try:
            parallel_api_test(env, num_cycles=cycles)
        finally:
            env.close()
