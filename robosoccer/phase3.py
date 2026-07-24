"""Phase 3 small-sided team-play environments, baselines, and calibration.

The phase keeps a fixed three-attacker observation/state interface while activating
two or three attackers. Abstract and Pymunk transitions share task semantics but use
separate integration paths. This module intentionally contains the cohesive Phase 3
task layer; Phase 2 environments remain frozen for reproducibility.
"""

import copy
import json
import logging
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from PIL import Image, ImageDraw, ImageFont

from robosoccer.environment import ACTION_NAMES, angle_wrap, clip_length, one_hot, unit_vector
from robosoccer.utils import write_json

try:
    import pymunk
except ImportError:
    pymunk = None

logger = logging.getLogger(__name__)

PHASE3_AGENTS = ["attacker_0", "attacker_1", "attacker_2"]
PHASE3_DEFENDERS = ["goalie", "field_defender_0", "field_defender_1"]
PHASE3_SCENARIOS = {
    "phase3_2v2_open": {
        "attackers": 2,
        "defenders": 2,
        "pass_required": False,
        "press": False,
    },
    "phase3_2v2_pass_required": {
        "attackers": 2,
        "defenders": 2,
        "pass_required": True,
        "press": False,
    },
    "phase3_3v2_open": {
        "attackers": 3,
        "defenders": 2,
        "pass_required": False,
        "press": False,
    },
    "phase3_3v2_press": {
        "attackers": 3,
        "defenders": 2,
        "pass_required": False,
        "press": True,
    },
}
DEFENDER_STYLES = ["goalie", "lane_block", "press", "predictive", "zonal", "mixed"]
PHASE3_OBSERVATION_SCHEMA = 1
PHASE3_STATE_SCHEMA = 1


def phase3_scenario(config, name=None):
    phase3 = config.get("phase3", {})
    selected = name or phase3.get("scenario", "phase3_2v2_open")
    configured = phase3.get("scenarios", {}).get(selected, {})
    if selected not in PHASE3_SCENARIOS:
        raise ValueError("Unknown Phase 3 scenario: " + str(selected))
    result = copy.deepcopy(PHASE3_SCENARIOS[selected])
    result.update(copy.deepcopy(configured))
    result["name"] = selected
    return result


def _safe_normalized(vector, scales):
    return np.asarray(vector, dtype=np.float32) / np.asarray(scales, dtype=np.float32)


def _segment_distance(point, start, end):
    segment = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-10:
        return float(np.linalg.norm(np.asarray(point) - np.asarray(start)))
    fraction = np.dot(np.asarray(point) - np.asarray(start), segment) / denominator
    projection = np.asarray(start) + np.clip(fraction, 0.0, 1.0) * segment
    return float(np.linalg.norm(np.asarray(point) - projection))


class Phase3SoccerEnv(ParallelEnv):
    """Fixed-roster 2-v-2/3-v-2 environment with explicit masks and team-play events."""

    metadata = {"render_modes": ["rgb_array"], "name": "robosoccer_phase3_v1", "render_fps": 20}

    def __init__(
        self,
        config,
        simulator="abstract",
        render_mode=None,
        scenario=None,
        profile_name=None,
        defender_style=None,
    ):
        if simulator not in {"abstract", "pymunk"}:
            raise ValueError("simulator must be abstract or pymunk")
        if simulator == "pymunk" and pymunk is None:
            raise ImportError("Pymunk is required for the Phase 3 rigid-body environment")
        self.config = copy.deepcopy(config)
        self.env_config = self.config["environment"]
        self.phase3_config = self.config.get("phase3", {})
        self.reward_config = self.config.get("phase3_reward", self.config["reward"])
        self.simulator = simulator
        self.render_mode = render_mode
        self.forced_scenario = scenario
        self.forced_profile = profile_name
        self.forced_defender_style = defender_style
        maximum_attackers = int(self.phase3_config.get("maximum_attackers", 3))
        maximum_defenders = int(self.phase3_config.get("maximum_defenders", 3))
        if not 2 <= maximum_attackers <= len(PHASE3_AGENTS):
            raise ValueError("Phase 3 maximum_attackers must be between 2 and 3")
        if not 2 <= maximum_defenders <= len(PHASE3_DEFENDERS):
            raise ValueError("Phase 3 maximum_defenders must be between 2 and 3")
        self.possible_agents = PHASE3_AGENTS[:maximum_attackers]
        self.possible_defenders = PHASE3_DEFENDERS[:maximum_defenders]
        self.agents = []
        self.action_size = len(ACTION_NAMES)
        self.observation_dimension = (
            10
            + 8
            + 4
            + (maximum_attackers - 1) * 10
            + maximum_defenders * 6
            + self.action_size * 3
            + 4
            + int(self.phase3_config.get("observation_padding", 0))
        )
        self.state_dimension = (
            maximum_attackers * 8
            + maximum_defenders * 7
            + 4
            + maximum_attackers
            + 1
            + 6
            + int(self.phase3_config.get("state_padding", 0))
        )
        self._observation_spaces = {
            agent: spaces.Box(-10.0, 10.0, (self.observation_dimension,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(self.action_size) for agent in self.possible_agents
        }
        self.state_space = spaces.Box(-10.0, 10.0, (self.state_dimension,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.seed_value = None
        self.scenario = None
        self.active_agents = []
        self.active_defenders = []
        self.players = {}
        self.defenders = {}
        self.ball = {}
        self.metrics = {}
        self.step_count = 0
        self.match_score = 0
        self.match_restart_count = 0
        self.selected_profile = "nominal"
        self.scenario_group = "unspecified"
        self.defender_style = "lane_block"
        self.last_possessor = None
        self.pending_pass = None
        self.pass_chain = []
        self.chain_progress_high_water = 0.0
        self.cooperative_sequence = False
        self.cooperative_horizon_remaining = 0
        self.termination_reason = None
        self._last_potential = 0.0
        self._last_potential_components = {
            "ball_progress": 0.0,
            "team_possession": 0.0,
            "support_quality": 0.0,
        }
        self._event_rewards = {}
        self._action_masks = {}
        self.action_queues = {}
        self.observation_queues = {}
        self.message_queues = {}
        self.perception_masks = {}
        self.last_nearest_agent = None
        self.sequence_start_step = 0
        self.defender_clear_event = False
        self.restart_pause_remaining = 0
        self.space = None
        self.player_bodies = {}
        self.defender_bodies = {}
        self.ball_body = None

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def action_mask(self, agent):
        if agent not in self.active_agents:
            return np.zeros(self.action_size, dtype=np.float32)
        mask = np.ones(self.action_size, dtype=np.float32)
        possesses = self.ball.get("possession") == agent
        if not possesses:
            mask[3] = 0.0
            mask[4] = 0.0
        if len(self.active_agents) < 2:
            mask[4] = 0.0
        elif possesses:
            _, target = self._pass_target(agent)
            half_length = float(self.env_config["field_length"]) / 2.0
            half_width = float(self.env_config["field_width"]) / 2.0
            if (
                target is None
                or abs(float(target[0])) >= half_length
                or abs(float(target[1])) >= half_width
            ):
                mask[4] = 0.0
        return mask

    def action_masks(self):
        return {agent: self.action_mask(agent) for agent in self.active_agents}

    def _select_scenario(self, options):
        selected = options.get("scenario", self.forced_scenario)
        if selected is None:
            names = self.phase3_config.get("scenario_mixture", ["phase3_2v2_open"])
            probabilities = self.phase3_config.get("scenario_probabilities")
            selected = str(self.rng.choice(names, p=probabilities))
        return phase3_scenario(self.config, selected)

    def _select_defender_style(self, options):
        selected = options.get("defender_style", self.forced_defender_style)
        if selected is None:
            selected = self.phase3_config.get("defender_style", "mixed")
        if selected not in DEFENDER_STYLES:
            raise ValueError("Unknown Phase 3 defender style: " + str(selected))
        if selected == "mixed":
            names = self.phase3_config.get(
                "mixed_defender_styles", ["lane_block", "press", "predictive", "zonal"]
            )
            probabilities = self.phase3_config.get("mixed_defender_probabilities")
            selected = str(self.rng.choice(names, p=probabilities))
        return selected

    def _sample_profile(self, options):
        selected = options.get("profile", self.forced_profile or "nominal")
        profile = self.phase3_config.get("profiles", {}).get(selected, {})
        if selected != "nominal" and not profile:
            raise ValueError("Unknown Phase 3 profile: " + str(selected))
        parameters = {
            "speed_multiplier": 1.0,
            "ball_drag_multiplier": 1.0,
            "kick_multiplier": 1.0,
            "localization_noise": 0.0,
            "packet_loss": 0.0,
            "action_latency": 0,
        }
        for key, value in profile.items():
            if isinstance(value, list) and len(value) == 2:
                if all(isinstance(item, int) for item in value):
                    parameters[key] = int(self.rng.integers(value[0], value[1] + 1))
                else:
                    parameters[key] = float(self.rng.uniform(value[0], value[1]))
            else:
                parameters[key] = copy.deepcopy(value)
        parameters.update(options.get("sampled_parameters", {}))
        return selected, parameters

    def reset(self, seed=None, options=None):
        options = options or {}
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        elif self.seed_value is None:
            self.seed_value = int(np.random.SeedSequence().entropy) % (2**31)
            self.rng = np.random.default_rng(self.seed_value)
        self.scenario = self._select_scenario(options)
        if int(self.scenario["attackers"]) > len(self.possible_agents):
            raise ValueError("Scenario attacker count exceeds configured maximum")
        if int(self.scenario["defenders"]) > len(self.possible_defenders):
            raise ValueError("Scenario defender count exceeds configured maximum")
        self.active_agents = self.possible_agents[: int(self.scenario["attackers"])]
        self.active_defenders = self.possible_defenders[: int(self.scenario["defenders"])]
        self.agents = self.active_agents[:]
        self.defender_style = self._select_defender_style(options)
        self.selected_profile, self.sampled_parameters = self._sample_profile(options)
        self.scenario_group = str(options.get("scenario_group", "evaluation"))
        self.step_count = 0
        self.match_score = 0
        self.match_restart_count = 0
        self.termination_reason = None
        self.pass_chain = []
        self.pending_pass = None
        self.cooperative_sequence = False
        self.cooperative_horizon_remaining = 0
        self.last_possessor = None
        self.last_nearest_agent = None
        self.sequence_start_step = 0
        self.defender_clear_event = False
        self.restart_pause_remaining = 0
        self._initialize_entities(options.get("initial_state"))
        self._initialize_physics()
        latency = int(self.sampled_parameters.get("action_latency", 0))
        self.action_queues = {
            agent: deque([6] * latency) for agent in self.active_agents
        }
        self._reset_delay_queues()
        self._reset_metrics()
        self._update_possession()
        self._reset_chain_progress()
        self._last_potential_components = self._potential_components()
        self._last_potential = sum(self._last_potential_components.values())
        observations = self._observations()
        infos = {agent: self._info(agent) for agent in self.active_agents}
        return observations, infos

    def _initialize_entities(self, initial_state=None):
        half_length = float(self.env_config["field_length"]) / 2.0
        half_width = float(self.env_config["field_width"]) / 2.0
        starts = [
            np.array([-0.33 * half_length, -0.22 * half_width]),
            np.array([-0.18 * half_length, 0.32 * half_width]),
            np.array([-0.43 * half_length, 0.02 * half_width]),
        ]
        jitter = float(self.phase3_config.get("start_jitter", 0.18))
        self.players = {}
        for index, agent in enumerate(self.active_agents):
            self.players[agent] = {
                "position": starts[index] + self.rng.uniform(-jitter, jitter, size=2),
                "velocity": np.zeros(2, dtype=np.float64),
                "heading": 0.0,
                "previous_action": 6,
                "executed_action": 6,
                "possesses_ball": False,
                "message": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            }
        self.motion_trails = {
            agent: deque([record["position"].copy()], maxlen=8)
            for agent, record in self.players.items()
        }
        goal_x = half_length - 0.45
        defender_starts = {
            "goalie": np.array([goal_x, 0.0]),
            "field_defender_0": np.array([0.55, 0.0]),
            "field_defender_1": np.array([1.25, 1.0]),
        }
        self.defenders = {}
        for name in self.active_defenders:
            position = defender_starts[name].copy()
            if name != "goalie":
                position += self.rng.uniform(-jitter, jitter, size=2)
            self.defenders[name] = {
                "position": position,
                "velocity": np.zeros(2, dtype=np.float64),
                "heading": math.pi,
            }
        carrier = self.active_agents[0]
        ball_position = self.players[carrier]["position"] + np.array([0.34, 0.0])
        if self.scenario["pass_required"]:
            self.defenders["field_defender_0"]["position"] = np.array([0.15, -0.10])
            ball_position = self.players[carrier]["position"] + np.array([0.29, 0.0])
        self.ball = {
            "position": ball_position,
            "velocity": np.zeros(2, dtype=np.float64),
            "possession": None,
            "last_touch": None,
            "shot_in_flight": False,
            "cooperative_shot": False,
        }
        if initial_state:
            self._apply_initial_state(initial_state)

    def _apply_initial_state(self, state):
        for agent, record in state.get("players", {}).items():
            if agent not in self.players:
                raise ValueError("Initial state references inactive attacker: " + str(agent))
            for key in ["position", "velocity"]:
                if key in record:
                    value = np.asarray(record[key], dtype=np.float64)
                    if value.shape != (2,) or not np.isfinite(value).all():
                        raise ValueError("Initial " + key + " must be a finite 2-vector")
                    self.players[agent][key] = value
            if "heading" in record:
                self.players[agent]["heading"] = angle_wrap(float(record["heading"]))
        for name, record in state.get("defenders", {}).items():
            if name not in self.defenders:
                raise ValueError("Initial state references inactive defender: " + str(name))
            for key in ["position", "velocity"]:
                if key in record:
                    self.defenders[name][key] = np.asarray(record[key], dtype=np.float64)
        for key in ["position", "velocity"]:
            if key in state.get("ball", {}):
                self.ball[key] = np.asarray(state["ball"][key], dtype=np.float64)

    def _initialize_physics(self):
        if self.simulator != "pymunk":
            self.space = None
            return
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 1.0
        radius = float(self.env_config["player_radius"])
        ball_radius = float(self.env_config["ball_radius"])
        self.player_bodies = {}
        self.defender_bodies = {}
        for agent, player in self.players.items():
            body = pymunk.Body(2.4, pymunk.moment_for_circle(2.4, 0.0, radius))
            body.position = tuple(player["position"])
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.05
            shape.friction = 0.8
            self.space.add(body, shape)
            self.player_bodies[agent] = body
        for name, defender in self.defenders.items():
            body = pymunk.Body(2.6, pymunk.moment_for_circle(2.6, 0.0, radius))
            body.position = tuple(defender["position"])
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.05
            shape.friction = 0.8
            self.space.add(body, shape)
            self.defender_bodies[name] = body
        self.ball_body = pymunk.Body(0.43, pymunk.moment_for_circle(0.43, 0.0, ball_radius))
        self.ball_body.position = tuple(self.ball["position"])
        ball_shape = pymunk.Circle(self.ball_body, ball_radius)
        ball_shape.elasticity = float(self.env_config["ball_restitution"])
        ball_shape.friction = 0.45
        self.space.add(self.ball_body, ball_shape)

    def _reset_metrics(self):
        self.metrics = {
            "scenario": self.scenario["name"],
            "simulator": self.simulator,
            "defender_style": self.defender_style,
            "scenario_group": self.scenario_group,
            "goal": 0,
            "cooperative_success": 0,
            "pass_attempts": 0,
            "valid_pass_attempts": 0,
            "completed_receptions": 0,
            "pass_to_goal": 0,
            "pass_and_advance": 0,
            "possessions": 0,
            "possession_steps": 0,
            "turnovers": 0,
            "interceptions": 0,
            "defender_clears": 0,
            "shots": 0,
            "shots_on_target": 0,
            "direct_shots_pass_required": 0,
            "role_switches": 0,
            "nearest_steps_attacker_0": 0,
            "nearest_steps_attacker_1": 0,
            "nearest_steps_attacker_2": 0,
            "pairwise_separation_sum": 0.0,
            "field_width_sum": 0.0,
            "collisions": 0,
            "out_of_bounds": 0,
            "completed_sequences": 0,
            "sequence_steps_sum": 0,
            "possession_switches": 0,
            "redundant_chase_steps": 0,
            "active_steps": 0,
            "invalid_action_requests": 0,
            "action_switches": 0,
            "possession_chain_max": 0,
            "expected_threat_gain": 0.0,
            "new_chain_progress": 0.0,
            "chain_progress_high_water": 0.0,
            "match_restarts": 0,
            "match_score": 0,
            "terminal_reason": None,
            "episode_steps": 0,
            "time_to_score": None,
            "reward_goal": 0.0,
            "reward_ball_progress": 0.0,
            "reward_team_possession": 0.0,
            "reward_controlled_pass": 0.0,
            "reward_pass_and_advance": 0.0,
            "reward_chain_progress": 0.0,
            "reward_pass_and_goal": 0.0,
            "reward_support_quality": 0.0,
            "reward_collision": 0.0,
            "reward_out_of_bounds": 0.0,
            "reward_turnover": 0.0,
            "reward_time": 0.0,
        }
        for action_name in ACTION_NAMES:
            self.metrics["requested_" + action_name] = 0
            self.metrics["masked_" + action_name] = 0

    def _uses_chain_frontier_reward(self):
        return int(self.phase3_config.get("reward_schema_version", 1)) >= 2

    def _chain_progress_value(self):
        return float(self.ball["position"][0]) / float(self.env_config["field_length"])

    def _reset_chain_progress(self):
        self.chain_progress_high_water = self._chain_progress_value()
        if self.metrics:
            self.metrics["chain_progress_high_water"] = self.chain_progress_high_water

    def _track_controlled_chain_progress(self):
        if not self._uses_chain_frontier_reward():
            return
        current = self._chain_progress_value()
        self.chain_progress_high_water = max(self.chain_progress_high_water, current)
        self.metrics["chain_progress_high_water"] = max(
            self.metrics["chain_progress_high_water"],
            self.chain_progress_high_water,
        )

    def _claim_new_chain_progress(self):
        current = self._chain_progress_value()
        new_progress = max(0.0, current - self.chain_progress_high_water)
        self.chain_progress_high_water = max(self.chain_progress_high_water, current)
        self.metrics["new_chain_progress"] += new_progress
        self.metrics["chain_progress_high_water"] = max(
            self.metrics["chain_progress_high_water"],
            self.chain_progress_high_water,
        )
        return new_progress

    def _reset_attacking_chain(self):
        self.pass_chain = []
        self.pending_pass = None
        self.cooperative_sequence = False
        self.cooperative_horizon_remaining = 0
        self.last_possessor = None
        self._reset_chain_progress()

    def _reset_delay_queues(self):
        observation_latency = int(
            self.sampled_parameters.get("observation_latency", 0)
        )
        communication_latency = int(
            self.sampled_parameters.get("communication_latency", 0)
        )
        self.observation_queues = {
            agent: deque(maxlen=observation_latency + 1)
            for agent in self.active_agents
        }
        invalid_message = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.message_queues = {
            agent: deque(
                [invalid_message.copy() for _ in range(communication_latency)]
            )
            for agent in self.active_agents
        }
        self.perception_masks = {
            agent: {"ball": True, "defenders": True}
            for agent in self.active_agents
        }

    def _profiled_position(self, position):
        noise = float(self.sampled_parameters.get("localization_noise", 0.0))
        if noise <= 0:
            return np.asarray(position)
        return np.asarray(position) + self.rng.normal(0.0, noise, size=2)

    def _observations(self):
        missed_ball = float(
            self.sampled_parameters.get("missed_ball_probability", 0.0)
        )
        missed_defender = float(
            self.sampled_parameters.get("missed_defender_probability", 0.0)
        )
        current = {}
        for agent in self.active_agents:
            self.perception_masks[agent] = {
                "ball": True
                if missed_ball <= 0.0
                else self.rng.random() >= missed_ball,
                "defenders": True
                if missed_defender <= 0.0
                else self.rng.random() >= missed_defender,
            }
            current[agent] = self.observe(agent)
        result = {}
        for agent, observation in current.items():
            queue = self.observation_queues[agent]
            if not queue:
                queue.extend(
                    [observation.copy() for _ in range(queue.maxlen - 1)]
                )
            queue.append(observation.copy())
            result[agent] = queue[0].copy()
        return result

    def observe(self, agent):
        if agent not in self.active_agents:
            return np.zeros(self.observation_dimension, dtype=np.float32)
        half_length = float(self.env_config["field_length"]) / 2.0
        half_width = float(self.env_config["field_width"]) / 2.0
        speed = max(1.0, float(self.env_config["player_max_speed"]))
        player = self.players[agent]
        position = self._profiled_position(player["position"])
        perception = self.perception_masks.get(
            agent, {"ball": True, "defenders": True}
        )
        values = []
        values.extend(_safe_normalized(position, [half_length, half_width]))
        values.extend(_safe_normalized(player["velocity"], [speed, speed]))
        values.extend([math.sin(player["heading"]), math.cos(player["heading"])])
        values.extend(
            [
                float(self.ball["possession"] == agent),
                float(PHASE3_AGENTS.index(agent)) / 2.0,
                float(len(self.active_agents)) / 3.0,
                float(len(self.active_defenders)) / 3.0,
            ]
        )
        relative_ball = (
            self._profiled_position(self.ball["position"]) - position
            if perception["ball"]
            else np.zeros(2)
        )
        values.extend(_safe_normalized(relative_ball, [half_length, half_width]))
        perceived_ball_velocity = (
            self.ball["velocity"] if perception["ball"] else np.zeros(2)
        )
        values.extend(
            _safe_normalized(
                perceived_ball_velocity, [speed * 2.0, speed * 2.0]
            )
        )
        values.extend(
            [
                float(perception["ball"] and self.ball["possession"] is not None),
                float(perception["ball"] and self.ball["last_touch"] == agent),
                float(self.pending_pass is not None),
                min(1.0, len(self.pass_chain) / 4.0),
            ]
        )
        goal = np.array([half_length, 0.0])
        values.extend(_safe_normalized(goal - position, [half_length * 2.0, half_width]))
        values.extend([float(self.scenario["pass_required"]), float(self.scenario["press"])])
        for teammate in self.possible_agents:
            if teammate == agent:
                continue
            if teammate in self.players:
                record = self.players[teammate]
                values.append(1.0)
                values.extend(
                    _safe_normalized(
                        self._profiled_position(record["position"]) - position,
                        [half_length, half_width],
                    )
                )
                values.extend(_safe_normalized(record["velocity"], [speed, speed]))
                values.append(float(self.ball["possession"] == teammate))
                values.extend(record["message"])
            else:
                values.extend([0.0] * 10)
        for defender in self.possible_defenders:
            if defender in self.defenders and perception["defenders"]:
                record = self.defenders[defender]
                values.append(1.0)
                values.extend(
                    _safe_normalized(
                        self._profiled_position(record["position"]) - position,
                        [half_length, half_width],
                    )
                )
                values.extend(_safe_normalized(record["velocity"], [speed, speed]))
                values.append(float(defender == "goalie"))
            else:
                values.extend([0.0] * 6)
        values.extend(one_hot(player["previous_action"], self.action_size))
        values.extend(one_hot(player["executed_action"], self.action_size))
        values.extend(self.action_mask(agent))
        values.extend(
            [
                min(1.0, self.step_count / max(1, self._maximum_steps())),
                min(1.0, self.match_restart_count / 10.0),
                float(self.cooperative_sequence),
                float(self.selected_profile != "nominal"),
            ]
        )
        array = np.asarray(values, dtype=np.float32)
        if len(array) > self.observation_dimension:
            raise RuntimeError(
                f"Phase 3 observation schema overflow: {len(array)} > {self.observation_dimension}"
            )
        result = np.zeros(self.observation_dimension, dtype=np.float32)
        result[: len(array)] = array
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite Phase 3 observation")
        return result

    def state(self):
        half_length = float(self.env_config["field_length"]) / 2.0
        half_width = float(self.env_config["field_width"]) / 2.0
        speed = max(1.0, float(self.env_config["player_max_speed"]))
        values = []
        for agent in self.possible_agents:
            if agent in self.players:
                record = self.players[agent]
                values.append(1.0)
                values.extend(_safe_normalized(record["position"], [half_length, half_width]))
                values.extend(_safe_normalized(record["velocity"], [speed, speed]))
                values.extend([math.sin(record["heading"]), math.cos(record["heading"])])
                values.append(float(self.ball["possession"] == agent))
            else:
                values.extend([0.0] * 8)
        for name in self.possible_defenders:
            if name in self.defenders:
                record = self.defenders[name]
                values.append(1.0)
                values.extend(_safe_normalized(record["position"], [half_length, half_width]))
                values.extend(_safe_normalized(record["velocity"], [speed, speed]))
                values.extend([math.sin(record["heading"]), math.cos(record["heading"])])
            else:
                values.extend([0.0] * 7)
        values.extend(_safe_normalized(self.ball["position"], [half_length, half_width]))
        values.extend(_safe_normalized(self.ball["velocity"], [speed * 2.0, speed * 2.0]))
        for possible in [None, *self.possible_agents]:
            values.append(float(self.ball["possession"] == possible))
        values.extend(
            [
                float(self.scenario["pass_required"]),
                float(self.scenario["press"]),
                float(len(self.active_agents)) / 3.0,
                float(len(self.active_defenders)) / 3.0,
                min(1.0, self.step_count / max(1, self._maximum_steps())),
                float(self.cooperative_sequence),
            ]
        )
        array = np.asarray(values, dtype=np.float32)
        if len(array) > self.state_dimension:
            raise RuntimeError(f"Phase 3 state schema overflow: {len(array)}")
        result = np.zeros(self.state_dimension, dtype=np.float32)
        result[: len(array)] = array
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite Phase 3 global state")
        return result

    def _maximum_steps(self):
        if self.phase3_config.get("match_mode", False):
            seconds = float(self.phase3_config.get("match_seconds", 50.0))
            decision_seconds = float(self.env_config["dt"]) * int(
                self.env_config["macro_action_repeat"]
            )
            return max(1, int(seconds / decision_seconds))
        return int(self.phase3_config.get("max_episode_steps", self.env_config["max_episode_steps"]))

    def _desired_attacker_motion(self, agent, action):
        player = self.players[agent]
        ball = self.ball["position"]
        goal = np.array([float(self.env_config["field_length"]) / 2.0, 0.0])
        if action == 0:
            target = ball
        elif action == 1:
            target = ball + np.array([0.45, 0.65])
        elif action == 2:
            target = ball + np.array([0.45, -0.65])
        elif action == 5:
            teammates = [
                self.players[name]["position"] for name in self.active_agents if name != agent
            ]
            carrier = self.ball["possession"]
            anchor = self.players[carrier]["position"] if carrier in self.players else ball
            side = -1.0 if PHASE3_AGENTS.index(agent) % 2 == 0 else 1.0
            target = anchor + np.array([0.9, side * 1.15])
            if teammates:
                target[1] += 0.12 * np.mean([item[1] for item in teammates])
        else:
            target = player["position"]
        desired = unit_vector(target - player["position"])
        speed = float(self.env_config["player_max_speed"])
        speed *= float(self.sampled_parameters.get("speed_multiplier", 1.0))
        if action == 6:
            speed = 0.0
        return desired * speed, goal

    def _turn_toward(self, record, direction, rate, dt):
        if np.linalg.norm(direction) <= 1e-8:
            return
        target = math.atan2(direction[1], direction[0])
        change = float(np.clip(angle_wrap(target - record["heading"]), -rate * dt, rate * dt))
        record["heading"] = angle_wrap(record["heading"] + change)

    def _defender_target(self, name):
        half_length = float(self.env_config["field_length"]) / 2.0
        if name == "goalie":
            y_value = float(np.clip(self.ball["position"][1] * 0.65, -0.62, 0.62))
            return np.array([half_length - 0.42, y_value])
        style = self.defender_style
        if self.scenario["pass_required"] and not self.cooperative_sequence:
            carrier = self.ball["possession"]
            if carrier in self.players:
                start = self.players[carrier]["position"]
            else:
                start = self.ball["position"]
            return start + 0.48 * (np.array([half_length, 0.0]) - start)
        if style == "press":
            return self.ball["position"]
        if style == "predictive":
            return self.ball["position"] + 0.55 * self.ball["velocity"]
        if style == "zonal":
            return np.array([0.75, np.clip(self.ball["position"][1], -1.25, 1.25)])
        carrier = self.ball["possession"]
        start = self.players[carrier]["position"] if carrier in self.players else self.ball["position"]
        return start + 0.55 * (np.array([half_length, 0.0]) - start)

    def _pass_target(self, agent):
        candidates = [name for name in self.active_agents if name != agent]
        if not candidates:
            return None, None
        receiver = max(candidates, key=lambda name: self.players[name]["position"][0])
        target = (
            self.players[receiver]["position"]
            + 0.30 * self.players[receiver]["velocity"]
        )
        return receiver, target

    def _execute_kicks(self, actions):
        goal_x = float(self.env_config["field_length"]) / 2.0 + 0.3
        kick_speed = float(self.env_config["shoot_speed"])
        kick_speed *= float(self.sampled_parameters.get("kick_multiplier", 1.0))
        for agent in self.active_agents:
            action = actions[agent]
            if self.ball["possession"] != agent:
                continue
            if action == 3:
                self.metrics["shots"] += 1
                if self.scenario["pass_required"] and not self.cooperative_sequence:
                    self.metrics["direct_shots_pass_required"] += 1
                goalie_y = float(self.defenders.get("goalie", {"position": [0.0, 0.0]})["position"][1])
                if abs(goalie_y) < 0.08:
                    goalie_y = float(self.players[agent]["position"][1])
                target_y = (
                    -1.0 if goalie_y >= 0.0 else 1.0
                ) * float(self.env_config["goal_width"]) * 0.30
                goal = np.array([goal_x, target_y])
                direction = unit_vector(goal - self.ball["position"])
                shot_speed = kick_speed * (1.25 if self.cooperative_sequence else 1.0)
                self._release_ball(agent, direction * shot_speed, "shot")
            elif action == 4:
                receiver, target = self._pass_target(agent)
                if receiver is None:
                    continue
                direction = unit_vector(target - self.ball["position"])
                self.metrics["pass_attempts"] += 1
                lane_clearance = min(
                    _segment_distance(
                        defender["position"],
                        self.ball["position"],
                        target,
                    )
                    for defender in self.defenders.values()
                )
                valid = lane_clearance > float(self.phase3_config.get("pass_lane_clearance", 0.32))
                if valid:
                    self.metrics["valid_pass_attempts"] += 1
                self.pending_pass = {
                    "passer": agent,
                    "receiver": receiver,
                    "start_x": float(self.ball["position"][0]),
                    "valid": bool(valid),
                    "steps": 0,
                }
                distance = float(np.linalg.norm(target - self.ball["position"]))
                pass_speed = min(
                    float(self.env_config["pass_speed"]),
                    max(1.35, distance / 0.85),
                )
                self._release_ball(agent, direction * pass_speed, "pass")

    def _release_ball(self, agent, velocity, event):
        self.ball["possession"] = None
        self.ball["last_touch"] = agent
        self.ball["shot_in_flight"] = event == "shot"
        self.ball["cooperative_shot"] = bool(
            event == "shot" and self.cooperative_sequence
        )
        self.players[agent]["possesses_ball"] = False
        self.ball["velocity"] = np.asarray(velocity, dtype=np.float64)
        self.ball["position"] = self.players[agent]["position"] + 0.31 * unit_vector(velocity)
        if self.ball_body is not None:
            self.ball_body.position = tuple(self.ball["position"])
            self.ball_body.velocity = tuple(self.ball["velocity"])
        self._event_rewards[event] = self._event_rewards.get(event, 0.0) + 1.0

    def _integrate_abstract(self, desired_players, desired_defenders, dt):
        acceleration = float(self.env_config["player_acceleration"])
        for agent, desired in desired_players.items():
            player = self.players[agent]
            delta = clip_length(desired - player["velocity"], acceleration * dt)
            player["velocity"] += delta
            player["position"] += player["velocity"] * dt
        defender_acceleration = float(
            self.phase3_config.get("defender_acceleration", acceleration)
        )
        for name, desired in desired_defenders.items():
            defender = self.defenders[name]
            delta = clip_length(
                desired - defender["velocity"], defender_acceleration * dt
            )
            defender["velocity"] += delta
            defender["position"] += defender["velocity"] * dt
        if self.ball["possession"] is None:
            self.ball["position"] += self.ball["velocity"] * dt
            drag = float(self.env_config["ball_drag"])
            drag *= float(self.sampled_parameters.get("ball_drag_multiplier", 1.0))
            self.ball["velocity"] *= max(0.0, 1.0 - drag * dt)

    def _integrate_pymunk(self, desired_players, desired_defenders, dt):
        acceleration = float(self.env_config["player_acceleration"])
        for agent, desired in desired_players.items():
            body = self.player_bodies[agent]
            velocity = np.array([body.velocity.x, body.velocity.y])
            body.velocity = tuple(velocity + clip_length(desired - velocity, acceleration * dt))
        defender_acceleration = float(
            self.phase3_config.get("defender_acceleration", acceleration)
        )
        for name, desired in desired_defenders.items():
            body = self.defender_bodies[name]
            velocity = np.array([body.velocity.x, body.velocity.y])
            body.velocity = tuple(
                velocity
                + clip_length(
                    desired - velocity, defender_acceleration * dt
                )
            )
        if self.ball["possession"] is None:
            drag = float(self.env_config["ball_drag"])
            drag *= float(self.sampled_parameters.get("ball_drag_multiplier", 1.0))
            self.ball_body.velocity *= max(0.0, 1.0 - drag * dt)
        else:
            possessor = self.player_bodies[self.ball["possession"]]
            heading = self.players[self.ball["possession"]]["heading"]
            offset = pymunk.Vec2d(0.36, 0.0).rotated(heading)
            self.ball_body.position = possessor.position + offset
            self.ball_body.velocity = possessor.velocity
        self.space.step(dt)
        for agent, body in self.player_bodies.items():
            self.players[agent]["position"] = np.array(body.position)
            self.players[agent]["velocity"] = np.array(body.velocity)
        for name, body in self.defender_bodies.items():
            self.defenders[name]["position"] = np.array(body.position)
            self.defenders[name]["velocity"] = np.array(body.velocity)
        self.ball["position"] = np.array(self.ball_body.position)
        self.ball["velocity"] = np.array(self.ball_body.velocity)

    def _constrain_players(self):
        half_length = float(self.env_config["field_length"]) / 2.0
        half_width = float(self.env_config["field_width"]) / 2.0
        radius = float(self.env_config["player_radius"])
        for collection, bodies in [
            (self.players, self.player_bodies),
            (self.defenders, self.defender_bodies),
        ]:
            for name, record in collection.items():
                clipped = np.clip(
                    record["position"],
                    [-half_length + radius, -half_width + radius],
                    [half_length - radius, half_width - radius],
                )
                if name == "goalie":
                    clipped[0] = np.clip(
                        clipped[0], half_length - 1.20, half_length - radius
                    )
                    clipped[1] = np.clip(
                        clipped[1],
                        -float(self.env_config["goal_width"]),
                        float(self.env_config["goal_width"]),
                    )
                if not np.allclose(clipped, record["position"]):
                    record["velocity"] *= 0.15
                record["position"] = clipped
                if name in bodies:
                    bodies[name].position = tuple(clipped)
                    bodies[name].velocity = tuple(record["velocity"])

    def _update_possession(self):
        if self.ball["possession"] in self.players:
            carrier = self.ball["possession"]
            distance = np.linalg.norm(
                self.players[carrier]["position"] - self.ball["position"]
            )
            if distance < 0.48:
                direction = np.array(
                    [
                        math.cos(self.players[carrier]["heading"]),
                        math.sin(self.players[carrier]["heading"]),
                    ]
                )
                self.ball["position"] = self.players[carrier]["position"] + 0.36 * direction
                self.ball["velocity"] = self.players[carrier]["velocity"].copy()
                self._track_controlled_chain_progress()
                return
            self.ball["possession"] = None
        if np.linalg.norm(self.ball["velocity"]) > float(self.phase3_config.get("capture_speed", 3.0)):
            return
        distances = {
            agent: float(np.linalg.norm(player["position"] - self.ball["position"]))
            for agent, player in self.players.items()
        }
        carrier = min(distances, key=distances.get)
        if distances[carrier] > float(self.phase3_config.get("capture_distance", 0.42)):
            return
        previous = self.last_possessor
        self.ball["possession"] = carrier
        self.ball["last_touch"] = carrier
        self.ball["shot_in_flight"] = False
        self.ball["cooperative_shot"] = False
        self.last_possessor = carrier
        for agent, player in self.players.items():
            player["possesses_ball"] = agent == carrier
        if previous is not None and previous != carrier:
            self.metrics["possession_switches"] += 1
        if previous != carrier:
            self.metrics["possessions"] += 1
        if self.pending_pass is not None and carrier != self.pending_pass["passer"]:
            completed = carrier == self.pending_pass["receiver"] and self.pending_pass["valid"]
            if completed:
                self.metrics["completed_receptions"] += 1
                advance = float(
                    self.players[carrier]["position"][0] - self.pending_pass["start_x"]
                )
                if self._uses_chain_frontier_reward():
                    threat_gain = self._claim_new_chain_progress()
                    credited_advance = threat_gain * float(
                        self.env_config["field_length"]
                    )
                else:
                    threat_gain = max(0.0, advance) / float(
                        self.env_config["field_length"]
                    )
                    credited_advance = advance
                self.metrics["expected_threat_gain"] += threat_gain
                self.pass_chain.append((self.pending_pass["passer"], carrier))
                self.metrics["possession_chain_max"] = max(
                    self.metrics["possession_chain_max"], len(self.pass_chain)
                )
                minimum_advance = float(
                    self.phase3_config.get("minimum_reception_advance", 0.20)
                )
                self.cooperative_sequence = (
                    self.cooperative_sequence or credited_advance >= minimum_advance
                )
                if credited_advance >= minimum_advance:
                    self.metrics["pass_and_advance"] += 1
                    self.cooperative_horizon_remaining = int(
                        self.phase3_config.get("cooperative_horizon_steps", 24)
                    )
                self._event_rewards["controlled_reception"] = 1.0
                self._event_rewards["pass_advance"] = threat_gain
            self.pending_pass = None

    def _defender_clear(self, previous_ball=None):
        if self.ball["possession"] is not None:
            carrier = self.ball["possession"]
            for name, defender in self.defenders.items():
                distance = float(
                    np.linalg.norm(defender["position"] - self.ball["position"])
                )
                tackle_radius = 0.27 if name == "goalie" else 0.25
                if distance < tackle_radius:
                    self.ball["possession"] = None
                    self.players[carrier]["possesses_ball"] = False
                    self.ball["velocity"] = np.array(
                        [-1.25, self.rng.uniform(-0.45, 0.45)]
                    )
                    self.ball["last_touch"] = name
                    self.ball["shot_in_flight"] = False
                    self.metrics["defender_clears"] += 1
                    self.defender_clear_event = True
                    self.metrics["turnovers"] += 1
                    self._event_rewards["turnover"] = 1.0
                    self.metrics["completed_sequences"] += 1
                    self.metrics["sequence_steps_sum"] += (
                        self.step_count - self.sequence_start_step
                    )
                    self.sequence_start_step = self.step_count
                    if self.pending_pass is not None:
                        self.metrics["interceptions"] += 1
                        self.pending_pass = None
                    self._reset_attacking_chain()
                    if self.ball_body is not None:
                        self.ball_body.velocity = tuple(self.ball["velocity"])
                    return
            return
        for name, defender in self.defenders.items():
            distance = np.linalg.norm(defender["position"] - self.ball["position"])
            swept = distance
            if previous_ball is not None:
                swept = _segment_distance(
                    defender["position"], previous_ball, self.ball["position"]
                )
            if self.ball.get("cooperative_shot", False):
                if self.defender_style in {"press", "predictive"}:
                    control_radius = 0.07 if name == "goalie" else 0.14
                else:
                    control_radius = 0.12 if name == "goalie" else 0.18
            else:
                control_radius = 0.50 if name == "goalie" else 0.42
            if min(distance, swept) < control_radius:
                new_clear = self.ball["last_touch"] not in self.defenders
                self.ball["velocity"] = np.array([-3.2, self.rng.uniform(-0.8, 0.8)])
                self.ball["last_touch"] = name
                if new_clear:
                    self.metrics["defender_clears"] += 1
                    self.defender_clear_event = True
                    self.metrics["turnovers"] += 1
                    self._event_rewards["turnover"] = 1.0
                    self.metrics["completed_sequences"] += 1
                    self.metrics["sequence_steps_sum"] += (
                        self.step_count - self.sequence_start_step
                    )
                    self.sequence_start_step = self.step_count
                    if self.pending_pass is not None:
                        self.metrics["interceptions"] += 1
                        self.pending_pass = None
                    self._reset_attacking_chain()
                if self.ball_body is not None:
                    self.ball_body.velocity = tuple(self.ball["velocity"])
                break

    def _potential_components(self):
        progress = float(self.ball["position"][0]) / float(
            self.env_config["field_length"]
        )
        possession = 0.20 if self.ball["possession"] in self.players else 0.0
        spread = 0.0
        if len(self.active_agents) > 1:
            y_values = [self.players[agent]["position"][1] for agent in self.active_agents]
            spread = min(1.0, float(np.std(y_values))) * 0.05
        return {
            "ball_progress": progress,
            "team_possession": possession,
            "support_quality": spread,
        }

    def _potential(self):
        return sum(self._potential_components().values())

    def _terminal(self):
        half_length = float(self.env_config["field_length"]) / 2.0
        half_width = float(self.env_config["field_width"]) / 2.0
        goal_width = float(self.env_config["goal_width"])
        x_value, y_value = self.ball["position"]
        if x_value >= half_length and abs(y_value) <= goal_width / 2.0:
            elapsed = (
                self.step_count
                * float(self.env_config["dt"])
                * int(self.env_config["macro_action_repeat"])
            )
            if elapsed < float(self.phase3_config.get("minimum_goal_seconds", 8.0)):
                self.ball["position"][0] = half_length - 0.2
                self.ball["velocity"] = np.array([-1.5, self.rng.uniform(-0.3, 0.3)])
                if self.ball_body is not None:
                    self.ball_body.position = tuple(self.ball["position"])
                    self.ball_body.velocity = tuple(self.ball["velocity"])
                return None
            if not self.cooperative_sequence and not self.ball.get("shot_in_flight", False):
                self.ball["position"][0] = half_length - 0.2
                self.ball["velocity"] = np.array([-1.5, self.rng.uniform(-0.3, 0.3)])
                if self.ball_body is not None:
                    self.ball_body.position = tuple(self.ball["position"])
                    self.ball_body.velocity = tuple(self.ball["velocity"])
                return None
            if self.scenario["pass_required"] and not self.cooperative_sequence:
                self.ball["position"][0] = half_length - 0.2
                self.ball["velocity"] = np.array([-1.4, 0.0])
                if self.ball_body is not None:
                    self.ball_body.position = tuple(self.ball["position"])
                    self.ball_body.velocity = tuple(self.ball["velocity"])
                return None
            return "goal"
        if abs(y_value) > half_width or x_value < -half_length or x_value > half_length + 0.55:
            return "out_of_bounds"
        if self.step_count >= self._maximum_steps():
            return "timeout"
        return None

    def _restart_match(self):
        score = self.match_score
        restarts = self.match_restart_count + 1
        self.last_possessor = None
        self.last_nearest_agent = None
        self.sequence_start_step = self.step_count
        self._initialize_entities()
        self._initialize_physics()
        latency = int(self.sampled_parameters.get("action_latency", 0))
        self.action_queues = {
            agent: deque([6] * latency) for agent in self.active_agents
        }
        self._reset_delay_queues()
        self._update_possession()
        self._reset_chain_progress()
        self._last_potential_components = self._potential_components()
        self._last_potential = sum(self._last_potential_components.values())
        self.match_score = score
        self.match_restart_count = restarts
        self.restart_pause_remaining = int(
            self.phase3_config.get("restart_pause_steps", 2)
        )
        self.metrics["match_restarts"] = restarts
        self.metrics["match_score"] = score
        self.cooperative_sequence = False
        self.cooperative_horizon_remaining = 0
        self.pending_pass = None
        self.pass_chain = []

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        expected = set(self.active_agents)
        if set(actions) != expected:
            raise ValueError("Actions must cover exactly the active Phase 3 attackers")
        if self.restart_pause_remaining > 0:
            for agent, action in actions.items():
                if not self.action_space(agent).contains(int(action)):
                    raise ValueError("Invalid action index for " + agent)
            self.restart_pause_remaining -= 1
            self.step_count += 1
            self.metrics["active_steps"] += 1
            self.metrics["episode_steps"] = self.step_count
            truncated = self.step_count >= self._maximum_steps()
            reward = -float(self.reward_config.get("step_penalty", 0.002))
            self.metrics["reward_time"] += reward
            observations = self._observations()
            rewards = {agent: reward for agent in self.active_agents}
            terminations = {agent: False for agent in self.active_agents}
            truncations = {agent: truncated for agent in self.active_agents}
            infos = {agent: self._info(agent) for agent in self.active_agents}
            if truncated:
                self.termination_reason = "timeout"
                self.metrics["terminal_reason"] = "timeout"
                for info in infos.values():
                    info["episode_metrics"] = copy.deepcopy(self._final_metrics())
                self.agents = []
            return observations, rewards, terminations, truncations, infos
        self._event_rewards = {}
        self.defender_clear_event = False
        valid_actions = {}
        for agent in self.active_agents:
            requested = int(actions[agent])
            if not self.action_space(agent).contains(requested):
                raise ValueError("Invalid action index for " + agent)
            self.metrics["requested_" + ACTION_NAMES[requested]] += 1
            mask = self.action_mask(agent)
            if mask[requested] < 0.5:
                self.metrics["invalid_action_requests"] += 1
                self.metrics["masked_" + ACTION_NAMES[requested]] += 1
                requested = 6
            queue = self.action_queues[agent]
            if queue:
                queue.append(requested)
                action = int(queue.popleft())
            else:
                action = requested
            player = self.players[agent]
            if action != player["executed_action"]:
                self.metrics["action_switches"] += 1
            player["previous_action"] = player["executed_action"]
            player["executed_action"] = action
            valid_actions[agent] = action
        packet_loss = float(self.sampled_parameters.get("packet_loss", 0.0))
        for agent in self.active_agents:
            player = self.players[agent]
            player["message"][2] = min(1.0, float(player["message"][2]) + 0.05)
            player["message"][3] = 0.0
            outgoing = None
            if self.rng.random() >= packet_loss:
                outgoing = np.asarray(
                    [
                        player["executed_action"] / max(1, self.action_size - 1),
                        float(self.ball["possession"] == agent),
                        0.0,
                        1.0,
                    ],
                    dtype=np.float32,
                )
            queue = self.message_queues[agent]
            if queue:
                queue.append(outgoing)
                delivered = queue.popleft()
            else:
                delivered = outgoing
            if delivered is not None:
                player["message"] = delivered
        self._execute_kicks(valid_actions)
        repeat = int(self.env_config["macro_action_repeat"])
        dt = float(self.env_config["dt"])
        for _ in range(repeat):
            desired_players = {}
            for agent, action in valid_actions.items():
                desired, _ = self._desired_attacker_motion(agent, action)
                desired_players[agent] = desired
                self._turn_toward(
                    self.players[agent],
                    desired,
                    float(self.env_config["player_max_turn_rate"]),
                    dt,
                )
            defender_speed = float(self.phase3_config.get("defender_speed", 1.15))
            desired_defenders = {}
            for name in self.active_defenders:
                direction = unit_vector(
                    self._defender_target(name) - self.defenders[name]["position"]
                )
                desired_defenders[name] = direction * defender_speed
                self._turn_toward(
                    self.defenders[name],
                    direction,
                    float(self.phase3_config.get("defender_turn_rate", 2.5)),
                    dt,
                )
            previous_ball = self.ball["position"].copy()
            if self.simulator == "abstract":
                self._integrate_abstract(desired_players, desired_defenders, dt)
            else:
                substeps = int(self.config["transfer_environment"].get("substeps", 5))
                for _ in range(substeps):
                    self._integrate_pymunk(
                        desired_players, desired_defenders, dt / substeps
                    )
            self._constrain_players()
            self._update_possession()
            self._defender_clear(previous_ball)
            if self.pending_pass is not None:
                self.pending_pass["steps"] += 1
                if self.pending_pass["steps"] > int(
                    self.phase3_config.get("reception_window_steps", 18)
                ):
                    self.pending_pass = None
        self.step_count += 1
        if self.cooperative_horizon_remaining > 0:
            self.cooperative_horizon_remaining -= 1
            if self.cooperative_horizon_remaining == 0:
                self.cooperative_sequence = False
        self.metrics["active_steps"] += 1
        self.metrics["episode_steps"] = self.step_count
        if self.ball["possession"] in self.players:
            self.metrics["possession_steps"] += 1
        nearest = min(
            self.active_agents,
            key=lambda name: np.linalg.norm(
                self.players[name]["position"] - self.ball["position"]
            ),
        )
        self.metrics["nearest_steps_" + nearest] += 1
        if self.last_nearest_agent is not None and nearest != self.last_nearest_agent:
            self.metrics["role_switches"] += 1
        self.last_nearest_agent = nearest
        pairwise = []
        step_collisions = 0
        for first_index, first in enumerate(self.active_agents):
            for second in self.active_agents[first_index + 1 :]:
                distance = float(
                    np.linalg.norm(
                        self.players[first]["position"]
                        - self.players[second]["position"]
                    )
                )
                pairwise.append(distance)
                if distance < 2.0 * float(self.env_config["player_radius"]):
                    self.metrics["collisions"] += 1
                    step_collisions += 1
        if pairwise:
            self.metrics["pairwise_separation_sum"] += float(np.mean(pairwise))
        y_positions = [
            float(self.players[agent]["position"][1]) for agent in self.active_agents
        ]
        self.metrics["field_width_sum"] += max(y_positions) - min(y_positions)
        for agent in self.active_agents:
            self.motion_trails[agent].append(self.players[agent]["position"].copy())
        if sum(valid_actions[agent] == 0 for agent in self.active_agents) > 1:
            self.metrics["redundant_chase_steps"] += 1
        current_components = self._potential_components()
        potential_weight = float(self.reward_config.get("phase3_progress", 0.7))
        progress_reward = potential_weight * (
            current_components["ball_progress"]
            - self._last_potential_components["ball_progress"]
        )
        possession_reward = potential_weight * (
            current_components["team_possession"]
            - self._last_potential_components["team_possession"]
        )
        support_reward = potential_weight * (
            current_components["support_quality"]
            - self._last_potential_components["support_quality"]
        )
        controlled_reward = float(
            self.reward_config.get("controlled_reception", 0.35)
        ) * self._event_rewards.get("controlled_reception", 0.0)
        pass_advance_reward = float(
            self.reward_config.get("pass_advance", 0.30)
        ) * self._event_rewards.get("pass_advance", 0.0)
        turnover_reward = float(
            self.reward_config.get("turnover_penalty", -0.50)
        ) * self._event_rewards.get("turnover", 0.0)
        collision_reward = float(
            self.reward_config.get("collision_penalty", -0.02)
        ) * step_collisions
        time_reward = -float(self.reward_config.get("step_penalty", 0.002))
        reward = (
            progress_reward
            + possession_reward
            + support_reward
            + controlled_reward
            + pass_advance_reward
            + turnover_reward
            + collision_reward
            + time_reward
        )
        self.metrics["reward_ball_progress"] += progress_reward
        self.metrics["reward_team_possession"] += possession_reward
        self.metrics["reward_support_quality"] += support_reward
        self.metrics["reward_controlled_pass"] += controlled_reward
        self.metrics["reward_pass_and_advance"] += pass_advance_reward
        self.metrics["reward_chain_progress"] += pass_advance_reward
        self.metrics["reward_turnover"] += turnover_reward
        self.metrics["reward_collision"] += collision_reward
        self.metrics["reward_time"] += time_reward
        self._last_potential_components = current_components
        self._last_potential = sum(current_components.values())
        reason = self._terminal()
        match_mode = bool(self.phase3_config.get("match_mode", False))
        if reason == "goal":
            self.metrics["goal"] += 1
            self.metrics["shots_on_target"] += int(self.ball.get("shot_in_flight", False))
            self.metrics["completed_sequences"] += 1
            self.metrics["sequence_steps_sum"] += (
                self.step_count - self.sequence_start_step
            )
            self.sequence_start_step = self.step_count
            self.match_score += 1
            self.metrics["match_score"] = self.match_score
            if self.cooperative_sequence:
                self.metrics["cooperative_success"] = 1
                self.metrics["pass_to_goal"] += 1
                pass_goal_reward = float(
                    self.reward_config.get("pass_to_goal", 1.0)
                )
                reward += pass_goal_reward
                self.metrics["reward_pass_and_goal"] += pass_goal_reward
            goal_reward = float(
                self.reward_config.get("goal_reward", self.config["reward"]["goal"])
            )
            reward += goal_reward
            self.metrics["reward_goal"] += goal_reward
            if self.metrics["time_to_score"] is None:
                self.metrics["time_to_score"] = (
                    self.step_count
                    * float(self.env_config["dt"])
                    * int(self.env_config["macro_action_repeat"])
                )
            if match_mode and self.step_count < self._maximum_steps():
                self._restart_match()
                reason = None
        elif reason == "out_of_bounds":
            self.metrics["out_of_bounds"] += 1
            self.metrics["completed_sequences"] += 1
            self.metrics["sequence_steps_sum"] += (
                self.step_count - self.sequence_start_step
            )
            self.sequence_start_step = self.step_count
            out_of_bounds_reward = float(
                self.reward_config.get(
                    "out_of_bounds_penalty", self.config["reward"]["out_of_bounds"]
                )
            )
            reward += out_of_bounds_reward
            self.metrics["reward_out_of_bounds"] += out_of_bounds_reward
            if match_mode and self.step_count < self._maximum_steps():
                self._restart_match()
                reason = None
        elif self.defender_clear_event and match_mode and self.step_count < self._maximum_steps():
            self._restart_match()
        self.termination_reason = reason
        terminated = reason in {"goal", "out_of_bounds"}
        truncated = reason == "timeout"
        observations = self._observations() if not (terminated or truncated) else {
            agent: self.observe(agent) for agent in self.active_agents
        }
        rewards = {agent: float(reward) for agent in self.active_agents}
        terminations = {agent: bool(terminated) for agent in self.active_agents}
        truncations = {agent: bool(truncated) for agent in self.active_agents}
        infos = {agent: self._info(agent) for agent in self.active_agents}
        if terminated or truncated:
            self.metrics["terminal_reason"] = reason
            for info in infos.values():
                info["episode_metrics"] = copy.deepcopy(self._final_metrics())
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _final_metrics(self):
        result = copy.deepcopy(self.metrics)
        steps = max(1, int(result["active_steps"]))
        result["redundant_chase_fraction"] = result["redundant_chase_steps"] / steps
        decisions = max(1, steps * len(self.active_agents))
        result["invalid_action_fraction"] = result["invalid_action_requests"] / decisions
        result["success"] = int(result["goal"] > 0)
        result["match_score"] = self.match_score
        result["goals_per_match"] = result["goal"]
        result["scoring_rate_per_possession"] = result["goal"] / max(
            1, result["possessions"]
        )
        result["mean_possession_seconds"] = (
            result["possession_steps"]
            * float(self.env_config["dt"])
            * int(self.env_config["macro_action_repeat"])
            / max(1, result["possessions"])
        )
        result["pass_completion_rate"] = result["completed_receptions"] / max(
            1, result["valid_pass_attempts"]
        )
        result["pass_and_advance_rate"] = result["pass_and_advance"] / max(
            1, result["valid_pass_attempts"]
        )
        result["pass_and_goal_rate"] = result["pass_to_goal"] / max(
            1, result["valid_pass_attempts"]
        )
        result["shot_conversion"] = result["goal"] / max(1, result["shots"])
        result["mean_pairwise_attacker_separation"] = (
            result["pairwise_separation_sum"] / steps
        )
        result["field_width_utilization"] = (
            result["field_width_sum"] / steps / float(self.env_config["field_width"])
        )
        result["mean_sequence_steps"] = result["sequence_steps_sum"] / max(
            1, result["completed_sequences"]
        )
        result["collision_counting_semantics"] = (
            "macro_action_decisions_with_attacker_overlap"
        )
        return result

    def metrics_snapshot(self):
        result = self._final_metrics()
        result["terminal_reason"] = self.termination_reason
        return result

    def _info(self, agent):
        return {
            "seed": self.seed_value,
            "scenario": self.scenario["name"],
            "simulator": self.simulator,
            "selected_profile": self.selected_profile,
            "defender_style": self.defender_style,
            "scenario_group": self.scenario_group,
            "active_agent_mask": [
                float(name in self.active_agents) for name in self.possible_agents
            ],
            "action_mask": self.action_mask(agent),
            "termination_reason": self.termination_reason,
            "match_restart": self.match_restart_count,
        }

    def render(self):
        width = int(self.config["video"].get("width", 1280))
        height = int(self.config["video"].get("height", 720))
        image = Image.new("RGB", (width, height), (17, 91, 51))
        draw = ImageDraw.Draw(image)
        margin_x = int(width * 0.065)
        margin_y = int(height * 0.09)
        field_width = width - 2 * margin_x
        field_height = height - 2 * margin_y
        draw.rounded_rectangle(
            (margin_x, margin_y, margin_x + field_width, margin_y + field_height),
            radius=14,
            outline=(238, 244, 231),
            width=max(2, width // 420),
        )
        draw.line(
            (width // 2, margin_y, width // 2, margin_y + field_height),
            fill=(238, 244, 231),
            width=max(2, width // 500),
        )
        draw.ellipse(
            (
                width // 2 - field_height * 0.12,
                height // 2 - field_height * 0.12,
                width // 2 + field_height * 0.12,
                height // 2 + field_height * 0.12,
            ),
            outline=(238, 244, 231),
            width=max(2, width // 500),
        )
        half_length = float(self.env_config["field_length"]) / 2.0
        half_width = float(self.env_config["field_width"]) / 2.0

        def pixel(position):
            x_value = margin_x + (position[0] + half_length) / (2 * half_length) * field_width
            y_value = margin_y + (half_width - position[1]) / (2 * half_width) * field_height
            return int(x_value), int(y_value)

        robot_radius = max(10, int(field_height * 0.035))
        for index, agent in enumerate(self.active_agents):
            center = pixel(self.players[agent]["position"])
            color = [(41, 132, 255), (70, 200, 255), (122, 96, 255)][index]
            trail = [pixel(position) for position in self.motion_trails.get(agent, [])]
            if len(trail) > 1:
                draw.line(trail, fill=tuple(int(channel * 0.75) for channel in color), width=3)
            self._draw_robot(draw, center, robot_radius, color, index + 1, True)
            heading = float(self.players[agent]["heading"])
            heading_end = (
                int(center[0] + math.cos(heading) * robot_radius * 1.35),
                int(center[1] - math.sin(heading) * robot_radius * 1.35),
            )
            draw.line((center, heading_end), fill=(250, 250, 245), width=3)
            if self.ball["possession"] == agent:
                draw.ellipse(
                    (
                        center[0] - robot_radius - 5,
                        center[1] - robot_radius - 5,
                        center[0] + robot_radius + 5,
                        center[1] + robot_radius + 5,
                    ),
                    outline=(255, 232, 92),
                    width=4,
                )
            nearest = min(
                self.active_agents,
                key=lambda name: np.linalg.norm(
                    self.players[name]["position"] - self.ball["position"]
                ),
            )
            if self.ball["possession"] == agent:
                role = "carrier"
            elif nearest == agent:
                role = "chaser"
            else:
                role = "support"
            action_name = ACTION_NAMES[int(self.players[agent]["executed_action"])]
            draw.text(
                (center[0] - robot_radius, center[1] + robot_radius + 4),
                role + " | " + action_name,
                fill=(245, 247, 240),
                font=ImageFont.load_default(),
            )
        for index, name in enumerate(self.active_defenders):
            center = pixel(self.defenders[name]["position"])
            color = (246, 84, 73) if name != "goalie" else (255, 181, 48)
            self._draw_robot(draw, center, robot_radius, color, index + 1, False)
        ball_center = pixel(self.ball["position"])
        ball_radius = max(6, int(robot_radius * 0.42))
        draw.ellipse(
            (
                ball_center[0] - ball_radius,
                ball_center[1] - ball_radius,
                ball_center[0] + ball_radius,
                ball_center[1] + ball_radius,
            ),
            fill=(248, 245, 235),
            outline=(30, 30, 30),
            width=2,
        )
        if self.pending_pass is not None:
            receiver = self.pending_pass["receiver"]
            if receiver in self.players:
                draw.line(
                    (ball_center, pixel(self.players[receiver]["position"])),
                    fill=(255, 232, 92),
                    width=3,
                )
        elapsed = (
            self.step_count
            * float(self.env_config["dt"])
            * int(self.env_config["macro_action_repeat"])
        )
        overlay = (
            f"{self.scenario['name']}  |  {self.simulator.upper()}  |  "
            f"{self.defender_style}  |  {self.selected_profile}  |  "
            f"{self.config['experiment']['name']}  |  t={elapsed:.1f}s  |  "
            f"score={self.match_score}"
        )
        draw.rounded_rectangle((18, 14, width - 18, 53), radius=9, fill=(9, 31, 25))
        draw.text((31, 25), overlay, fill=(245, 247, 240), font=ImageFont.load_default())
        return np.asarray(image)

    @staticmethod
    def _draw_robot(draw, center, radius, color, number, attacker):
        x_value, y_value = center
        draw.ellipse(
            (x_value - radius, y_value - radius, x_value + radius, y_value + radius),
            fill=color,
            outline=(19, 26, 31),
            width=max(2, radius // 7),
        )
        visor = radius * 0.58
        draw.rounded_rectangle(
            (
                x_value - visor,
                y_value - radius * 0.40,
                x_value + visor,
                y_value + radius * 0.05,
            ),
            radius=max(2, radius // 6),
            fill=(18, 33, 44),
        )
        accent = (218, 246, 255) if attacker else (255, 232, 196)
        draw.ellipse(
            (
                x_value - radius * 0.48,
                y_value - radius * 0.28,
                x_value - radius * 0.18,
                y_value,
            ),
            fill=accent,
        )
        draw.ellipse(
            (
                x_value + radius * 0.18,
                y_value - radius * 0.28,
                x_value + radius * 0.48,
                y_value,
            ),
            fill=accent,
        )
        draw.text((x_value - 3, y_value + radius * 0.22), str(number), fill=(255, 255, 255))

    def close(self):
        self.space = None
        self.player_bodies = {}
        self.defender_bodies = {}
        self.ball_body = None


def make_phase3_environment(config, simulator="abstract", **kwargs):
    return Phase3SoccerEnv(config, simulator=simulator, **kwargs)


def run_stage_r_reward_invariants(config):
    """Exercise Stage-R reward semantics and return a machine-readable result."""
    if int(config.get("phase3", {}).get("reward_schema_version", 1)) != 2:
        raise ValueError("Stage-R reward invariants require reward schema version 2")
    if float(config["phase3_reward"].get("controlled_reception", -1.0)) != 0.0:
        raise ValueError("Stage-R controlled-reception reward must be zero")
    test_config = copy.deepcopy(config)
    test_config["phase3"]["match_mode"] = False
    test_config["phase3"]["minimum_goal_seconds"] = 0.0
    env = make_phase3_environment(
        test_config,
        simulator="abstract",
        scenario="phase3_2v2_pass_required",
        defender_style="lane_block",
    )

    def receive(passer, receiver, receiver_x):
        env.ball["possession"] = None
        env.last_possessor = passer
        env.players[receiver]["position"][0] = receiver_x
        env.ball["position"] = env.players[receiver]["position"].copy()
        env.ball["velocity"] = np.zeros(2)
        env.pending_pass = {
            "passer": passer,
            "receiver": receiver,
            "start_x": float(env.players[passer]["position"][0]),
            "valid": True,
            "steps": 1,
        }
        env._event_rewards = {}
        before_receptions = env.metrics["completed_receptions"]
        before_progress = env.metrics["new_chain_progress"]
        env._update_possession()
        return {
            "reception_increment": (
                env.metrics["completed_receptions"] - before_receptions
            ),
            "new_progress": env.metrics["new_chain_progress"] - before_progress,
            "controlled_reception_event": env._event_rewards.get(
                "controlled_reception", 0.0
            ),
            "pass_advance_event": env._event_rewards.get("pass_advance", 0.0),
        }

    try:
        env.reset(seed=390000)
        first, second = env.active_agents
        start_x = float(env.players[first]["position"][0])
        env.chain_progress_high_water = start_x / float(
            env.env_config["field_length"]
        )
        env.metrics["chain_progress_high_water"] = env.chain_progress_high_water
        forward_x = start_x + 0.60
        forward = receive(first, second, forward_x)
        backward = receive(second, first, start_x + 0.10)
        revisit = receive(first, second, forward_x)
        farther_x = forward_x + 0.40
        new_frontier = receive(first, second, farther_x)
        progress_before_turnover = env.chain_progress_high_water
        env.ball["position"][0] = start_x - 0.20
        env._reset_attacking_chain()
        turnover_reset = (
            env.chain_progress_high_water < progress_before_turnover
            and env.pending_pass is None
            and not env.cooperative_sequence
        )
        env._restart_match()
        restart_reset = (
            abs(env.chain_progress_high_water - env._chain_progress_value()) < 1e-12
        )
        checks = {
            "controlled_reception_reward_zero": {
                "observed": float(config["phase3_reward"]["controlled_reception"])
                * forward["controlled_reception_event"],
                "passed": (
                    float(config["phase3_reward"]["controlled_reception"])
                    * forward["controlled_reception_event"]
                    == 0.0
                ),
            },
            "first_forward_frontier_positive": {
                "observed": forward["new_progress"],
                "passed": forward["new_progress"] > 0.0
                and forward["pass_advance_event"] > 0.0,
            },
            "backward_reception_no_progress": {
                "observed": backward["new_progress"],
                "passed": backward["reception_increment"] == 1
                and backward["new_progress"] == 0.0,
            },
            "aba_no_repeated_progress": {
                "observed": backward["new_progress"],
                "passed": backward["new_progress"] == 0.0,
            },
            "revisited_frontier_no_progress": {
                "observed": revisit["new_progress"],
                "passed": revisit["new_progress"] == 0.0,
            },
            "new_frontier_incremental_only": {
                "observed": new_frontier["new_progress"],
                "expected": 0.40 / float(env.env_config["field_length"]),
                "passed": abs(
                    new_frontier["new_progress"]
                    - 0.40 / float(env.env_config["field_length"])
                )
                < 1e-9,
            },
            "turnover_resets_chain": {
                "observed": turnover_reset,
                "passed": turnover_reset,
            },
            "restart_resets_chain": {
                "observed": restart_reset,
                "passed": restart_reset,
            },
            "pass_without_goal_has_no_pass_to_goal": {
                "observed": env.metrics["pass_to_goal"],
                "passed": env.metrics["pass_to_goal"] == 0,
            },
            "circulation_reward_does_not_scale": {
                "observed": (
                    forward["new_progress"]
                    + backward["new_progress"]
                    + revisit["new_progress"]
                ),
                "expected": forward["new_progress"],
                "passed": (
                    forward["new_progress"]
                    + backward["new_progress"]
                    + revisit["new_progress"]
                    == forward["new_progress"]
                ),
            },
        }
    finally:
        env.close()

    goal_env = make_phase3_environment(
        test_config,
        simulator="abstract",
        scenario="phase3_2v2_pass_required",
        defender_style="lane_block",
    )
    try:
        goal_env.reset(seed=390001)
        goal_env.cooperative_sequence = True
        goal_env.cooperative_horizon_remaining = 10
        goal_env.ball["possession"] = None
        goal_env.ball["shot_in_flight"] = True
        goal_env.ball["cooperative_shot"] = True
        goal_env.ball["position"] = np.array(
            [test_config["environment"]["field_length"] / 2 + 0.01, 0.0]
        )
        for defender in goal_env.defenders.values():
            defender["position"] = np.array([-4.0, 2.5])
        _, _, _, _, infos = goal_env.step(
            {agent: 6 for agent in goal_env.active_agents}
        )
        metrics = next(iter(infos.values()))["episode_metrics"]
        checks["pass_to_goal_fires_once"] = {
            "observed": metrics["pass_to_goal"],
            "passed": metrics["pass_to_goal"] == 1
            and metrics["reward_pass_and_goal"]
            == float(test_config["phase3_reward"]["pass_to_goal"]),
        }
    finally:
        goal_env.close()
    return {
        "schema_version": 1,
        "reward_schema_version": 2,
        "scientific_status": "deterministic_invariant_check",
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }
def phase3_baseline_actions(env, method, memory=None):
    """Return legal actions for calibration baselines without using privileged state."""
    memory = memory if memory is not None else {}
    actions = {}
    if method == "random":
        for agent in env.active_agents:
            legal = np.flatnonzero(env.action_mask(agent) > 0.5)
            actions[agent] = int(env.rng.choice(legal))
        return actions
    if method == "double_chase":
        for agent in env.active_agents:
            actions[agent] = 3 if env.ball["possession"] == agent else 0
        return actions
    if method != "role_based":
        raise ValueError("Unknown Phase 3 baseline method: " + str(method))
    if env.pending_pass is not None:
        receiver = env.pending_pass["receiver"]
        for agent in env.active_agents:
            actions[agent] = 0 if agent == receiver else 5
        return actions
    carrier = env.ball["possession"]
    if carrier not in env.active_agents:
        carrier = min(
            env.active_agents,
            key=lambda name: np.linalg.norm(
                env.players[name]["position"] - env.ball["position"]
            ),
        )
    memory["carrier"] = carrier
    for agent in env.active_agents:
        if agent == carrier:
            goal = np.array([float(env.env_config["field_length"]) / 2.0, 0.0])
            blockers = [
                _segment_distance(
                    record["position"], env.players[agent]["position"], goal
                )
                for record in env.defenders.values()
            ]
            direct_lane = min(blockers) > 0.52
            if env.ball["possession"] == agent:
                next_elapsed = (
                    (env.step_count + 1)
                    * float(env.env_config["dt"])
                    * int(env.env_config["macro_action_repeat"])
                )
                shot_ready = next_elapsed >= float(
                    env.phase3_config.get("minimum_goal_seconds", 8.0)
                )
                if (direct_lane or env.cooperative_sequence) and shot_ready:
                    actions[agent] = 3
                elif env.cooperative_sequence:
                    actions[agent] = 1 if env.step_count % 2 == 0 else 2
                else:
                    actions[agent] = 4
            else:
                actions[agent] = 0
        else:
            actions[agent] = 5
    if 4 in actions.values():
        for agent in env.active_agents:
            if actions[agent] != 4:
                actions[agent] = 0
    return actions


def run_phase3_baseline_episode(config, simulator, scenario, method, seed, defender_style="mixed"):
    env = make_phase3_environment(
        config,
        simulator=simulator,
        scenario=scenario,
        defender_style=defender_style,
    )
    memory = {}
    team_return = 0.0
    metrics = {}
    try:
        env.reset(seed=seed)
        while env.agents:
            actions = phase3_baseline_actions(env, method, memory)
            _, rewards, _, _, infos = env.step(actions)
            team_return += next(iter(rewards.values()))
            if not env.agents:
                metrics = next(iter(infos.values()))["episode_metrics"]
    finally:
        env.close()
    row = dict(metrics)
    row.update(
        {
            "method": method,
            "simulator": simulator,
            "scenario": scenario,
            "seed": int(seed),
            "team_return": float(team_return),
        }
    )
    return row


def summarize_phase3_calibration(rows, episodes_per_cell):
    groups = {}
    for row in rows:
        key = (row["method"], row["simulator"], row["scenario"])
        groups.setdefault(key, []).append(row)
    cells = {}
    for key, records in sorted(groups.items()):
        successes = [float(row["success"]) for row in records]
        cooperative = [float(row["cooperative_success"]) for row in records]
        times = [
            float(row["time_to_score"])
            for row in records
            if row.get("time_to_score") is not None
        ]
        cell_name = "__".join(key)
        cells[cell_name] = {
            "method": key[0],
            "simulator": key[1],
            "scenario": key[2],
            "episodes": len(records),
            "success_rate": float(np.mean(successes)),
            "cooperative_success_rate": float(np.mean(cooperative)),
            "median_success_time": float(np.median(times)) if times else None,
            "mean_return": float(np.mean([row["team_return"] for row in records])),
            "finite": bool(
                np.isfinite(
                    [
                        *successes,
                        *cooperative,
                        *[row["team_return"] for row in records],
                    ]
                ).all()
            ),
            "complete": len(records) >= int(episodes_per_cell),
        }
    return cells


def phase3_calibration_gate(cells, config, smoke=False):
    gate = config.get("phase3", {}).get("calibration", {})
    required_episodes = int(
        gate.get("smoke_episodes" if smoke else "episodes_per_cell", 20 if smoke else 100)
    )
    scenarios = ["phase3_2v2_open", "phase3_2v2_pass_required"]
    required = [
        (method, simulator, scenario)
        for method in ["random", "double_chase", "role_based"]
        for simulator in ["abstract", "pymunk"]
        for scenario in scenarios
    ]
    checks = {}

    def add(name, passed, observed, criterion):
        checks[name] = {
            "passed": bool(passed),
            "observed": observed,
            "criterion": criterion,
        }

    missing = []
    incomplete = []
    nonfinite = []
    for key in required:
        cell_name = "__".join(key)
        if cell_name not in cells:
            missing.append(cell_name)
            continue
        if cells[cell_name]["episodes"] < required_episodes:
            incomplete.append(cell_name)
        if not cells[cell_name]["finite"]:
            nonfinite.append(cell_name)
    add("required_cells", not missing, missing, "all required method/simulator/scenario cells exist")
    add(
        "episode_count",
        not incomplete,
        incomplete,
        f"every required cell has at least {required_episodes} episodes",
    )
    add("finite_metrics", not nonfinite, nonfinite, "all required cell metrics are finite")
    random_pymunk = cells.get("random__pymunk__phase3_2v2_open", {}).get("success_rate")
    double_pymunk = cells.get("double_chase__pymunk__phase3_2v2_open", {}).get(
        "success_rate"
    )
    role_pymunk = cells.get("role_based__pymunk__phase3_2v2_open", {}).get("success_rate")
    direct_pass = cells.get(
        "double_chase__pymunk__phase3_2v2_pass_required", {}
    ).get("success_rate")
    add(
        "random_pymunk_hard",
        random_pymunk is not None and random_pymunk <= float(gate.get("random_max", 0.05)),
        random_pymunk,
        "random Pymunk open success <= 0.05",
    )
    add(
        "double_chase_pymunk_hard",
        double_pymunk is not None and double_pymunk <= float(gate.get("double_chase_max", 0.20)),
        double_pymunk,
        "double-chase Pymunk open success <= 0.20",
    )
    role_low = float(gate.get("role_min", 0.20))
    role_high = float(gate.get("role_max", 0.55))
    add(
        "role_pymunk_band",
        role_pymunk is not None and role_low <= role_pymunk <= role_high,
        role_pymunk,
        f"role Pymunk open success in [{role_low:.2f}, {role_high:.2f}]",
    )
    margin = float(gate.get("role_random_margin", 0.15))
    add(
        "role_advantage",
        role_pymunk is not None
        and random_pymunk is not None
        and role_pymunk >= random_pymunk + margin,
        {"role": role_pymunk, "random": random_pymunk},
        f"role Pymunk success >= random + {margin:.2f}",
    )
    add(
        "pass_required_direct_shot_blocked",
        direct_pass is not None and direct_pass <= float(gate.get("direct_shot_max", 0.05)),
        direct_pass,
        "double-chase direct-shot success in pass-required Pymunk <= 0.05",
    )
    medians = {
        name: cell["median_success_time"]
        for name, cell in cells.items()
        if name in {"__".join(key) for key in required}
        and cell["median_success_time"] is not None
        and cell["success_rate"] > 0
    }
    minimum_time = float(gate.get("minimum_median_success_time", 8.0))
    add(
        "nontrivial_duration",
        bool(medians) and min(medians.values()) >= minimum_time,
        medians,
        f"every successful required cell median time >= {minimum_time:.1f}s",
    )
    maximum = float(gate.get("saturation_max", 0.85))
    saturated = {
        name: cell["success_rate"]
        for name, cell in cells.items()
        if name in {"__".join(key) for key in required} and cell["success_rate"] > maximum
    }
    add(
        "no_saturation",
        not saturated,
        saturated,
        f"no required cell success > {maximum:.2f}",
    )
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": 1,
        "smoke": bool(smoke),
        "training_authorized": bool(passed and not smoke),
        "passed": passed,
        "checks": checks,
    }


def run_phase3_calibration(config, output_dir, episodes=None, smoke=False, seed_base=310000):
    gate = config.get("phase3", {}).get("calibration", {})
    default_episodes = gate.get("smoke_episodes" if smoke else "episodes_per_cell", 20 if smoke else 100)
    episodes = int(episodes or default_episodes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cell_index = 0
    for method in ["random", "double_chase", "role_based"]:
        for simulator in ["abstract", "pymunk"]:
            for scenario in ["phase3_2v2_open", "phase3_2v2_pass_required"]:
                for episode in range(episodes):
                    rows.append(
                        run_phase3_baseline_episode(
                            config,
                            simulator,
                            scenario,
                            method,
                            seed_base + cell_index * 1000 + episode,
                        )
                    )
                cell_index += 1
    cells = summarize_phase3_calibration(rows, episodes)
    result = phase3_calibration_gate(cells, config, smoke=smoke)
    result["cells"] = cells
    result["episodes_per_cell"] = episodes
    result["seed_base"] = seed_base
    write_json(output_dir / "calibration_summary.json", result)
    Path(output_dir / "calibration_episodes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return result


class Phase3EnvironmentBatch:
    """Authoritative multi-lane executor with batched observations and timing counters."""

    def __init__(self, config, num_envs, simulator="abstract", scenario=None, seed_base=0):
        self.environments = [
            make_phase3_environment(config, simulator=simulator, scenario=scenario)
            for _ in range(int(num_envs))
        ]
        self.num_envs = len(self.environments)
        self.seed_base = int(seed_base)
        self.reset_seconds = 0.0
        self.step_seconds = 0.0

    def reset(self):
        started = time.perf_counter()
        observations = []
        infos = []
        for index, env in enumerate(self.environments):
            obs, info = env.reset(seed=self.seed_base + index)
            observations.append(obs)
            infos.append(info)
        self.reset_seconds += time.perf_counter() - started
        return observations, infos

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError("One action mapping is required per batch lane")
        started = time.perf_counter()
        results = [
            env.step(lane_actions)
            for env, lane_actions in zip(self.environments, actions, strict=True)
        ]
        self.step_seconds += time.perf_counter() - started
        return results

    def close(self):
        for env in self.environments:
            env.close()
        self.environments = []
