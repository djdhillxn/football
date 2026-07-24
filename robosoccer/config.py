"""Small YAML configuration system with inheritance and dot-list overrides."""

import copy
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def deep_merge(base, override):
    """Recursively merge dictionaries without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_recursive(path, active_paths):
    path = Path(path).expanduser().resolve()
    if path in active_paths:
        chain = " -> ".join(str(item) for item in [*active_paths, path])
        raise ValueError("Circular configuration inheritance detected: " + chain)
    if not path.is_file():
        raise FileNotFoundError("Configuration file does not exist: " + str(path))
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid YAML in configuration " + str(path) + ": " + str(exc)) from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level configuration must be a mapping: " + str(path))
    parent_name = loaded.pop("inherits", None)
    if parent_name is None:
        return loaded
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ValueError("'inherits' must be a non-empty path string in " + str(path))
    parent_path = (path.parent / parent_name).resolve()
    parent = _load_recursive(parent_path, [*active_paths, path])
    return deep_merge(parent, loaded)


def parse_override(text):
    """Parse one key=value override using YAML value semantics."""
    if "=" not in text:
        raise ValueError("Override must use key=value syntax: " + text)
    dotted_key, raw_value = text.split("=", 1)
    keys = dotted_key.split(".")
    if any(not key for key in keys):
        raise ValueError("Override contains an empty key component: " + text)
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError("Invalid YAML value in override " + text + ": " + str(exc)) from exc
    return keys, value


def apply_overrides(config, overrides):
    """Apply command-line dot-list overrides to a deep copy of a configuration."""
    result = copy.deepcopy(config)
    for text in overrides or []:
        keys, value = parse_override(text)
        cursor = result
        for key in keys[:-1]:
            existing = cursor.get(key)
            if existing is None:
                cursor[key] = {}
            elif not isinstance(existing, dict):
                raise ValueError(
                    "Cannot set nested override below non-mapping key: " + ".".join(keys[:-1])
                )
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return result


def _require(config, dotted_key):
    cursor = config
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            raise ValueError("Missing required configuration field: " + dotted_key)
        cursor = cursor[key]
    return cursor


def validate_config(config):
    """Validate fields whose mistakes would invalidate training or evaluation."""
    required_sections = [
        "experiment",
        "environment",
        "transfer_environment",
        "reward",
        "opponent",
        "observations",
        "randomization",
        "curriculum",
        "model",
        "ppo",
        "train",
        "evaluation",
        "video",
        "logging",
    ]
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError("Missing required configuration section: " + section)

    positive_fields = [
        "environment.field_length",
        "environment.field_width",
        "environment.dt",
        "environment.macro_action_repeat",
        "environment.max_episode_steps",
        "train.total_steps",
        "train.num_envs",
        "train.rollout_steps",
        "ppo.actor_learning_rate",
        "ppo.critic_learning_rate",
        "ppo.gamma",
        "ppo.gae_lambda",
    ]
    for field in positive_fields:
        value = _require(config, field)
        if not isinstance(value, int | float) or value <= 0:
            raise ValueError(field + " must be a positive number")

    if config["ppo"].get("algorithm") not in {"ippo", "mappo"}:
        raise ValueError("ppo.algorithm must be 'ippo' or 'mappo'")
    if config["randomization"].get("mode") not in {"none", "uniform", "failure_directed"}:
        raise ValueError("randomization.mode must be none, uniform, or failure_directed")
    if not 0 < config["ppo"]["gamma"] <= 1:
        raise ValueError("ppo.gamma must be in (0, 1]")
    if not 0 <= config["ppo"]["gae_lambda"] <= 1:
        raise ValueError("ppo.gae_lambda must be in [0, 1]")
    if config["environment"]["goal_width"] >= config["environment"]["field_width"]:
        raise ValueError("environment.goal_width must be smaller than field_width")
    profiles = config["randomization"].get("profiles")
    if not isinstance(profiles, dict) or "nominal" not in profiles:
        raise ValueError("randomization.profiles must be a mapping containing 'nominal'")
    maximum = config["curriculum"].get("maximum_profile_probability", 1.0)
    if not 0 < maximum <= 1:
        raise ValueError("curriculum.maximum_profile_probability must be in (0, 1]")
    disabled = config["randomization"].get("disabled_families", [])
    if not isinstance(disabled, list):
        raise ValueError("randomization.disabled_families must be a list")
    disabled_parameters = config["randomization"].get("disabled_parameters", [])
    if not isinstance(disabled_parameters, list):
        raise ValueError("randomization.disabled_parameters must be a list")
    phase3 = config.get("phase3", {})
    if phase3.get("enabled") and phase3.get("active_stage") == "stage_r":
        if int(phase3.get("reward_schema_version", 1)) < 2:
            raise ValueError("Stage R requires phase3.reward_schema_version >= 2")
        if float(config.get("phase3_reward", {}).get("controlled_reception", -1.0)) != 0.0:
            raise ValueError("Stage R requires zero controlled-reception reward")
        stage = phase3.get("stages", {}).get("stage_r")
        stage_r = phase3.get("stage_r", {})
        if not stage:
            raise ValueError("Stage R configuration is missing phase3.stages.stage_r")
        training_seed_base = int(stage_r.get("training_episode_seed_base", -1))
        evaluation_seed_bases = config.get("evaluation", {}).get("seed_bases", {})
        if training_seed_base < 0 or training_seed_base in {
            int(value) for value in evaluation_seed_bases.values()
        }:
            raise ValueError(
                "Stage R requires a declared training seed base distinct from "
                "evaluation protocols"
            )
        for names_key, probabilities_key in [
            ("scenarios", "probabilities"),
            ("defender_styles", "defender_probabilities"),
        ]:
            names = stage.get(names_key, [])
            probabilities = stage.get(probabilities_key, [])
            if len(names) != len(probabilities) or abs(sum(probabilities) - 1.0) > 1e-8:
                raise ValueError(
                    "Stage R " + probabilities_key + " must align and sum to one"
                )
    return config


def load_config(path, overrides=None):
    """Load, inherit, override, and validate a YAML configuration."""
    resolved = _load_recursive(path, [])
    resolved = apply_overrides(resolved, overrides)
    validate_config(resolved)
    logger.debug("Resolved configuration from %s", path)
    return resolved


def save_config(config, path):
    """Save a resolved configuration as readable YAML."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )
