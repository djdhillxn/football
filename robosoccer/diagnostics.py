"""Deterministic, artifact-producing diagnostics for simulator control semantics."""

import copy
import csv
import json
from pathlib import Path
from types import MethodType

from robosoccer.environment import ACTION_NAMES, AGENTS, make_environment
from robosoccer.utils import write_json


def expected_delayed_actions(requested_actions, latency, initial_action=6):
    """Return FIFO outputs for an action queue prefilled with the hold action."""
    queue = [int(initial_action)] * int(latency)
    applied = []
    for action in requested_actions:
        queue.append(int(action))
        applied.append(int(queue.pop(0)))
    return applied


def audit_action_delay(config, output_dir=None, maximum_latency=5):
    """Audit FIFO, reset, and macro-repeat semantics in both simulator classes.

    Physics is deliberately frozen in this diagnostic. That isolates the control
    adapter from goals, collisions, and early termination while still exercising
    the production reset and ParallelEnv step paths.
    """
    maximum_latency = int(maximum_latency)
    if maximum_latency < 0:
        raise ValueError("maximum_latency must be non-negative")
    requested = [index % len(ACTION_NAMES) for index in range(maximum_latency + 9)]
    audit_rows = []
    cases = []

    for simulator in ["abstract", "pymunk"]:
        for latency in range(maximum_latency + 1):
            case_config = copy.deepcopy(config)
            case_config["randomization"]["mode"] = "none"
            case_config["environment"]["max_episode_steps"] = len(requested) + 5
            case_config["environment"]["stationary_truncation_steps"] = len(requested) + 5
            env = make_environment(case_config, simulator)
            repeated_actions = []

            def frozen_physics(self, actions, _dt, records=repeated_actions):
                records.append(
                    {
                        "actions": {agent: int(actions[agent]) for agent in AGENTS},
                        "remaining": {
                            agent: int(self.players[agent]["action_repeat_remaining"])
                            for agent in AGENTS
                        },
                    }
                )

            try:
                env.reset(
                    seed=int(config["evaluation"]["seed_bases"].get("delay_audit", 260000))
                    + latency,
                    options={"sampled_parameters": {"action_latency": latency}},
                )
                env._physics_substep = MethodType(frozen_physics, env)
                expected = expected_delayed_actions(requested, latency)
                case_passed = len(env.action_queues[AGENTS[0]]) == latency
                previous_repeat_count = 0
                for step_index, requested_action in enumerate(requested):
                    actions = {
                        AGENTS[0]: int(requested_action),
                        AGENTS[1]: int((requested_action + 1) % len(ACTION_NAMES)),
                    }
                    _, _, terminations, truncations, infos = env.step(actions)
                    repeat_records = repeated_actions[previous_repeat_count:]
                    previous_repeat_count = len(repeated_actions)
                    observed_repeat = len(repeat_records)
                    applied = int(infos[AGENTS[0]]["applied_action"])
                    queued = list(infos[AGENTS[0]]["queued_actions"])
                    expected_applied = int(expected[step_index])
                    expected_age = None if step_index < latency else latency
                    observed_age = infos[AGENTS[0]]["applied_action_age_steps"]
                    row_passed = (
                        applied == expected_applied
                        and observed_age == expected_age
                        and observed_repeat == int(case_config["environment"]["macro_action_repeat"])
                        and all(
                            record["actions"][AGENTS[0]] == applied for record in repeat_records
                        )
                        and len(queued) == latency
                        and not any(terminations.values())
                        and not any(truncations.values())
                    )
                    case_passed = case_passed and row_passed
                    audit_rows.append(
                        {
                            "simulator": simulator,
                            "latency": latency,
                            "step": step_index,
                            "requested_action": int(requested_action),
                            "expected_applied_action": expected_applied,
                            "applied_action": applied,
                            "expected_action_age_steps": expected_age,
                            "applied_action_age_steps": observed_age,
                            "queued_actions": json.dumps(queued),
                            "macro_repeats_observed": observed_repeat,
                            "passed": bool(row_passed),
                        }
                    )

                env.reset(
                    seed=int(config["evaluation"]["seed_bases"].get("delay_audit", 260000))
                    + 100 + latency,
                    options={"sampled_parameters": {"action_latency": latency}},
                )
                env._physics_substep = MethodType(frozen_physics, env)
                _, _, _, _, reset_infos = env.step({agent: 2 for agent in AGENTS})
                expected_after_reset = 2 if latency == 0 else 6
                reset_passed = (
                    int(reset_infos[AGENTS[0]]["applied_action"]) == expected_after_reset
                    and len(reset_infos[AGENTS[0]]["queued_actions"]) == latency
                )
                case_passed = case_passed and reset_passed
                cases.append(
                    {
                        "simulator": simulator,
                        "latency": latency,
                        "fifo_passed": bool(case_passed),
                        "reset_passed": bool(reset_passed),
                    }
                )
            finally:
                env.close()

    result = {
        "passed": bool(cases) and all(case["fifo_passed"] for case in cases),
        "maximum_latency": maximum_latency,
        "macro_action_repeat": int(config["environment"]["macro_action_repeat"]),
        "case_count": len(cases),
        "cases": cases,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "action_delay_audit.json", result)
        fieldnames = list(audit_rows[0]) if audit_rows else []
        with (output_dir / "action_delay_trace.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(audit_rows)
    return result, audit_rows
