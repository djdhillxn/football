"""Benchmark Phase 3 collection/update throughput without creating scientific runs."""

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from robosoccer.config import apply_overrides, load_config
from robosoccer.recurrent import RecurrentMAPPOTrainer
from robosoccer.utils import get_pyplot, write_json


def benchmark(config, num_envs, updates):
    config = apply_overrides(
        config,
        [
            "train.num_envs=" + str(num_envs),
            "train.progress_bar=false",
            "experiment.tensorboard=false",
        ],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix="robosoccer-throughput-") as temporary:
        root = Path(temporary)
        for relative in ["models", "checkpoints", "logs", "videos"]:
            (root / relative).mkdir(parents=True, exist_ok=True)
        reset_started = time.perf_counter()
        trainer = RecurrentMAPPOTrainer(config, root)
        reset_seconds = time.perf_counter() - reset_started
        totals = {
            "rollout_seconds": 0.0,
            "rollout_transfer_seconds": 0.0,
            "actor_inference_seconds": 0.0,
            "environment_stepping_seconds": 0.0,
            "actor_optimization_seconds": 0.0,
            "critic_optimization_seconds": 0.0,
        }
        transitions = 0
        try:
            for _ in range(int(updates)):
                started = time.perf_counter()
                rollout = trainer.collect_rollout()
                totals["rollout_seconds"] += time.perf_counter() - started
                for name, value in trainer.last_rollout_timing.items():
                    totals[name] += float(value)
                update_metrics = trainer.ppo_update(rollout)
                totals["actor_optimization_seconds"] += update_metrics[
                    "actor_optimization_seconds"
                ]
                totals["critic_optimization_seconds"] += update_metrics[
                    "critic_optimization_seconds"
                ]
                transitions += int(num_envs) * int(config["train"]["rollout_steps"])
        finally:
            trainer.close()
    optimization = (
        totals["actor_optimization_seconds"] + totals["critic_optimization_seconds"]
    )
    total = totals["rollout_seconds"] + optimization
    allocated = reserved = maximum = 0
    if torch.cuda.is_available():
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        maximum = int(torch.cuda.max_memory_allocated())
    return {
        "num_envs": int(num_envs),
        "updates": int(updates),
        "rollout_steps": int(config["train"]["rollout_steps"]),
        "actor_minibatch_size": int(config["ppo"]["actor_minibatch_size"]),
        "recurrent_sequence_length": int(
            config["phase3"]["recurrent"]["sequence_length"]
        ),
        "reset_seconds": reset_seconds,
        **totals,
        "ppo_update_seconds": optimization,
        "effective_transitions_per_second": transitions / max(1e-9, total),
        "agent_steps_per_second": transitions
        * int(config["phase3"]["maximum_attackers"])
        / max(1e-9, total),
        "transitions": transitions,
        "cpu_count": os.cpu_count(),
        "cuda_allocated_bytes": allocated,
        "cuda_reserved_bytes": reserved,
        "cuda_peak_memory_bytes": maximum,
        "backend": "authoritative_lane_physics_batched_policy",
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path, rows):
    plt = get_pyplot()
    counts = [row["num_envs"] for row in rows]
    throughput = [row["effective_transitions_per_second"] for row in rows]
    rollout = [row["rollout_seconds"] for row in rows]
    actor = [row["actor_optimization_seconds"] for row in rows]
    critic = [row["critic_optimization_seconds"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(counts, throughput, marker="o")
    axes[0].set(xlabel="Environments", ylabel="Transitions/s", title="Throughput")
    axes[0].grid(alpha=0.25)
    axes[1].bar(counts, rollout, label="rollout")
    axes[1].bar(counts, actor, bottom=rollout, label="actor opt")
    bottoms = [rollout[index] + actor[index] for index in range(len(rows))]
    axes[1].bar(counts, critic, bottom=bottoms, label="critic opt")
    axes[1].set(xlabel="Environments", ylabel="Seconds", title="Measured wall time")
    axes[1].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase3_smoke.yaml")
    parser.add_argument("--num-envs", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--output", default="runs/phase3_throughput.json")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = [benchmark(config, count, args.updates) for count in args.num_envs]
    output = Path(args.output)
    result = {
        "schema_version": 1,
        "scientific_result": False,
        "rows": rows,
        "csv": str(output.with_suffix(".csv")),
        "plot": str(output.with_suffix(".png")),
        "capability_note": (
            "The executor supports 128--512 lanes subject to host/GPU memory. "
            "Physics lanes remain authoritative; policy inference and tensors are batched."
        ),
    }
    write_json(output, result)
    write_csv(output.with_suffix(".csv"), rows)
    write_plot(output.with_suffix(".png"), rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
