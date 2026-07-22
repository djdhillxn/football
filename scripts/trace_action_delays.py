"""Persist matched policy traces for requested, queued, and applied delayed actions."""

import argparse

from robosoccer.evaluation import trace_learned_action_delays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--simulator", choices=["abstract", "pymunk"], default="pymunk")
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--localization-noise", type=float, default=0.0)
    parser.add_argument("--prefix", default="confirmatory_delay_traces")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    trace_learned_action_delays(
        run_dir=args.run_dir,
        delays=args.delays,
        seed=args.seed,
        simulator=args.simulator,
        output_name=args.prefix,
        checkpoint=args.checkpoint,
        localization_noise=args.localization_noise,
        device=args.device,
    )


if __name__ == "__main__":
    main()
