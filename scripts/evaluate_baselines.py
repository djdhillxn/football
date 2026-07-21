"""Evaluate random, double-chase, and role-based policies in both simulators."""

import argparse

from robosoccer.config import load_config
from robosoccer.evaluation import evaluate_baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--method",
        choices=["random", "double_chase", "role_based", "all"],
        default="all",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    methods = (
        ["random", "double_chase", "role_based"] if args.method == "all" else [args.method]
    )
    evaluate_baselines(
        config,
        args.episodes,
        methods,
        args.run_name,
        args.videos,
        source_config=args.config,
        parsed_args=vars(args),
    )


if __name__ == "__main__":
    main()
