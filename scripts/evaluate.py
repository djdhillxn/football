"""Evaluate a frozen learned policy in nominal, profile, or robustness suites."""

import argparse

from robosoccer.evaluation import evaluate_learned_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--simulator", choices=["abstract", "pymunk", "webots"], default="abstract")
    parser.add_argument(
        "--suite",
        choices=["standard", "transfer", "profiles", "robustness", "all"],
        default="standard",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.simulator == "webots":
        raise RuntimeError(
            "Webots evaluation is optional and not implemented; see webots/README.md for the adapter contract"
        )
    evaluate_learned_run(
        args.run_dir,
        args.checkpoint,
        args.simulator,
        args.suite,
        args.episodes,
        args.seed,
        args.deterministic,
        args.prefix,
        args.device,
    )


if __name__ == "__main__":
    main()

