"""Record headless MP4 rollouts for a learned policy or scripted baseline."""

import argparse
from pathlib import Path

from robosoccer.config import load_config
from robosoccer.evaluation import record_videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--simulator", choices=["abstract", "pymunk"], default="abstract")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--baseline", choices=["random", "double_chase", "role_based"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", default="nominal")
    parser.add_argument("--scenario", choices=["nominal", "cooperation"], default="nominal")
    parser.add_argument(
        "--matched",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record exactly the requested seeds instead of selecting outcomes from extra candidates.",
    )
    args = parser.parse_args()
    if args.run_dir is None:
        if args.config is None or args.baseline is None:
            parser.error("--run-dir is required for learned policies; baselines require --config")
        run_dir = Path("runs") / "manual_baseline_videos"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = load_config(args.config)
    else:
        run_dir = Path(args.run_dir).expanduser().resolve()
        config_path = args.config or run_dir / "resolved_config.yaml"
        config = load_config(config_path)
    record_videos(
        config=config,
        run_dir=run_dir,
        simulator=args.simulator,
        episodes=args.episodes,
        checkpoint=args.checkpoint,
        baseline=args.baseline,
        seed=args.seed,
        deterministic=True,
        profile=args.profile,
        scenario=args.scenario,
        matched=args.matched,
    )


if __name__ == "__main__":
    main()
