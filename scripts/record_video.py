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
        config,
        run_dir,
        args.simulator,
        args.episodes,
        args.checkpoint,
        args.baseline,
        args.seed,
        True,
        args.profile,
    )


if __name__ == "__main__":
    main()

