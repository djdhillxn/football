"""Run the locked Phase 3 baseline calibration and fail closed."""

import argparse
import json

from robosoccer.config import load_config
from robosoccer.phase3 import run_phase3_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase3_base.yaml")
    parser.add_argument("--output-dir", default="runs/phase3_calibration")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-base", type=int, default=310000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    full_episodes = int(config["phase3"]["calibration"]["episodes_per_cell"])
    smoke = bool(
        args.smoke
        or "smoke" in str(config["experiment"]["name"]).lower()
        or (args.episodes is not None and args.episodes < full_episodes)
    )
    result = run_phase3_calibration(
        config,
        args.output_dir,
        episodes=args.episodes,
        smoke=smoke,
        seed_base=args.seed_base,
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
