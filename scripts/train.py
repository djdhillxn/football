"""Train a parameter-shared IPPO or MAPPO policy."""

import argparse

from robosoccer.config import load_config
from robosoccer.training import run_training


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("overrides", nargs="*", help="YAML-valued section.key=value overrides")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append("experiment.seed=" + str(args.seed))
    if args.total_steps is not None:
        overrides.append("train.total_steps=" + str(args.total_steps))
    if args.num_envs is not None:
        overrides.append("train.num_envs=" + str(args.num_envs))
    if args.device is not None:
        overrides.append("train.device=" + args.device)
    if args.run_name is not None:
        overrides.append("experiment.name=" + args.run_name)
    config = load_config(args.config, overrides)
    run_training(
        config,
        source_config=args.config,
        parsed_args=vars(args),
        run_name=args.run_name,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()

