"""Verify action-latency FIFO, reset, and macro-repeat semantics."""

import argparse
import json

from robosoccer.config import load_config
from robosoccer.diagnostics import audit_action_delay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="runs/logs/phase2_protocol")
    parser.add_argument("--maximum-latency", type=int, default=5)
    args = parser.parse_args()
    result, _ = audit_action_delay(
        load_config(args.config), args.output_dir, args.maximum_latency
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
