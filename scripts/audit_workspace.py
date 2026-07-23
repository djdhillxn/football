"""Audit run artifacts, portable pointers, and notebook references without mutation."""

import argparse
import json

from robosoccer.artifacts import audit_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit_workspace(args.repository_root)
    print(json.dumps(result, indent=2))
    if args.strict and result["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
