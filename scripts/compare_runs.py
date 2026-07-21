"""Aggregate completed baseline and learned-policy evaluation artifacts."""

import argparse

from robosoccer.evaluation import compare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*")
    parser.add_argument("--phase", choices=["smoke", "final", "all"], default="final")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--export-report", action="store_true")
    args = parser.parse_args()
    compare_runs(args.runs, args.phase, args.runs_root, args.export_report)


if __name__ == "__main__":
    main()

