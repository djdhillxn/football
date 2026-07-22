"""Synchronize RobotSoccerTransfer artifacts between Drive and this checkout."""

import argparse
import json
import logging
import sys
from pathlib import Path

from robosoccer.artifacts import (
    REPOSITORY_ROOT,
    prune_local_training_artifacts,
    sync_artifacts_from_drive,
    sync_run_to_drive,
    sync_workspace_to_drive,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=["pull", "push-run", "push-all"],
        default="pull",
        help="pull by default; push-run saves one run; push-all is a final safety sync.",
    )
    parser.add_argument(
        "--drive-project",
        help=(
            "RobotSoccerTransfer folder on Drive. Auto-detected in Colab and on macOS "
            "through Google Drive for desktop."
        ),
    )
    parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
        help="Local repository root. Defaults to the checkout containing this script.",
    )
    parser.add_argument("--run-dir", help="Finished run directory required by push-run.")
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Include Drive runs still marked running during a pull.",
    )
    parser.add_argument(
        "--include-training-artifacts",
        action="store_true",
        help="Copy models, checkpoints, weights, replay pickles, and archives. Used by Colab.",
    )
    parser.add_argument(
        "--prune-local-training-artifacts",
        action="store_true",
        help="Remove already-downloaded local training artifacts; never changes Drive.",
    )
    parser.add_argument(
        "--verify-text",
        action="store_true",
        help="Content-check existing text files during a pull; slower on cloud-only Drive files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only.")
    return parser


def setup_cli_logging(repository_root, write_file=True):
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)
    if not write_file:
        return
    log_path = Path(repository_root) / "runs" / "logs" / "artifact_sync.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(file_handler)


def main():
    args = build_parser().parse_args()
    if args.include_training_artifacts and args.prune_local_training_artifacts:
        raise ValueError(
            "Choose either --include-training-artifacts or "
            "--prune-local-training-artifacts, not both."
        )
    if args.action != "pull" and (
        args.include_training_artifacts or args.prune_local_training_artifacts
    ):
        raise ValueError(
            "--include-training-artifacts and --prune-local-training-artifacts "
            "apply only to pull."
        )
    repository_root = Path(args.repository_root).expanduser().resolve()
    setup_cli_logging(repository_root, write_file=not args.dry_run)
    if args.action == "pull":
        result = sync_artifacts_from_drive(
            args.drive_project,
            repository_root,
            include_running=args.include_running,
            include_training_artifacts=args.include_training_artifacts,
            dry_run=args.dry_run,
            verify_text_contents=args.verify_text,
        )
        if args.prune_local_training_artifacts:
            result["pruned_local_training_artifacts"] = prune_local_training_artifacts(
                repository_root / "runs",
                dry_run=args.dry_run,
            )
    elif args.action == "push-run":
        if not args.run_dir:
            raise ValueError("push-run requires --run-dir")
        result = sync_run_to_drive(args.run_dir, args.drive_project, dry_run=args.dry_run)
    else:
        result = sync_workspace_to_drive(
            args.drive_project,
            repository_root,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.getLogger(__name__).debug("Artifact synchronization failed", exc_info=True)
        logging.getLogger(__name__).error("%s", exc)
        raise SystemExit(2) from None
