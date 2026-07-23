"""Standard-library artifact synchronization for Colab, Drive, and local checkouts."""

import filecmp
import json
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNC_IGNORED_NAMES = {".DS_Store"}
TRAINING_ARTIFACT_DIRECTORIES = {"checkpoints", "models"}
TRAINING_ARTIFACT_SUFFIXES = {".ckpt", ".pickle", ".pkl", ".pt", ".pth", ".ts", ".zip"}
SYNC_STATE_NAME = ".robosoccer_runs_sync_state.json"
SYNC_WORKERS = 8
AUTHORED_REPORT_FILES = {"main.tex", "surrogate_notes.tex", "references.bib"}
REPORT_BUILD_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".run.xml",
    ".toc",
}
CONTENT_CHECKED_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


def safe_name(text):
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in str(text)
    )
    return cleaned.strip("-") or "run"


def process_id():
    return os.getpid()


def discover_drive_project(explicit_path=None):
    """Find the persistent RobotSoccerTransfer folder in Colab or Drive Desktop."""
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"Google Drive project folder was not found: {candidate}")
        return candidate

    environment_path = os.environ.get("ROBOSOCCER_DRIVE_PROJECT")
    if environment_path:
        return discover_drive_project(environment_path)

    colab_path = Path("/content/drive/MyDrive/RobotSoccerTransfer")
    if colab_path.is_dir():
        return colab_path

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    matches = []
    for project_name in ["RobotSoccerTransfer", "Robot Soccer Transfer"]:
        matches.extend(cloud_storage.glob(f"GoogleDrive-*/My Drive/{project_name}"))
    matches = sorted({path.resolve() for path in matches})
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            "Multiple RobotSoccerTransfer Drive folders were found. "
            f"Pass --drive-project explicitly:\n{choices}"
        )
    raise FileNotFoundError(
        "Could not find Google Drive's RobotSoccerTransfer folder. Make the folder "
        "available in Google Drive for desktop, pass --drive-project, or set "
        "ROBOSOCCER_DRIVE_PROJECT."
    )


def read_run_metadata(run_dir):
    metadata_path = Path(run_dir) / "run_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def artifact_file_is_current(source, destination, verify_contents=None):
    """Use size by default so unchanged cloud-only Drive files remain unhydrated."""
    source = Path(source)
    destination = Path(destination)
    if not destination.is_file():
        return False
    source_stat = source.stat()
    destination_stat = destination.stat()
    if source_stat.st_size != destination_stat.st_size:
        return False
    if verify_contents:
        return filecmp.cmp(source, destination, shallow=False)
    return True


def copy_artifact_file(
    source,
    destination,
    dry_run=False,
    verify_contents=None,
    preserve_newer_destination=False,
):
    """Copy one artifact through a sibling temporary file for atomic replacement."""
    source = Path(source)
    destination = Path(destination)
    if (
        preserve_newer_destination
        and destination.is_file()
        and destination.stat().st_mtime > source.stat().st_mtime
    ):
        logger.debug("Kept newer report artifact %s instead of %s", destination, source)
        return False
    if artifact_file_is_current(source, destination, verify_contents=verify_contents):
        return False
    if dry_run:
        logger.info("Would copy %s -> %s", source, destination)
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.robosoccer-sync-{process_id()}.tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return True


def run_signature(run_dir):
    """Return cheap Drive metadata used to avoid re-walking unchanged runs."""
    metadata_path = Path(run_dir) / "run_metadata.json"
    try:
        metadata_stat = metadata_path.stat()
    except OSError:
        return None
    videos_mtime = None
    try:
        videos_mtime = (Path(run_dir) / "videos").stat().st_mtime_ns
    except OSError:
        pass
    return {
        "metadata_size": metadata_stat.st_size,
        "metadata_mtime_ns": metadata_stat.st_mtime_ns,
        "videos_mtime_ns": videos_mtime,
    }


def load_sync_state(local_runs):
    state_path = Path(local_runs).parent / SYNC_STATE_NAME
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault("analysis", {})
    state.setdefault("full", {})
    return state_path, state


def save_sync_state(state_path, state):
    state_path = Path(state_path)
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(state_path)


def iter_artifact_files(source, include_training_artifacts=True):
    """Yield files while pruning heavy training trees before Drive hydrates them."""
    for directory, directory_names, file_names in os.walk(source):
        if not include_training_artifacts:
            directory_names[:] = [
                name for name in directory_names if name not in TRAINING_ARTIFACT_DIRECTORIES
            ]
        directory_path = Path(directory)
        for file_name in file_names:
            path = directory_path / file_name
            if path.name in SYNC_IGNORED_NAMES:
                continue
            if not include_training_artifacts and path.suffix.lower() in TRAINING_ARTIFACT_SUFFIXES:
                continue
            yield path


def merge_artifact_directory(
    source,
    destination,
    dry_run=False,
    skipped_relative_paths=None,
    verify_text_contents=False,
    include_training_artifacts=True,
):
    """Merge files without deleting destination-only artifacts."""
    source = Path(source)
    destination = Path(destination)
    skipped = set(skipped_relative_paths or [])
    paths = list(
        iter_artifact_files(
            source,
            include_training_artifacts=include_training_artifacts,
        )
    )
    # Metadata is the completion marker and must become visible only after the payload.
    paths.sort(key=lambda path: (path.name == "run_metadata.json", path.as_posix()))

    def copy_one(path):
        relative = path.relative_to(source)
        if path.name in SYNC_IGNORED_NAMES or relative.as_posix() in skipped:
            return False
        verify_contents = verify_text_contents and path.suffix.lower() in CONTENT_CHECKED_SUFFIXES
        return copy_artifact_file(
            path,
            destination / relative,
            dry_run=dry_run,
            verify_contents=verify_contents,
        )

    payload_paths = [path for path in paths if path.name != "run_metadata.json"]
    metadata_paths = [path for path in paths if path.name == "run_metadata.json"]
    if dry_run or len(payload_paths) < 2:
        copied = sum(bool(copy_one(path)) for path in payload_paths)
    else:
        with ThreadPoolExecutor(max_workers=min(SYNC_WORKERS, len(payload_paths))) as executor:
            copied = sum(bool(changed) for changed in executor.map(copy_one, payload_paths))
    copied += sum(bool(copy_one(path)) for path in metadata_paths)
    return copied


def prune_local_training_artifacts(local_runs, dry_run=False):
    """Remove local weights/checkpoints only; the persistent Drive copy is untouched."""
    local_runs = Path(local_runs)
    removed_directories = 0
    removed_files = 0
    reclaimed_bytes = 0
    if not local_runs.exists():
        return {
            "removed_directories": 0,
            "removed_files": 0,
            "reclaimed_bytes": 0,
        }

    for path in local_runs.iterdir():
        if not path.is_file() or path.suffix.lower() not in TRAINING_ARTIFACT_SUFFIXES:
            continue
        removed_files += 1
        reclaimed_bytes += path.stat().st_size
        if dry_run:
            logger.info("Would remove local training artifact: %s", path)
        else:
            path.unlink()

    for run_dir in local_runs.iterdir():
        if not run_dir.is_dir():
            continue
        for directory_name in TRAINING_ARTIFACT_DIRECTORIES:
            artifact_dir = run_dir / directory_name
            if not artifact_dir.is_dir():
                continue
            file_sizes = [path.stat().st_size for path in artifact_dir.rglob("*") if path.is_file()]
            removed_directories += 1
            removed_files += len(file_sizes)
            reclaimed_bytes += sum(file_sizes)
            if dry_run:
                logger.info("Would remove local training artifacts: %s", artifact_dir)
            else:
                shutil.rmtree(artifact_dir)

        for path in run_dir.rglob("*"):
            relative_parts = path.relative_to(run_dir).parts
            if any(name in TRAINING_ARTIFACT_DIRECTORIES for name in relative_parts[:-1]):
                continue
            if not path.is_file() or path.suffix.lower() not in TRAINING_ARTIFACT_SUFFIXES:
                continue
            removed_files += 1
            reclaimed_bytes += path.stat().st_size
            if dry_run:
                logger.info("Would remove local training artifact: %s", path)
            else:
                path.unlink()

    # A later explicit full restore must not trust cache entries created before
    # these local files were removed.
    if not dry_run and (removed_directories or removed_files):
        state_path, state = load_sync_state(local_runs)
        if state_path.is_file():
            state["full"] = {}
            save_sync_state(state_path, state)

    return {
        "removed_directories": removed_directories,
        "removed_files": removed_files,
        "reclaimed_bytes": reclaimed_bytes,
    }


def latest_run_for_experiment(runs_root, experiment_name, statuses=None):
    """Return the newest metadata-backed run, including failed runs when requested."""
    runs_root = Path(runs_root)
    accepted = set(statuses or ["complete"])
    candidates = []
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs folder was not found: {runs_root}")
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        metadata = read_run_metadata(run_dir)
        if not metadata:
            continue
        if metadata.get("experiment_name") != experiment_name:
            continue
        if metadata.get("status") in accepted:
            candidates.append(run_dir)
    if not candidates:
        states = ", ".join(sorted(accepted))
        raise FileNotFoundError(
            f"No {states} run for experiment '{experiment_name}' was found in {runs_root}"
        )
    return max(candidates, key=lambda path: path.name).resolve()


def resolve_run_pointer(pointer_path, repository_root=None):
    """Resolve portable pointers and recover stale Colab/Mac absolute paths by basename."""
    pointer_path = Path(pointer_path)
    if not pointer_path.is_file():
        raise FileNotFoundError("Run pointer was not found: " + str(pointer_path))
    raw = pointer_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("Run pointer is empty: " + str(pointer_path))
    candidate = Path(raw).expanduser()
    repository = Path(repository_root or REPOSITORY_ROOT).expanduser().resolve()
    attempts = []
    if candidate.is_absolute():
        attempts.append(candidate)
    else:
        attempts.extend([repository / candidate, pointer_path.parent.parent / candidate])
    basename = candidate.name
    attempts.extend([pointer_path.parent / basename, repository / "runs" / basename])
    for attempt in attempts:
        if attempt.is_dir():
            return attempt.resolve()
    raise FileNotFoundError(
        "Pointer target is unavailable and basename recovery failed: "
        + raw
        + " ("
        + str(pointer_path)
        + ")"
    )


def audit_workspace(repository_root=None):
    """Read-only artifact audit covering pointers, metadata, configs, and notebook references."""
    repository = Path(repository_root or REPOSITORY_ROOT).expanduser().resolve()
    runs_root = repository / "runs"
    issues = []
    runs = {}
    duplicate_ids = {}
    auxiliary_directories = {"comparisons", "logs", "manual_baseline_videos"}
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if (
                run_dir.name in auxiliary_directories
                or run_dir.name.startswith("phase3_calibration")
            ):
                continue
            metadata = read_run_metadata(run_dir)
            if metadata is None:
                issues.append(
                    {
                        "severity": "warning",
                        "kind": "missing_metadata",
                        "path": str(run_dir.relative_to(repository)),
                    }
                )
                continue
            run_id = run_dir.name
            duplicate_ids.setdefault(run_id, []).append(str(run_dir))
            record = {
                "path": str(run_dir.relative_to(repository)),
                "status": metadata.get("status"),
                "experiment_name": metadata.get("experiment_name"),
                "seed": metadata.get("seed"),
                "config_present": (run_dir / "resolved_config.yaml").is_file(),
            }
            runs[run_id] = record
            if metadata.get("status") not in {"complete", "failed"}:
                issues.append(
                    {
                        "severity": "warning",
                        "kind": "incomplete_run",
                        "path": record["path"],
                        "status": metadata.get("status"),
                    }
                )
            if not record["config_present"]:
                issues.append(
                    {
                        "severity": "error",
                        "kind": "missing_resolved_config",
                        "path": record["path"],
                    }
                )
            for label, artifact_path in metadata.get("output_artifact_paths", {}).items():
                if not isinstance(artifact_path, str):
                    continue
                candidate = Path(artifact_path)
                if candidate.is_absolute() and candidate.is_file():
                    continue
                recovered = next(iter(run_dir.rglob(candidate.name)), None)
                if (recovered is None or not recovered.is_file()) and label in {
                    "final_checkpoint",
                    "best_checkpoint",
                    "metrics_csv",
                }:
                    issues.append(
                        {
                            "severity": "warning",
                            "kind": "missing_referenced_artifact",
                            "path": record["path"],
                            "label": label,
                            "reference": artifact_path,
                        }
                    )
    for run_id, paths in duplicate_ids.items():
        if len(paths) > 1:
            issues.append(
                {
                    "severity": "error",
                    "kind": "duplicate_run_id",
                    "run_id": run_id,
                    "paths": paths,
                }
            )
    pointers = {}
    for pointer in sorted(runs_root.glob("latest_*.txt")) if runs_root.is_dir() else []:
        raw_pointer = pointer.read_text(encoding="utf-8").strip()
        if Path(raw_pointer).is_absolute():
            issues.append(
                {
                    "severity": "warning",
                    "kind": "nonportable_pointer",
                    "path": str(pointer.relative_to(repository)),
                    "reference": raw_pointer,
                }
            )
        try:
            target = resolve_run_pointer(pointer, repository)
            pointers[pointer.name] = {
                "target": str(target.relative_to(repository)),
                "resolved": True,
            }
        except (FileNotFoundError, ValueError) as exc:
            pointers[pointer.name] = {"target": None, "resolved": False}
            issues.append(
                {
                    "severity": "warning",
                    "kind": "stale_pointer",
                    "path": str(pointer.relative_to(repository)),
                    "message": str(exc),
                }
            )
    notebook_references = {}
    for notebook in sorted((repository / "notebooks").glob("*.ipynb")):
        text = notebook.read_text(encoding="utf-8")
        references = sorted(
            set(
                re.findall(
                    r"20[0-9]{6}_[0-9]{6}_[A-Za-z0-9_-]+_seed[0-9]+",
                    text,
                )
            )
        )
        missing = [
            reference
            for reference in references
            if not (runs_root / Path(reference).name).is_dir()
        ]
        notebook_references[notebook.name] = {
            "run_reference_count": len(references),
            "missing_local_references": missing,
        }
        for reference in missing:
            issues.append(
                {
                    "severity": "warning",
                    "kind": "notebook_run_absent_locally",
                    "path": str(notebook.relative_to(repository)),
                    "reference": reference,
                }
            )
    manifest_entries = []
    missing_manifest_runs = []
    manifest_path = runs_root / "experiment_manifest.jsonl"
    if manifest_path.is_file():
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(
                    {
                        "severity": "error",
                        "kind": "invalid_manifest_entry",
                        "path": str(manifest_path.relative_to(repository)),
                        "line": line_number,
                        "message": str(exc),
                    }
                )
                continue
            run_directory = entry.get("run_directory")
            run_id = Path(run_directory).name if run_directory else None
            manifest_entries.append(
                {
                    "run_id": run_id,
                    "status": entry.get("status"),
                    "experiment_name": entry.get("experiment_name"),
                }
            )
            if run_id and not (runs_root / run_id).is_dir():
                missing_manifest_runs.append(run_id)
                issues.append(
                    {
                        "severity": "warning",
                        "kind": "manifest_run_absent_locally",
                        "path": str(manifest_path.relative_to(repository)),
                        "reference": run_id,
                    }
                )
    stale_reports = []
    reports_root = repository / "reports"
    for stem in ["main", "surrogate_notes"]:
        source = reports_root / (stem + ".tex")
        pdf = reports_root / (stem + ".pdf")
        dependencies = [source]
        if stem == "main":
            dependencies.append(reports_root / "generated_results.tex")
        newest_source = max(
            (path.stat().st_mtime for path in dependencies if path.is_file()),
            default=0.0,
        )
        if source.is_file() and (not pdf.is_file() or newest_source > pdf.stat().st_mtime):
            stale_reports.append(stem)
            issues.append(
                {
                    "severity": "warning",
                    "kind": "report_pdf_stale_or_missing",
                    "path": str(source.relative_to(repository)),
                }
            )
    comparison_missing_sources = {}
    comparisons_root = runs_root / "comparisons"
    if comparisons_root.is_dir():
        for comparison in sorted(comparisons_root.rglob("*")):
            if not comparison.is_file() or comparison.suffix not in {".csv", ".json", ".jsonl"}:
                continue
            text = comparison.read_text(encoding="utf-8", errors="replace")
            references = sorted(
                set(
                    re.findall(
                        r"20[0-9]{6}_[0-9]{6}_[A-Za-z0-9_-]+_seed[0-9]+",
                        text,
                    )
                )
            )
            missing = [
                reference
                for reference in references
                if not (runs_root / reference).is_dir()
            ]
            if missing:
                relative = str(comparison.relative_to(repository))
                comparison_missing_sources[relative] = missing
                issues.append(
                    {
                        "severity": "warning",
                        "kind": "comparison_source_absent_locally",
                        "path": relative,
                        "references": missing,
                    }
                )
    return {
        "schema_version": 1,
        "repository": str(repository),
        "run_count": len(runs),
        "runs": runs,
        "manifest_entry_count": len(manifest_entries),
        "manifest_entries": manifest_entries,
        "missing_manifest_runs": missing_manifest_runs,
        "pointers": pointers,
        "notebooks": notebook_references,
        "stale_reports": stale_reports,
        "comparison_missing_sources": comparison_missing_sources,
        "issues": issues,
        "error_count": sum(issue["severity"] == "error" for issue in issues),
        "warning_count": sum(issue["severity"] == "warning" for issue in issues),
    }


def refresh_artifact_index(runs_root, portable=False, dry_run=False):
    """Rebuild latest pointers and the manifest from destination run metadata."""
    runs_root = Path(runs_root)
    if not dry_run:
        runs_root.mkdir(parents=True, exist_ok=True)
    candidates = {}
    manifest_rows = []
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir(), key=lambda path: path.name):
            if not run_dir.is_dir():
                continue
            metadata = read_run_metadata(run_dir)
            if not metadata or metadata.get("status") not in {"complete", "failed"}:
                continue
            normalized = dict(metadata)
            normalized["run_directory"] = (
                f"runs/{run_dir.name}" if portable else str(run_dir.resolve())
            )
            manifest_rows.append(normalized)
            if metadata.get("status") == "complete" and metadata.get("experiment_name"):
                name = safe_name(metadata["experiment_name"])
                candidates.setdefault(name, []).append(run_dir)

    pointers = {}
    for name, run_dirs in sorted(candidates.items()):
        latest = max(run_dirs, key=lambda path: path.name)
        contents = f"runs/{latest.name}\n" if portable else f"{latest.resolve()}\n"
        pointer = runs_root / f"latest_{name}.txt"
        current = pointer.read_text(encoding="utf-8") if pointer.is_file() else None
        if current != contents:
            if dry_run:
                logger.info("Would write %s -> %s", pointer, contents.strip())
            else:
                temporary = pointer.with_suffix(".txt.tmp")
                temporary.write_text(contents, encoding="utf-8")
                temporary.replace(pointer)
        pointers[name] = latest

    manifest = runs_root / "experiment_manifest.jsonl"
    contents = "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in manifest_rows)
    current = manifest.read_text(encoding="utf-8") if manifest.is_file() else None
    if current != contents:
        if dry_run:
            logger.info("Would rebuild %s with %d entries", manifest, len(manifest_rows))
        else:
            temporary = manifest.with_suffix(".jsonl.tmp")
            temporary.write_text(contents, encoding="utf-8")
            temporary.replace(manifest)
    return pointers


def _sync_report_artifacts(source, destination, dry_run=False):
    if not Path(source).is_dir():
        return 0
    copied = 0
    for path in sorted(Path(source).rglob("*")):
        if not path.is_file() or path.name in SYNC_IGNORED_NAMES:
            continue
        relative = path.relative_to(source)
        if relative.parent == Path(".") and path.name in AUTHORED_REPORT_FILES:
            continue
        if any(path.name.endswith(suffix) for suffix in REPORT_BUILD_SUFFIXES):
            continue
        if copy_artifact_file(
            path,
            Path(destination) / relative,
            dry_run=dry_run,
            verify_contents=True,
            preserve_newer_destination=True,
        ):
            copied += 1
    return copied


def sync_artifacts_from_drive(
    drive_project=None,
    repository_root=None,
    include_running=False,
    include_training_artifacts=False,
    dry_run=False,
    verify_text_contents=False,
):
    """Pull Drive artifacts; the default Mac workflow excludes training weights."""
    drive_project = discover_drive_project(drive_project)
    repository_root = Path(repository_root or REPOSITORY_ROOT).expanduser().resolve()
    drive_runs = drive_project / "runs"
    local_runs = repository_root / "runs"
    if not drive_runs.is_dir():
        if dry_run:
            logger.info("Drive runs folder does not exist yet: %s", drive_runs)
        else:
            drive_runs.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        local_runs.mkdir(parents=True, exist_ok=True)
    state_path, state = load_sync_state(local_runs)
    state_section = "full" if include_training_artifacts else "analysis"
    run_states = state[state_section]

    result = {
        "drive_project": str(drive_project),
        "repository_root": str(repository_root),
        "copied_run_files": 0,
        "copied_report_files": 0,
        "changed_runs": [],
        "skipped_running": [],
        "skipped_invalid": [],
        "skipped_unchanged": [],
        "training_artifacts_included": include_training_artifacts,
    }
    auxiliary_names = {"comparisons", "logs", "manual_baseline_videos"}
    drive_entries = drive_runs.iterdir() if drive_runs.is_dir() else []
    for source in sorted(drive_entries, key=lambda path: path.name):
        if source.name in SYNC_IGNORED_NAMES:
            continue
        if source.is_file():
            if (
                not include_training_artifacts
                and source.suffix.lower() in TRAINING_ARTIFACT_SUFFIXES
            ):
                continue
            if source.name == "experiment_manifest.jsonl" or (
                source.name.startswith("latest_") and source.suffix == ".txt"
            ):
                # These contain destination-specific paths and are rebuilt below.
                continue
            if copy_artifact_file(source, local_runs / source.name, dry_run=dry_run):
                result["copied_run_files"] += 1
            continue
        if not source.is_dir():
            continue
        if source.name in auxiliary_names:
            result["copied_run_files"] += merge_artifact_directory(
                source,
                local_runs / source.name,
                dry_run=dry_run,
                verify_text_contents=verify_text_contents,
                include_training_artifacts=include_training_artifacts,
            )
            continue
        signature = run_signature(source)
        if signature is None:
            result["skipped_invalid"].append(source.name)
            continue
        destination = local_runs / source.name
        if destination.is_dir() and run_states.get(source.name) == signature:
            result["skipped_unchanged"].append(source.name)
            continue
        metadata = read_run_metadata(source)
        if metadata is None:
            result["skipped_invalid"].append(source.name)
            continue
        if metadata.get("status") == "running" and not include_running:
            result["skipped_running"].append(source.name)
            continue
        changed = merge_artifact_directory(
            source,
            destination,
            dry_run=dry_run,
            verify_text_contents=verify_text_contents,
            include_training_artifacts=include_training_artifacts,
        )
        result["copied_run_files"] += changed
        if changed:
            result["changed_runs"].append(source.name)
        run_states[source.name] = signature

    legacy_comparisons = drive_project / "comparisons"
    if legacy_comparisons.is_dir():
        result["copied_run_files"] += merge_artifact_directory(
            legacy_comparisons,
            local_runs / "comparisons",
            dry_run=dry_run,
            verify_text_contents=verify_text_contents,
            include_training_artifacts=include_training_artifacts,
        )
    result["copied_report_files"] = _sync_report_artifacts(
        drive_project / "reports", repository_root / "reports", dry_run=dry_run
    )
    pointers = refresh_artifact_index(local_runs, portable=True, dry_run=dry_run)
    if not dry_run:
        save_sync_state(state_path, state)
    result["latest_pointers"] = {name: str(path) for name, path in pointers.items()}
    logger.info(
        "Drive pull updated %d run files and %d report files.",
        result["copied_run_files"],
        result["copied_report_files"],
    )
    return result


def sync_run_to_drive(run_dir, drive_project=None, dry_run=False):
    """Persist one finished run immediately, including failed-run diagnostics."""
    run_dir = Path(run_dir).expanduser().resolve()
    metadata = read_run_metadata(run_dir)
    if metadata is None:
        raise FileNotFoundError(f"Run metadata was not found or is invalid: {run_dir}")
    if metadata.get("status") not in {"complete", "failed"}:
        raise RuntimeError(f"Refusing to sync unfinished run: {run_dir}")
    drive_project = discover_drive_project(drive_project)
    destination = drive_project / "runs" / run_dir.name
    copied = merge_artifact_directory(
        run_dir,
        destination,
        dry_run=dry_run,
        verify_text_contents=True,
    )
    refresh_artifact_index(drive_project / "runs", portable=True, dry_run=dry_run)
    logger.info("Persisted %s to Google Drive (%d files updated).", run_dir.name, copied)
    return {"destination": str(destination), "copied_files": copied, "status": metadata["status"]}


def sync_auxiliary_artifacts_to_drive(
    drive_project=None,
    repository_root=None,
    include_reports=True,
    dry_run=False,
):
    """Persist comparison, manual-video, and generated-report artifacts."""
    drive_project = discover_drive_project(drive_project)
    repository_root = Path(repository_root or REPOSITORY_ROOT).expanduser().resolve()
    local_runs = repository_root / "runs"
    drive_runs = drive_project / "runs"
    result = {
        "comparison_files": 0,
        "manual_video_files": 0,
        "report_files": 0,
        "sync_log_files": 0,
    }
    for name, result_key in [
        ("comparisons", "comparison_files"),
        ("manual_baseline_videos", "manual_video_files"),
        ("logs", "sync_log_files"),
    ]:
        source = local_runs / name
        if source.is_dir():
            result[result_key] = merge_artifact_directory(
                source,
                drive_runs / name,
                dry_run=dry_run,
                verify_text_contents=True,
            )
    if include_reports:
        result["report_files"] = _sync_report_artifacts(
            repository_root / "reports", drive_project / "reports", dry_run=dry_run
        )
    logger.info(
        "Persisted auxiliary artifacts: %d comparisons, %d manual-video files, "
        "%d reports, %d sync logs.",
        result["comparison_files"],
        result["manual_video_files"],
        result["report_files"],
        result["sync_log_files"],
    )
    return result


def sync_workspace_to_drive(drive_project=None, repository_root=None, dry_run=False):
    """Safety-net sync for every finished local run plus generated artifacts."""
    drive_project = discover_drive_project(drive_project)
    repository_root = Path(repository_root or REPOSITORY_ROOT).expanduser().resolve()
    runs_root = repository_root / "runs"
    synced_runs = []
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir(), key=lambda path: path.name):
            metadata = read_run_metadata(run_dir) if run_dir.is_dir() else None
            if metadata and metadata.get("status") in {"complete", "failed"}:
                sync_run_to_drive(run_dir, drive_project, dry_run=dry_run)
                synced_runs.append(run_dir.name)
    auxiliary = sync_auxiliary_artifacts_to_drive(
        drive_project,
        repository_root,
        include_reports=True,
        dry_run=dry_run,
    )
    return {"synced_runs": synced_runs, "auxiliary": auxiliary}
