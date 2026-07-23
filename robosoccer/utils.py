"""Logging, reproducibility, normalization, and run-artifact utilities."""

import csv
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_name(text):
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in text)
    return cleaned.strip("-") or "run"


def json_safe(value):
    """Convert NumPy, Torch, and Path objects into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, data):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def setup_logging(run_dir=None, console_level="INFO", file_level="DEBUG", filename="train.log"):
    """Configure one concise console handler and one detailed file handler."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, str(console_level).upper()))
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)
    if run_dir is not None:
        log_path = Path(run_dir) / "logs" / filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, str(file_level).upper()))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root.addHandler(file_handler)
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    return root


def get_pyplot():
    """Load Matplotlib lazily with a writable, headless-safe cache and backend."""
    cache = Path(tempfile.gettempdir()) / "robosoccer-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    return pyplot


def set_global_seeds(seed, deterministic_torch=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic_torch), warn_only=True)
    if deterministic_torch:
        logger.warning("Deterministic Torch algorithms are enabled and may reduce throughput.")


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def package_versions():
    names = [
        "torch",
        "numpy",
        "gymnasium",
        "pettingzoo",
        "pymunk",
        "PyYAML",
        "pandas",
        "matplotlib",
        "tqdm",
        "tensorboard",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_directory(config, run_name=None, method=None):
    root = Path(config["experiment"].get("output_dir", "runs")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = safe_name(run_name or config["experiment"]["name"])
    algorithm = safe_name(method or config["ppo"].get("algorithm", "method"))
    seed = int(config["experiment"]["seed"])
    stem = f"{timestamp}_{experiment}_{algorithm}_seed{seed}"
    run_dir = root / stem
    counter = 1
    while run_dir.exists():
        run_dir = root / f"{stem}_{counter}"
        counter += 1
    for relative in ["checkpoints", "models", "logs/tensorboard", "eval", "plots", "videos"]:
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def initial_metadata(config, run_dir, source_config, parsed_args, command=None):
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except RuntimeError:
            gpu_name = "unavailable"
    return {
        "status": "running",
        "utc_start": utc_now(),
        "utc_completion": None,
        "command": command or " ".join(sys.argv),
        "parsed_cli_arguments": parsed_args or {},
        "source_configuration_path": str(Path(source_config).resolve()) if source_config else None,
        "resolved_configuration": config,
        "run_directory": str(Path(run_dir).resolve()),
        "experiment_name": config["experiment"]["name"],
        "algorithm": config["ppo"].get("algorithm"),
        "randomization_mode": config["randomization"].get("mode"),
        "seed": config["experiment"]["seed"],
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "training_step_counts": {},
        "evaluation_seeds": config.get("evaluation", {}).get("seed_bases", {}),
        "output_artifact_paths": {},
        "failure_exception": None,
    }


def finalize_run(config, run_dir, metadata):
    """Write the latest pointer and append a complete run to the JSONL manifest."""
    root = Path(config["experiment"].get("output_dir", "runs")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = safe_name(config["experiment"]["name"])
    output_setting = str(config["experiment"].get("output_dir", "runs"))
    pointer_value = (
        "runs/" + Path(run_dir).name
        if output_setting == "runs"
        else str(Path(run_dir).resolve())
    )
    (root / ("latest_" + name + ".txt")).write_text(pointer_value + "\n", encoding="utf-8")
    with (root / "experiment_manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(metadata), sort_keys=True) + "\n")


class RunningMeanStd:
    """Numerically stable running moments for vector observations or scalar returns."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        if values.ndim == self.mean.ndim:
            values = np.expand_dims(values, 0)
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        first = self.var * self.count
        second = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = (first + second + correction) / total
        self.count = total

    def normalize(self, values, clip=10.0):
        normalized = (np.asarray(values) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class MetricsWriter:
    """Append scalar rows to CSV and mirror numeric values to TensorBoard."""

    def __init__(self, run_dir, enabled=True):
        self.path = Path(run_dir) / "logs" / "metrics.csv"
        self.fieldnames = None
        self.tensorboard = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard = SummaryWriter(Path(run_dir) / "logs" / "tensorboard")
            except ImportError:
                logger.warning("TensorBoard is unavailable; CSV metrics remain enabled.")

    def write(self, row):
        clean = {key: json_safe(value) for key, value in row.items()}
        if self.fieldnames is None:
            self.fieldnames = list(clean)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
        missing = [key for key in self.fieldnames if key not in clean]
        for key in missing:
            clean[key] = ""
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(clean)
        if self.tensorboard is not None:
            step = int(clean.get("environment_steps", clean.get("update", 0)))
            for key, value in clean.items():
                if key not in {"environment_steps", "update"} and isinstance(value, int | float):
                    self.tensorboard.add_scalar("train/" + key, value, step)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()


def check_finite(name, values):
    if not np.all(np.isfinite(np.asarray(values))):
        raise FloatingPointError(name + " contains NaN or infinity")


def process_id():
    return os.getpid()
