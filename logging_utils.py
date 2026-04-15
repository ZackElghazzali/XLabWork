"""
utils/logging_utils.py
Rule 2:  Every completed run must log its git commit hash, config,
         hardware, and final metrics atomically.
Rule 10: All metrics are (mean, std) tuples — never bare scalars.
Rule 12: No experiment left unnamed.
"""
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a YAML config file and validate required top-level keys.

    Every training script must load its config via this function rather than
    reading the file directly, so the key-presence check is never skipped.

    Raises:
        AssertionError: if any required key is missing or experiment.name is
                        empty (Rule 12).
    """
    config_path = Path(config_path)
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    required_top_level = [
        "experiment", "reproducibility", "data",
        "stage1", "stage2", "stage3", "evaluation", "logging",
    ]
    for key in required_top_level:
        assert key in config, (
            f"[config] Missing required top-level key '{key}' in {config_path}. "
            "Rule 2: every hyperparameter must live in the YAML config."
        )

    assert config["experiment"].get("name"), (
        f"[config] experiment.name is empty in {config_path}. "
        "Rule 12: every run must have a human-readable name."
    )

    logger.info(
        "[config] Loaded '%s' from %s",
        config["experiment"]["name"], config_path,
    )
    return config


# ── Hardware / git introspection ───────────────────────────────────────────────

def get_git_commit_hash() -> str:
    """Return the HEAD commit hash, or 'unknown' if not inside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("[logging] Could not retrieve git commit hash.")
        return "unknown"


def get_hardware_info() -> Dict[str, Any]:
    """Snapshot the hardware context for the run record."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    return info


# ── Atomic run logging ─────────────────────────────────────────────────────────

def log_run(
    run_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, Tuple[float, float]],
    log_dir: Path,
    notes: Optional[str] = None,
) -> Path:
    """
    Atomically write a complete run record to <log_dir>/<run_name>.json.

    Args:
        run_name: Human-readable name encoding key decisions (Rule 12).
                  e.g. "stage1_vae_beta0.01_dim512_seed42"
        config:   Full config dict (from load_config).
        metrics:  {metric_name: (mean, std)} — Rule 10 enforced here.
        log_dir:  Destination directory.
        notes:    Optional free-text annotation.

    Returns:
        Path of the written JSON file.

    Raises:
        AssertionError: if any metric value is not a 2-element tuple/list.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Rule 10 — hard check before writing anything
    for key, value in metrics.items():
        assert (
            isinstance(value, (tuple, list)) and len(value) == 2
        ), (
            f"[logging] Metric '{key}' must be a (mean, std) tuple. "
            f"Got {type(value).__name__}. "
            "Rule 10: uncertainty is a first-class output — never report bare scalars."
        )

    record = {
        "run_name": run_name,
        "git_commit": get_git_commit_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": get_hardware_info(),
        "config": config,
        "metrics": {k: {"mean": v[0], "std": v[1]} for k, v in metrics.items()},
        "notes": notes or "",
    }

    out_file = log_dir / f"{run_name}.json"
    tmp_file = out_file.with_suffix(".tmp")     # atomic: write then rename
    with open(tmp_file, "w") as fh:
        json.dump(record, fh, indent=2)
    tmp_file.rename(out_file)

    logger.info("[logging] Run record written → %s", out_file)
    return out_file
