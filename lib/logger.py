"""
Logger and environment variable utilities for Slurm-based experiment scripts.

Provides a dual-output logger (stdout + file), environment variable validation,
and path resolution for $PROJECT_ROOT / $OUTPUT_DIR set by slurm_entry.sh.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logger(output_dir: Path, name: str) -> logging.Logger:
    """
    Create a logger that writes to both stdout and output_dir/experiment.log.

    Idempotent: if a logger with the given name already has handlers, return it
    as-is to prevent duplicate log entries on repeated calls.

    Args:
        output_dir: directory where experiment.log is written (created if absent)
        name:       logger name (use the experiment ID, e.g. "0107_extract_triples")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def require_env(*names: str) -> dict[str, str]:
    """
    Assert that all named environment variables are set, and return their values.

    Prints a clear error message and calls sys.exit(1) if any are missing.

    Args:
        *names: environment variable names to check

    Returns:
        {name: value} dict for all requested variables
    """
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Error: Required env vars not set: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    return {n: os.environ[n] for n in names}


def get_project_root() -> Path:
    """Return $PROJECT_ROOT as a Path. Exits with error if not set."""
    val = os.environ.get("PROJECT_ROOT")
    if not val:
        print("Error: PROJECT_ROOT is not set", file=sys.stderr)
        sys.exit(1)
    return Path(val)


def get_output_dir() -> Path:
    """Return $OUTPUT_DIR as a Path. Exits with error if not set."""
    val = os.environ.get("OUTPUT_DIR")
    if not val:
        print("Error: OUTPUT_DIR is not set", file=sys.stderr)
        sys.exit(1)
    return Path(val)
