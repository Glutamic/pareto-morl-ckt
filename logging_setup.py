"""Logging infrastructure for MORL circuit sizing training.

Provides:
- JSONLHandler: writes structured JSON events to metrics.jsonl
- get_run_dir: creates timestamped run directories
- setup_logging: configures root logger with console + file handlers
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


class JSONLHandler(logging.Handler):
    """Writes log records with a `json_event` extra to a JSONL file."""

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.setLevel(logging.INFO)

    def emit(self, record):
        if not hasattr(record, "json_event"):
            return
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
        }
        entry.update(record.json_event)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


class _EnvFilter(logging.Filter):
    """Only passes records from the morl_ckt.env logger."""

    def filter(self, record):
        return record.name == "morl_ckt.env"


def get_run_dir(base_dir, env_name):
    """Create a timestamped run directory.

    Returns:
        Path to the run directory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{ts}_{env_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir, config, console_level="INFO"):
    """Configure root logger with handlers for console, train.log, env.log, and metrics.jsonl.

    Args:
        run_dir: Path to the run directory.
        config: Dict of training configuration (for config.yaml snapshot).
        console_level: "INFO" (default) or "DEBUG" for verbose console output.

    Returns:
        Dict with "jsonl" (JSONLHandler) and "run_dir" (Path).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any pre-existing handlers
    root.handlers.clear()

    # --- Console handler (INFO or DEBUG) ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(console)

    # --- train.log: INFO+ for all modules ---
    train_fh = logging.FileHandler(run_dir / "train.log")
    train_fh.setLevel(logging.INFO)
    train_fh.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(train_fh)

    # --- env.log: DEBUG for morl_ckt.env only ---
    env_fh = logging.FileHandler(run_dir / "env.log")
    env_fh.setLevel(logging.DEBUG)
    env_fh.addFilter(_EnvFilter())
    env_fh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(env_fh)

    # --- metrics.jsonl: structured events ---
    jsonl_h = JSONLHandler(run_dir / "metrics.jsonl")
    root.addHandler(jsonl_h)

    # --- Write config snapshot ---
    try:
        import yaml
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except ImportError:
        pass

    return {"jsonl": jsonl_h, "run_dir": run_dir}
