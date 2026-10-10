"""Unified logging abstraction (wandb + local CSV)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class Logger:
    """Thin wrapper over wandb + a local JSONL log file."""

    def __init__(self, run_dir: str | Path, project: str = "cce", use_wandb: bool = True,
                 run_name: str | None = None, config: dict | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.run_dir / "metrics.jsonl"
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_writer: csv.DictWriter | None = None
        self._csv_file: Any = None
        self._wandb = None

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                self._wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    dir=str(self.run_dir.parent),
                    reinit=True,
                )
            except ImportError:
                self._wandb = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a flat dict of scalars."""
        # Append to JSONL
        record = {"step": step, **metrics}
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Append to CSV (lazy header init)
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            fieldnames = ["step"] + list(metrics.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(record)
        self._csv_file.flush()

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log final summary (one-shot, no step axis)."""
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        if self._wandb is not None:
            self._wandb.summary.update(summary)

    def finish(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        if self._wandb is not None:
            self._wandb.finish()

    def save_model(self, state: dict, name: str) -> None:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        import torch
        torch.save(state, ckpt_dir / name)
