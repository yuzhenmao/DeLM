"""Read current run records without changing experiment data."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".local/runs"


@dataclass
class Run:
    folder: Path
    record: dict
    error: str | None = None

    @property
    def summary(self):
        # Failed cleanup can invalidate an otherwise complete experiment.
        value = self.record.get("summary")
        return value if self.record.get("status") == "finished" and isinstance(value, dict) else {}


def read_run(folder):
    folder = Path(folder)
    try:
        record = json.loads((folder / "run.json").read_text())
        if not isinstance(record, dict):
            raise ValueError("run.json must contain an object")
        workers = record.get("workers", [])
        if not isinstance(workers, list) or any(not isinstance(w, dict) for w in workers):
            raise ValueError("workers must contain objects")
        return Run(folder, record)
    except (OSError, ValueError) as error:
        return Run(folder, {}, f"{type(error).__name__}: {error}")


def scan_runs(root=RUNS):
    """Include started repeat folders even when their run.json is absent."""
    return [read_run(folder) for folder in sorted(Path(root).glob("*/*/repeat-*")) if folder.is_dir()]
