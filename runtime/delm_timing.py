"""Convert worker timestamps to the shared run clock."""

import math
import json
import os
from pathlib import Path
import time


def require_number(record, key):
    value = record.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Missing or invalid timestamp: {key}")
    return value


def task_start(record, run_started_ns):
    """Convert the session's monotonic clock to seconds since the run began."""
    started_ns = require_number(record, "started_ns")
    if type(started_ns) is not int:
        raise ValueError("started_ns must be an integer")
    start = (started_ns - run_started_ns) / 1e9 + require_number(record, "model_turn_start_seconds")
    if start < 0:
        raise ValueError("Task start precedes the run clock")
    return start


def await_release():
    """Optional controller barrier; existing per-worker budgets remain the default."""
    timeout = float(os.environ["AGENT_TIMEOUT_SECONDS"])
    if os.environ.get("DELM_TIMING") != "synchronized":
        return timeout
    print(json.dumps({"event": "ready"}), flush=True)
    path = Path("/records/release.json")
    limit = time.monotonic() + 120
    while not path.exists():
        if time.monotonic() >= limit:
            raise RuntimeError("Controller did not release prepared workers")
        time.sleep(0.02)
    remaining = (json.loads(path.read_text())["deadline_ns"] - time.monotonic_ns()) / 1e9
    if remaining <= 0:
        raise RuntimeError("Common deadline already elapsed")
    return remaining
