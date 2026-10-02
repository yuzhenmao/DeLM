"""Convert worker timestamps to the shared run clock."""

import math


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
