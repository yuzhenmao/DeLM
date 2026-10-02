"""Read model transport failures from the native agent's diagnostic log."""

import json
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path


def transport_records(database):
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            "SELECT id, ts, ts_nanos, level, target, feedback_log_body AS message "
            "FROM logs WHERE level IN ('WARN', 'ERROR') AND "
            "(target = 'codex_core::responses_retry' OR "
            "(target = 'codex_core::client' AND "
            "feedback_log_body LIKE '%falling back to HTTP%')) ORDER BY id"
        )]


def collect_transport(container, worker):
    """Called after the native process stops; never read credentials or tool logs."""
    with tempfile.TemporaryDirectory(prefix="delm-network-") as temporary:
        database = Path(temporary) / "logs_2.sqlite"
        source = f"{container}:/root/.codex/logs_2.sqlite"
        subprocess.run(["docker", "cp", source, str(database)], check=True,
                       capture_output=True, text=True, timeout=30)
        wal = subprocess.run(["docker", "cp", source + "-wal", str(database) + "-wal"],
                             capture_output=True, text=True, timeout=30)
        if wal.returncode and "Could not find the file" not in wal.stderr:
            raise RuntimeError("Could not read the native diagnostic write log")
        records = transport_records(database)
    report = dict(source="/root/.codex/logs_2.sqlite", records=records,
                  transport_failures=len(records))
    (worker / "model-transport.json").write_text(json.dumps(report, indent=2))
    return len(records)
