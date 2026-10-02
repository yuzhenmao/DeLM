"""Allow one experiment per machine user."""
import fcntl
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def machine_lock():
    with (Path.home() / ".delm-worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another experiment holds the worker lock") from None
        yield
