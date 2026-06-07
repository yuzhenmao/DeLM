import threading
from collections import deque
from typing import List, Optional
from dataclasses import dataclass, field

# ======================================================================
# Reader-writer lock
# ======================================================================

class RWLock:
    """Writer-preferring reader-writer lock.

    Multiple readers may hold the lock concurrently, but a writer has
    exclusive access. `with lock:` defaults to acquiring the WRITE lock
    (so existing `with shared_context.lock:` call sites still serialize
    mutations). Use `with lock.read():` / `with lock.write():` to be
    explicit.
    """
    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._write_waiting = 0

    def _acquire_read(self):
        with self._cond:
            while self._writer or self._write_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def _release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def _acquire_write(self):
        with self._cond:
            self._write_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._write_waiting -= 1

    def _release_write(self):
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    class _Ctx:
        def __init__(self, rw, write):
            self.rw = rw
            self.write = write
        def __enter__(self):
            if self.write:
                self.rw._acquire_write()
            else:
                self.rw._acquire_read()
            return self.rw
        def __exit__(self, exc_type, exc, tb):
            if self.write:
                self.rw._release_write()
            else:
                self.rw._release_read()

    def read(self):
        return RWLock._Ctx(self, write=False)

    def write(self):
        return RWLock._Ctx(self, write=True)

    # `with lock:` defaults to write lock (backward compat).
    def __enter__(self):
        self._acquire_write()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._release_write()


# ======================================================================
# Shared state
# ======================================================================

@dataclass
class SharedContext:
    """Ordered list of conclusion summaries, protected by a read/write lock."""
    entries: List[str] = field(default_factory=list)
    lock: RWLock = field(default_factory=RWLock)
    _token_count: int = field(default=0, repr=False)

    def snapshot(self) -> List[str]:
        with self.lock.read():
            return list(self.entries)

    def append(self, entry: str) -> None:
        with self.lock.write():
            self.entries.append(entry)
            self._token_count += self._estimate_tokens(entry)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars/token) for the running context size.
        Cheap heuristic — no tokenizer dependency."""
        return len(text) // 4


class TaskQueue:
    """Thread-safe FIFO of task description strings, using a read/write lock."""
    def __init__(self):
        self._dq = deque()
        self._lock = RWLock()

    def extend(self, items: List[str]) -> None:
        with self._lock.write():
            self._dq.extend(items)

    def pop(self) -> Optional[str]:
        with self._lock.write():
            if not self._dq:
                return None
            return self._dq.popleft()

    def snapshot(self) -> List[str]:
        with self._lock.read():
            return list(self._dq)

    def __len__(self) -> int:
        with self._lock.read():
            return len(self._dq)
