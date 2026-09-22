from __future__ import annotations

import threading
from contextlib import contextmanager
from enum import StrEnum


class LockClass(StrEnum):
    READ = "READ"
    HEAVY_READ = "HEAVY_READ"
    WRITE = "WRITE"


class HostLockManager:
    """Per-host coordination; ordinary reads coexist while heavy reads serialize."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._heavy: dict[str, threading.Lock] = {}

    @contextmanager
    def acquire(self, host: str, lock_class: LockClass):
        if lock_class == LockClass.READ:
            yield
            return
        with self._guard:
            lock = self._heavy.setdefault(host.casefold(), threading.Lock())
        with lock:
            yield
