from __future__ import annotations

import threading
from contextlib import contextmanager
from enum import StrEnum


class LockClass(StrEnum):
    READ = "READ"
    HEAVY_READ = "HEAVY_READ"
    WRITE = "WRITE"


class HostLockManager:
    """Writer-preferring per-host RW lock; heavy reads additionally serialize."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._states: dict[str, _HostState] = {}

    @contextmanager
    def acquire(self, host: str, lock_class: LockClass):
        with self._guard:
            state = self._states.setdefault(host.casefold(), _HostState())
        state.acquire(lock_class)
        try:
            yield
        finally:
            state.release(lock_class)


class _HostState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.readers = 0
        self.heavy = False
        self.writer = False
        self.waiting_writers = 0

    def acquire(self, kind: LockClass) -> None:
        with self.condition:
            if kind == LockClass.WRITE:
                self.waiting_writers += 1
                try:
                    self.condition.wait_for(lambda: not self.writer and not self.heavy and self.readers == 0)
                    self.writer = True
                finally:
                    self.waiting_writers -= 1
            elif kind == LockClass.HEAVY_READ:
                self.condition.wait_for(lambda: not self.writer and not self.heavy and not self.waiting_writers)
                self.heavy = True
                self.readers += 1
            else:
                self.condition.wait_for(lambda: not self.writer and not self.waiting_writers)
                self.readers += 1

    def release(self, kind: LockClass) -> None:
        with self.condition:
            if kind == LockClass.WRITE:
                self.writer = False
            else:
                self.readers -= 1
                if kind == LockClass.HEAVY_READ:
                    self.heavy = False
            self.condition.notify_all()
