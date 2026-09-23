from __future__ import annotations

import threading
from contextlib import contextmanager
from enum import StrEnum

from vast_agent.jobs.cancellation import CancellationToken


class LockClass(StrEnum):
    READ = "READ"
    HEAVY_READ = "HEAVY_READ"
    WRITE = "WRITE"


class LockAcquisitionCancelled(RuntimeError):
    """Raised when a queued host lock is cancelled before it is acquired."""


class HostLockManager:
    """Writer-preferring per-host RW lock; heavy reads additionally serialize."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._states: dict[str, _HostState] = {}

    @contextmanager
    def acquire(
        self, host: str, lock_class: LockClass,
        cancellation: CancellationToken | None = None,
    ):
        with self._guard:
            state = self._states.setdefault(host.casefold(), _HostState())
        state.acquire(lock_class, cancellation)
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

    def acquire(self, kind: LockClass, cancellation: CancellationToken | None = None) -> None:
        def wake_waiter() -> None:
            with self.condition:
                self.condition.notify_all()

        if cancellation:
            cancellation.register(wake_waiter)
        with self.condition:
            try:
                def cancelled() -> bool:
                    return bool(cancellation and cancellation.cancelled)

                if kind == LockClass.WRITE:
                    self.waiting_writers += 1
                    try:
                        self.condition.wait_for(
                            lambda: cancelled() or (
                                not self.writer and not self.heavy and self.readers == 0
                            )
                        )
                        if cancelled():
                            raise LockAcquisitionCancelled
                        self.writer = True
                    finally:
                        self.waiting_writers -= 1
                        self.condition.notify_all()
                elif kind == LockClass.HEAVY_READ:
                    self.condition.wait_for(
                        lambda: cancelled() or (
                            not self.writer and not self.heavy and not self.waiting_writers
                        )
                    )
                    if cancelled():
                        raise LockAcquisitionCancelled
                    self.heavy = True
                    self.readers += 1
                else:
                    self.condition.wait_for(
                        lambda: cancelled() or (not self.writer and not self.waiting_writers)
                    )
                    if cancelled():
                        raise LockAcquisitionCancelled
                    self.readers += 1
            finally:
                if cancellation:
                    cancellation.unregister(wake_waiter)

    def release(self, kind: LockClass) -> None:
        with self.condition:
            if kind == LockClass.WRITE:
                self.writer = False
            else:
                self.readers -= 1
                if kind == LockClass.HEAVY_READ:
                    self.heavy = False
            self.condition.notify_all()
