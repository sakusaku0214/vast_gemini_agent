from __future__ import annotations

import threading
from collections.abc import Callable
from contextvars import ContextVar


class CancellationToken:
    """Thread-safe cooperative cancellation plus process termination callbacks."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def register(self, callback) -> None:
        with self._lock:
            if self.cancelled:
                callback()
            else:
                self._callbacks.append(callback)

    def unregister(self, callback) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def cancel(self) -> bool:
        if self._event.is_set():
            return False
        self._event.set()
        with self._lock:
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback()
        return True


current_cancellation: ContextVar[CancellationToken | None] = ContextVar(
    "current_cancellation", default=None,
)
