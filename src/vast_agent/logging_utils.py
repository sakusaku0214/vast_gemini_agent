from __future__ import annotations

import logging

from vast_agent.execution.base import redact


class SecretRedactingFormatter(logging.Formatter):
    """Redact configured values after exception tracebacks have been formatted."""

    def __init__(self, fmt: str, secrets: tuple[str, ...]) -> None:
        super().__init__(fmt)
        self.secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record), self.secrets)


def configure_agent_logging(log_file, secrets: tuple[str, ...]) -> None:
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(SecretRedactingFormatter(
        "%(asctime)s %(levelname)s component=%(name)s %(message)s", secrets,
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
