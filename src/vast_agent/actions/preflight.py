from __future__ import annotations

from typing import Protocol

from vast_agent.actions.models import ActionRequest, PreflightSnapshot
from vast_agent.models.host import Host


class PreflightProvider(Protocol):
    """READ-only evidence provider. Implementations may only call registered read commands."""

    def collect(self, host: Host, request: ActionRequest) -> PreflightSnapshot: ...
