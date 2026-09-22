from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from vast_agent.models.host import Host
from vast_agent.models.tool_result import ErrorCode, ToolResult


class Executor(Protocol):
    def execute(self, host: Host, command: Sequence[str], timeout: int) -> ToolResult: ...


class FakeExecutor:
    """Deterministic executor whose fixtures are keyed by registry tool name."""

    def __init__(self, results: Mapping[str, ToolResult]) -> None:
        self.results = results

    def execute(self, host: Host, command: Sequence[str], timeout: int) -> ToolResult:
        del host, command, timeout
        return ToolResult(
            success=False, exit_code=127, stderr="FakeExecutor requires a tool name", duration_ms=0,
            error_code=ErrorCode.COMMAND_FAILED,
        )

    def execute_tool(
        self, name: str, host: Host, command: Sequence[str], timeout: int,
    ) -> ToolResult:
        del host, command, timeout
        return self.results.get(
            name,
            ToolResult(
                success=False, exit_code=127, stderr=f"No fixture for {name}", duration_ms=0,
                error_code=ErrorCode.COMMAND_FAILED,
            ),
        )


def redact(text: str, secrets: Sequence[str] = ()) -> str:
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    return cleaned
