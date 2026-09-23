from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vast_agent.config import HostRegistry
from vast_agent.execution.argv import ArgvRejected, validate_read_argv
from vast_agent.execution.base import Executor, redact


class ReadHostArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str
    argv: list[str] = Field(min_length=1, max_length=64)
    timeout: int = Field(default=20, ge=1, le=120)


FUNCTION_DECLARATIONS = [
    {
        "type": "function",
        "name": "read_host",
        "description": (
            "Run one non-mutating command on one configured host. "
            "Provide argv as separate tokens. Never use shell strings, pipes, redirects, eval, sh -c, or bash -c."
        ),
        "parameters": ReadHostArgs.model_json_schema(),
    },
]


class FunctionExecutor:
    """Mechanical READ boundary. It never decides what command is relevant."""

    def __init__(
        self,
        registry: HostRegistry,
        remote: Executor,
        max_chars: int = 2500,
        secrets: Sequence[str] = (),
    ) -> None:
        self.registry = registry
        self.remote = remote
        self.max_chars = max_chars
        self.secrets = tuple(secrets)

    def execute(
        self,
        name: str,
        arguments: dict[str, object],
        cancellation=None,
    ) -> dict[str, object]:
        if name != "read_host":
            return {"error": "FUNCTION_NOT_ALLOWED", "function": name}

        try:
            args = ReadHostArgs.model_validate(arguments)
        except ValidationError as exc:
            return {"error": "INVALID_ARGUMENTS", "details": exc.errors(include_input=False)}

        try:
            host = self.registry.resolve(args.host)
        except KeyError:
            return {"error": "HOST_NOT_FOUND", "host": args.host}
        if not host.enabled:
            return {"error": "HOST_DISABLED", "host": host.name}

        try:
            command = validate_read_argv(args.argv).argv
        except ArgvRejected as exc:
            return {"error": "READ_BOUNDARY_REJECTED", "reason": str(exc)}

        result = self.remote.execute(host, command, args.timeout, cancellation)
        return {
            "host": host.name,
            "argv": list(command),
            "success": result.success,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "error_code": result.error_code,
            "stdout": self._compact(result.stdout),
            "stderr": self._compact(result.stderr),
        }

    def _compact(self, value: str) -> str:
        text = redact(value, self.secrets)
        if len(text) <= self.max_chars:
            return text
        # Keep both context and the latest lines without trying to interpret meaning.
        head = self.max_chars // 3
        tail = self.max_chars - head
        return text[:head] + "\n…[truncated]…\n" + text[-tail:]
