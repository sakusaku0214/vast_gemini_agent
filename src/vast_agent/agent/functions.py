from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vast_agent.approval import ProposalStore
from vast_agent.config import HostRegistry
from vast_agent.execution.argv import (
    ArgvRejected,
    validate_read_argv,
    validate_write_argv,
)
from vast_agent.execution.base import Executor, redact


class ReadHostArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str
    argv: list[str] = Field(min_length=1, max_length=64)
    timeout: int = Field(default=20, ge=1, le=120)


class ProposeWriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str
    argv: list[str] = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=500)
    timeout: int = Field(default=30, ge=1, le=120)


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
    {
        "type": "function",
        "name": "propose_write",
        "description": (
            "Propose, but do not execute, one mutating command on one configured host. "
            "Use this only when a WRITE is required. The human OWNER must approve the stored proposal before execution. "
            "Provide argv as separate tokens; never use shell strings, pipes, redirects, eval, sh -c, or bash -c."
        ),
        "parameters": ProposeWriteArgs.model_json_schema(),
    },
]


class FunctionExecutor:
    """Mechanical boundary. Gemini chooses operations; human authorization controls WRITE."""

    def __init__(
        self,
        registry: HostRegistry,
        remote: Executor,
        proposals: ProposalStore,
        max_chars: int = 2500,
        secrets: Sequence[str] = (),
    ) -> None:
        self.registry = registry
        self.remote = remote
        self.proposals = proposals
        self.max_chars = max_chars
        self.secrets = tuple(secrets)

    def execute(
        self,
        name: str,
        arguments: dict[str, object],
        cancellation=None,
        owner_id: str = "cli",
        channel_id: str = "cli",
    ) -> dict[str, object]:
        if name == "read_host":
            return self._read(arguments, cancellation)
        if name == "propose_write":
            return self._propose(arguments, owner_id, channel_id)
        return {"error": "FUNCTION_NOT_ALLOWED", "function": name}

    def _read(self, arguments: dict[str, object], cancellation) -> dict[str, object]:
        try:
            args = ReadHostArgs.model_validate(arguments)
        except ValidationError as exc:
            return {"error": "INVALID_ARGUMENTS", "details": exc.errors(include_input=False)}

        host = self._host(args.host)
        if isinstance(host, dict):
            return host

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

    def _propose(
        self,
        arguments: dict[str, object],
        owner_id: str,
        channel_id: str,
    ) -> dict[str, object]:
        try:
            args = ProposeWriteArgs.model_validate(arguments)
        except ValidationError as exc:
            return {"error": "INVALID_ARGUMENTS", "details": exc.errors(include_input=False)}

        host = self._host(args.host)
        if isinstance(host, dict):
            return host

        try:
            command = validate_write_argv(args.argv).argv
        except ArgvRejected as exc:
            return {"error": "WRITE_BOUNDARY_REJECTED", "reason": str(exc)}

        proposal = self.proposals.create(
            owner_id=str(owner_id),
            channel_id=str(channel_id),
            host=host.name,
            argv=command,
            reason=redact(args.reason, self.secrets),
            timeout=args.timeout,
        )
        return {
            "proposal_id": proposal.id,
            "status": proposal.status,
            "host": proposal.host,
            "argv": list(proposal.argv),
            "reason": proposal.reason,
            "instruction": f"OWNER approval required: 承認 #{proposal.id}",
        }

    def _host(self, value: str):
        try:
            host = self.registry.resolve(value)
        except KeyError:
            return {"error": "HOST_NOT_FOUND", "host": value}
        if not host.enabled:
            return {"error": "HOST_DISABLED", "host": host.name}
        return host

    def _compact(self, value: str) -> str:
        text = redact(value, self.secrets)
        if len(text) <= self.max_chars:
            return text
        head = self.max_chars // 3
        tail = self.max_chars - head
        return text[:head] + "\n…[truncated]…\n" + text[-tail:]
