from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from vast_agent.execution.base import redact
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ErrorCode, ToolResult


class SSHExecutor:
    def __init__(self, known_hosts: Path, ssh_executable: str = "ssh") -> None:
        self.known_hosts = known_hosts.resolve()
        self.ssh_executable = ssh_executable

    def execute(self, host: Host, command: Sequence[str], timeout: int) -> ToolResult:
        # The remote command consists only of tokens owned by registered tools.
        args = [
            self.ssh_executable,
            "-p", str(host.ssh_port),
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={min(timeout, 30)}",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
            "-o", "StrictHostKeyChecking=yes",
            "--", f"{host.ssh_user}@{host.address}",
            *command,
        ]
        started = time.monotonic()
        try:
            proc = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            code = None
            low = stderr.lower()
            if "host key verification failed" in low:
                code = ErrorCode.SSH_HOST_KEY_UNKNOWN
            if "remote host identification has changed" in low:
                code = ErrorCode.SSH_HOST_KEY_CHANGED
            if proc.returncode and code is None:
                code = ErrorCode.COMMAND_FAILED
            return ToolResult(
                success=proc.returncode == 0, exit_code=proc.returncode,
                stdout=redact(stdout), stderr=redact(stderr),
                duration_ms=int((time.monotonic() - started) * 1000), error_code=code,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                success=False, stdout=_decode(exc.stdout), stderr=_decode(exc.stderr),
                duration_ms=int((time.monotonic() - started) * 1000), timed_out=True,
                error_code=ErrorCode.SSH_TIMEOUT,
            )
        except OSError as exc:
            return ToolResult(
                success=False, stderr=str(exc), duration_ms=int((time.monotonic() - started) * 1000),
                error_code=ErrorCode.COMMAND_FAILED,
            )


def _decode(value: bytes | str | None) -> str:
    if value is None: return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")
