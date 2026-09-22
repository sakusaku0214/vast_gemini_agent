from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from vast_agent.execution.base import redact
from vast_agent.jobs.cancellation import CancellationToken
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ErrorCode, ToolResult


class SSHExecutor:
    def __init__(self, known_hosts: Path, ssh_executable: str = "ssh") -> None:
        self.known_hosts = known_hosts.resolve()
        self.ssh_executable = ssh_executable

    def execute(self, host: Host, command: Sequence[str], timeout: int,
                cancellation: CancellationToken | None = None) -> ToolResult:
        # The remote command consists only of tokens owned by registered tools.
        args = [
            self.ssh_executable,
            "-p", str(host.ssh_port),
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={min(timeout, 30)}",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            "-o", f"UserKnownHostsFile={openssh_path(self.known_hosts)}",
            "-o", "StrictHostKeyChecking=yes",
            "--", f"{host.ssh_user}@{host.address}",
            *command,
        ]
        started = time.monotonic()
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            def terminate() -> None:
                if proc.poll() is None:
                    proc.terminate()
            if cancellation:
                cancellation.register(terminate)
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate()
                try: stdout_bytes, stderr_bytes = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill(); stdout_bytes, stderr_bytes = proc.communicate()
                return ToolResult(
                    success=False, stdout=redact(_decode(stdout_bytes)),
                    stderr=redact(_decode(stderr_bytes)),
                    duration_ms=int((time.monotonic() - started) * 1000), timed_out=True,
                    error_code=ErrorCode.SSH_TIMEOUT,
                )
            finally:
                if cancellation:
                    cancellation.unregister(terminate)
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            if cancellation and cancellation.cancelled:
                return ToolResult(
                    success=False, exit_code=proc.returncode, stdout=redact(stdout),
                    stderr=redact(stderr), duration_ms=int((time.monotonic() - started) * 1000),
                    error_code=ErrorCode.CANCELLED,
                )
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
        except OSError as exc:
            return ToolResult(
                success=False, stderr=str(exc), duration_ms=int((time.monotonic() - started) * 1000),
                error_code=ErrorCode.COMMAND_FAILED,
            )


def _decode(value: bytes | str | None) -> str:
    if value is None: return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def openssh_path(path: str | PathLike[str], *, windows: bool | None = None) -> str:
    """Render one path as a safe value for an OpenSSH ``-o`` argument."""
    value = str(path)
    if any(character in value for character in ('"', "\r", "\n", "\x00")):
        raise ValueError("OpenSSH paths must not contain quotes or control characters")
    if windows if windows is not None else sys.platform == "win32":
        value = value.replace("\\", "/")
    return f'"{value}"' if any(character.isspace() for character in value) else value
