from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field

from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult


class RebootResult(StrEnum):
    DISPATCH_FAILED = "DISPATCH_FAILED"
    SUCCEEDED = "SUCCEEDED"
    REBOOT_NOT_OBSERVED = "REBOOT_NOT_OBSERVED"
    REBOOT_RECOVERY_FAILED = "REBOOT_RECOVERY_FAILED"
    REBOOT_COMPLETED_BUT_FAULT_REMAINS = "REBOOT_COMPLETED_BUT_FAULT_REMAINS"


class RebootTrace(BaseModel):
    result: RebootResult
    states: list[str] = Field(default_factory=list)
    summary: str


class RebootRemote(Protocol):
    def dispatch(self, host: Host) -> ToolResult: ...
    def reachable(self, host: Host) -> bool: ...
    def postcheck(self, host: Host) -> bool: ...


class RebootCoordinator:
    """Dedicated one-shot reboot and down/up recovery monitor; no power fallback exists."""

    def __init__(self, remote: RebootRemote, timeout: int = 300, poll_seconds: int = 10,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.remote, self.timeout, self.poll_seconds = remote, timeout, poll_seconds
        self.clock, self.sleep = clock, sleep

    def execute_approved(self, host: Host) -> RebootTrace:
        states = ["COMMAND_DISPATCHED"]
        # Disconnect/non-zero after dispatch is not enough to conclude that dispatch failed.
        dispatch = self.remote.dispatch(host)  # exactly once; point of no return
        if not dispatch.success and self._definitely_not_dispatched(dispatch):
            return RebootTrace(result=RebootResult.DISPATCH_FAILED, states=[],
                               summary="Reboot command was not dispatched")
        deadline = self.clock() + self.timeout
        down = False
        while self.clock() < deadline:
            reachable = self.remote.reachable(host)
            if not reachable and not down:
                down = True; states.append("DOWN_OBSERVED")
            elif reachable and down:
                states.extend(("UP_OBSERVED", "VERIFYING"))
                clean = self.remote.postcheck(host)
                if clean:
                    states.append("POSTCHECK_OK")
                    return RebootTrace(result=RebootResult.SUCCEEDED, states=states,
                                       summary="Host rebooted and postcheck passed")
                return RebootTrace(result=RebootResult.REBOOT_COMPLETED_BUT_FAULT_REMAINS,
                                   states=states, summary="Host returned but fault remains")
            self.sleep(self.poll_seconds)
        if not down:
            return RebootTrace(result=RebootResult.REBOOT_NOT_OBSERVED, states=states,
                               summary="Host never became unreachable")
        return RebootTrace(result=RebootResult.REBOOT_RECOVERY_FAILED, states=states,
                           summary="SSH did not recover; physical handling may be required; "
                                   "automated power recovery was not attempted")

    @staticmethod
    def _definitely_not_dispatched(result: ToolResult) -> bool:
        if result.success or result.timed_out:
            return False
        if result.error_code and result.error_code.value in {
            "SSH_HOST_KEY_UNKNOWN", "SSH_HOST_KEY_CHANGED",
        }:
            return True
        message = result.stderr.casefold()
        return any(token in message for token in (
            "permission denied", "sudo:", "could not resolve hostname", "connection refused",
            "no such file", "not found",
        ))


class SSHRebootRemote:
    """Adapter intentionally exposes only the single code-owned reboot argv."""

    def __init__(self, executor, reachable, postcheck, timeout: int = 30) -> None:
        self.executor, self._reachable, self._postcheck, self.timeout = executor, reachable, postcheck, timeout

    def dispatch(self, host: Host) -> ToolResult:
        return self.executor.execute(host, ("sudo", "-n", "systemctl", "reboot"), self.timeout)

    def reachable(self, host: Host) -> bool:
        return self._reachable(host)

    def postcheck(self, host: Host) -> bool:
        return self._postcheck(host)
