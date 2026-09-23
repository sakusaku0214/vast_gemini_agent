"""Structured escape hatch for operations not represented by a typed action.

An OperationPlan is inert data.  Creating one never grants execution authority;
the approval coordinator must persist its fingerprint, obtain OWNER approval,
recheck policy and acquire the host WRITE lock before dispatching it.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vast_agent.actions.models import ProposalStatus

_META = re.compile(r"(?:\|\||&&|>>|[|<>;`$*?\[\]{}~]|[\r\n\x00])")
_SHELLS = frozenset({
    "sh", "bash", "dash", "zsh", "fish", "eval", "sudo", "env", "python", "python3",
    "perl", "ruby", "node", "php", "awk", "xargs",
})
_DESTRUCTIVE = frozenset({
    "mkfs", "fdisk", "parted", "sfdisk", "wipefs", "flashrom", "dd", "rm",
})


class OperationPlan(BaseModel):
    """Exact, self-contained proposal payload; arbitrary environment is intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    host: str = Field(min_length=1, max_length=128)
    host_source: Literal["current message", "conversation context"]
    executable: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
    argv: list[str] = Field(min_length=1, max_length=32)
    requires_sudo: bool = False
    target: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=500)
    expected_effect: str = Field(min_length=1, max_length=500)
    known_side_effects: list[str] = Field(default_factory=list, max_length=8)
    verification_plan: str = Field(min_length=1, max_length=500)
    verification_kind: Literal["service_state", "gpu_state", "executable", "unavailable"]
    verification_target: str | None = Field(default=None, max_length=128)
    command_source: str = Field(min_length=1, max_length=200)
    rollback: str | None = Field(default=None, max_length=500)
    current_relevant_state: str = Field(default="unknown", max_length=500)
    active_workload: str = Field(default="unknown", max_length=500)
    running_vm: str = Field(default="unknown", max_length=500)

    @field_validator("argv")
    @classmethod
    def safe_argv(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 256 or _META.search(item) for item in value):
            raise ValueError("argv contains an empty, oversized, or shell-active argument")
        return value

    @model_validator(mode="after")
    def hard_safety_floor(self) -> OperationPlan:
        exe = self.executable.casefold()
        words = [item.casefold() for item in self.argv]
        if exe in _SHELLS:
            raise ValueError("shell, sudo, and environment wrappers are not operation executables")
        if exe in _DESTRUCTIVE or exe.startswith("mkfs."):
            raise ValueError("destructive storage/firmware operation is prohibited")
        joined = " ".join(words)
        if any(marker in joined for marker in (".ssh", "credentials", "shadow", "token", "secret")):
            raise ValueError("credential or secret access is prohibited")
        if ("disable" in words or "mask" in words) and any(
            marker in joined for marker in ("vast-agent", "vast_gemini", "approval", "safety")
        ):
            raise ValueError("disabling approval or safety mechanisms is prohibited")
        if self.host.casefold() in {"*", "all", "fleet", "全台"}:
            raise ValueError("implicit fleet operation is prohibited")
        if self.verification_kind != "unavailable" and not self.verification_target:
            raise ValueError("verification target is required")
        if self.verification_kind == "service_state":
            if exe != "systemctl" or not words or words[0] not in {"restart", "start", "stop"}:
                raise ValueError("service verification does not match the operation")
            if self.verification_target not in self.argv[1:]:
                raise ValueError("service verification target must be an approved argument")
        if (self.verification_kind == "gpu_state"
                and (exe != "nvidia-smi" or not self.verification_target.isdigit())):
            raise ValueError("GPU verification must name a numeric nvidia-smi target")
        return self

    def fingerprint(self, relevant_preflight: dict[str, object]) -> str:
        payload = {"operation": self.model_dump(mode="json"), "preflight": relevant_preflight}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def execution_argv(self) -> tuple[str, ...]:
        command = (self.executable, *self.argv)
        return ("sudo", "-n", *command) if self.requires_sudo else command


class OperationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int | None = None
    host: str
    action_type: Literal["EXECUTE_APPROVED_ARGV"] = "EXECUTE_APPROVED_ARGV"
    plan: OperationPlan
    risk_class: Literal["DANGEROUS"] = "DANGEROUS"
    status: ProposalStatus
    created_at: datetime
    expires_at: datetime
    created_by: str
    preflight_summary: dict[str, bool]
    preflight_fingerprint: str


def render_operation_proposal(plan: OperationPlan, proposal_id: int, expires_at: str) -> str:
    command = " ".join((*(("sudo",) if plan.requires_sudo else ()), plan.executable, *plan.argv))
    return (
        f"⚠️ Operation Proposal #{proposal_id}\n"
        f"Host: {plan.host}\nHost source: {plan.host_source} ({plan.host})\n"
        "Action: EXECUTE_APPROVED_ARGV\nRisk: DANGEROUS\n"
        f"Sudo: {'required' if plan.requires_sudo else 'not required'}\n"
        f"Executable: {plan.executable}\nArguments: {json.dumps(plan.argv, ensure_ascii=False)}\n"
        f"Command: {command}\nCommand source: {plan.command_source}\nTarget: {plan.target}\n"
        f"Current relevant state: {plan.current_relevant_state}\n"
        f"Active workload: {plan.active_workload}\nRunning VM: {plan.running_vm}\n"
        f"Reason: {plan.reason}\nExpected effect: {plan.expected_effect}\n"
        f"Known side effects: {', '.join(plan.known_side_effects) or 'none known'}\n"
        f"Verification plan: {plan.verification_plan}\n"
        f"Rollback available: {'yes' if plan.rollback else 'no'}\nExpiry: {expires_at}"
    )
