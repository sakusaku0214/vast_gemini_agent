from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Route(StrEnum):
    DETERMINISTIC = "deterministic"
    AGENT = "agent"
    UNSUPPORTED_WRITE = "unsupported_write"
    UNKNOWN = "unknown"


class IntentDecision(BaseModel):
    route: Route
    host: str | None = None
    action: str | None = None
    scope: str | None = None
    reason: str | None = None


class InspectHostArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str
    scope: Literal["full", "gpu", "pci", "vast", "docker", "vm", "system"]


class CollectEvidenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str
    evidence_type: Literal[
        "gpu_status", "gpu_processes", "pci", "kernel_gpu_errors", "d_state",
        "vast_status", "vast_logs", "docker", "vm", "services", "journal_errors",
        "system_health",
    ]


class RecentIncidentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str
    signature: str | None = None
    limit: int = Field(default=3, ge=1, le=5)


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, object]
    call_id: str = ""


class AgentResponse(BaseModel):
    text: str | None = None
    function_calls: list[FunctionCall] = Field(default_factory=list)
    usage: dict[str, int | None] = Field(default_factory=dict)


class InvestigationResult(BaseModel):
    summary: str
    findings: list[str] = Field(default_factory=list)
    signatures: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    recommended_action: Literal[
        "NONE", "CONTINUE_OBSERVING", "SERVICE_RESTART_CANDIDATE",
        "GPU_RESET_CANDIDATE", "VM_REBIND_CANDIDATE", "HOST_REBOOT_CANDIDATE",
        "PHYSICAL_CHECK_REQUIRED",
    ] = "CONTINUE_OBSERVING"
    missing_evidence: list[str] = Field(default_factory=list)
