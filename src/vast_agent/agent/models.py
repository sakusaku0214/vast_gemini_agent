from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class HostArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str


class PackageQueryArgs(HostArgument):
    package_name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9+.-]*$")


class ExecutableQueryArgs(HostArgument):
    executable_name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class ServiceQueryArgs(HostArgument):
    service_name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$")

    @field_validator("service_name")
    @classmethod
    def safe_unit(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("service name cannot contain '..'")
        return value


class NetworkInspectionArgs(HostArgument):
    scope: Literal["summary", "interfaces", "addresses", "routes"] = "summary"


class InterfaceInspectionArgs(HostArgument):
    interface_name: str = Field(min_length=1, max_length=15, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")

    @field_validator("interface_name")
    @classmethod
    def safe_interface(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("interface name cannot contain '..'")
        return value


class TrafficHistoryArgs(HostArgument):
    period: Literal["today", "yesterday", "last_24h", "last_7d"]
    interface: str | None = Field(
        default=None, min_length=1, max_length=15,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )

    @field_validator("interface")
    @classmethod
    def safe_interface(cls, value: str | None) -> str | None:
        if value is not None and ".." in value:
            raise ValueError("interface cannot contain '..'")
        return value


class NvmeHealthArgs(HostArgument):
    device: str | None = Field(default=None, min_length=5, max_length=14)

    @field_validator("device")
    @classmethod
    def canonical_device(cls, value: str | None) -> str | None:
        if value is None:
            return None
        name = value.removeprefix("/dev/")
        if not name.startswith("nvme") or not name[4:].isdigit():
            raise ValueError("device must be an NVMe controller such as nvme0")
        return f"/dev/{name}"


class ProcessQueryArgs(HostArgument):
    process_name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, object]
    call_id: str = ""


class AgentResponse(BaseModel):
    output_text: str | None = None
    steps: list[dict[str, object]] = Field(default_factory=list)
    function_calls: list[FunctionCall] = Field(default_factory=list)
    usage: dict[str, int | None] = Field(default_factory=dict)


class CapabilityGap(BaseModel):
    """Validated READ conclusion; package suggestions are deliberately non-authoritative."""

    model_config = ConfigDict(extra="forbid")
    capability_id: str = Field(min_length=1, max_length=64)
    status: Literal["available", "degraded", "missing", "unknown", "unsupported"]
    software_status: Literal["available", "missing", "unknown"]
    service_status: Literal[
        "active", "inactive", "unavailable", "unknown", "not_required",
    ] = "unknown"
    reason: str = Field(max_length=500)
    evidence: list[str] = Field(default_factory=list, max_length=12)
    candidate_package: str | None = Field(default=None, max_length=128)
    confidence: Literal["low", "medium", "high"] = "low"

    @field_validator("capability_id")
    @classmethod
    def registered_capability(cls, value: str) -> str:
        # Local import avoids making the declarative catalog depend on Gemini models.
        from vast_agent.actions.package_catalog import capability_definition

        if capability_definition(value) is None:
            raise ValueError("capability_id is not in the code-owned registry")
        return value


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
    capability_gaps: list[CapabilityGap] = Field(default_factory=list, max_length=3)
