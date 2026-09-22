from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ActionType(StrEnum):
    PACKAGE_INSTALL = "PACKAGE_INSTALL"
    RESTART_VAST_SERVICE = "RESTART_VAST_SERVICE"
    RESTART_DOCKER_SERVICE = "RESTART_DOCKER_SERVICE"
    RESTART_LIBVIRT_SERVICE = "RESTART_LIBVIRT_SERVICE"
    RESTART_VAST_CONTAINER = "RESTART_VAST_CONTAINER"
    GPU_RESET = "GPU_RESET"
    VM_MODE_ENABLE = "VM_MODE_ENABLE"
    VM_MODE_DISABLE = "VM_MODE_DISABLE"
    HOST_REBOOT = "HOST_REBOOT"


class RiskClass(StrEnum):
    WRITE = "WRITE"
    DANGEROUS = "DANGEROUS"
    REBOOT = "REBOOT"


class ProposalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    BLOCKED = "BLOCKED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    FAILED_VERIFY = "FAILED_VERIFY"
    INVALIDATED = "INVALIDATED"
    INTERRUPTED = "INTERRUPTED"
    INTERRUPTED_UNKNOWN = "INTERRUPTED_UNKNOWN"
    COMMAND_DISPATCHED = "COMMAND_DISPATCHED"
    WAITING_FOR_HOST_DOWN = "WAITING_FOR_HOST_DOWN"
    WAITING_FOR_HOST_UP = "WAITING_FOR_HOST_UP"
    VERIFYING = "VERIFYING"
    RECOVERY_FAILED = "RECOVERY_FAILED"


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceParameters(_Params):
    kind: Literal["service"] = "service"
    service: Literal["vastai.service", "docker.service", "libvirtd.service"]


class ContainerParameters(_Params):
    kind: Literal["container"] = "container"
    container: str

    @field_validator("container")
    @classmethod
    def validate_container(cls, value: str) -> str:
        if not re.fullmatch(r"C\.[0-9]+", value):
            raise ValueError("container must match C.<digits>")
        return value


class GPUParameters(_Params):
    kind: Literal["gpu"] = "gpu"
    gpu_index: int = Field(ge=0)


class VMParameters(_Params):
    kind: Literal["vm"] = "vm"
    mode: Literal["on", "off"]


class RebootParameters(_Params):
    kind: Literal["reboot"] = "reboot"
    assessment: Literal["HOST_REBOOT_CANDIDATE"]


class PackageInstallParameters(_Params):
    kind: Literal["package_install"] = "package_install"
    package_name: str = Field(min_length=1, max_length=128)
    expected_capability: str | None = Field(default=None, max_length=128)
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("package_name")
    @classmethod
    def validate_package_name(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", value):
            raise ValueError("package_name must be a canonical Debian package name")
        return value


ActionParameters = Annotated[
    ServiceParameters | ContainerParameters | GPUParameters | VMParameters | RebootParameters
    | PackageInstallParameters,
    Field(discriminator="kind"),
]


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    host: str = Field(min_length=1)
    action_type: ActionType
    parameters: ActionParameters


class PreflightSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host_enabled: bool = True
    ssh_reachable: bool = False
    sudo_available: bool = False
    target_exists: bool = False
    evidence_complete: bool = False
    current_state: str = "unknown"
    active_workload: bool = False
    running_vm: bool = False
    package_installed: bool | None = None
    package_version: str | None = None
    package_candidate: str | None = None
    package_manager_busy: bool | None = None
    d_state: bool = False
    filesystem_healthy: bool = True
    gpu_mapping_resolved: bool = True
    gpu_binding: Literal["nvidia", "vfio", "unbound", "unknown"] | None = None
    gpu_processes: bool = False
    nvml_ok: bool | None = None
    gpu_present: bool | None = None
    pci_nvidia: int = 0
    pci_vfio: int = 0
    pci_unbound: int = 0
    failed_units: int | None = None
    vast_state: str | None = None
    docker_running: bool | None = None
    signatures: list[str] = Field(default_factory=list)
    details: dict[str, str | int | bool | None] = Field(default_factory=dict)

    def fingerprint(self, request: ActionRequest) -> str:
        stable = {"request": request.model_dump(mode="json"), "state": self.model_dump(mode="json")}
        payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int | None = None
    host: str
    action_type: ActionType
    parameters: ActionParameters
    risk_class: RiskClass
    status: ProposalStatus
    created_at: datetime
    expires_at: datetime
    created_by: str
    preflight_summary: PreflightSnapshot
    preflight_fingerprint: str
    job_id: int | None = None


class VerificationResult(BaseModel):
    success: bool
    status: Literal["RECOVERED", "UNCHANGED", "FAILED", "UNAVAILABLE", "VERIFIED"]
    summary: str
