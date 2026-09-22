from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from vast_agent.actions.models import ActionRequest, ActionType, PreflightSnapshot
from vast_agent.config import OperationsSettings


class PolicyDecision(StrEnum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


class PolicyResult(BaseModel):
    decision: PolicyDecision
    reasons: list[str]


class PolicyEngine:
    def evaluate(self, request: ActionRequest, state: PreflightSnapshot, settings: OperationsSettings, *, execution: bool = False) -> PolicyResult:
        reasons: list[str] = []
        if execution and not settings.enabled: reasons.append("OPERATIONS_DISABLED")
        if not state.host_enabled: reasons.append("HOST_DISABLED")
        if not state.ssh_reachable: reasons.append("SSH_UNREACHABLE")
        if not state.sudo_available: reasons.append("SUDO_NOT_AVAILABLE")
        if not state.target_exists: reasons.append("TARGET_NOT_FOUND")
        if not state.evidence_complete: reasons.append("PREFLIGHT_INCOMPLETE")
        if request.action_type == ActionType.GPU_RESET:
            if not state.gpu_mapping_resolved: reasons.append("GPU_MAPPING_UNRESOLVED")
            if state.gpu_binding != "nvidia": reasons.append("GPU_OWNERSHIP_UNSAFE")
            if state.gpu_processes: reasons.append("ACTIVE_GPU_PROCESS")
            if state.running_vm: reasons.append("RUNNING_VM_GPU_CONFLICT")
        if request.action_type in {ActionType.VM_MODE_ENABLE, ActionType.VM_MODE_DISABLE}:
            if not settings.enable_vms_script: reasons.append("VM_ACTION_NOT_CONFIGURED")
            if not state.gpu_mapping_resolved: reasons.append("VM_GPU_OWNERSHIP_UNRESOLVED")
            if state.running_vm or state.gpu_processes or state.active_workload: reasons.append("VM_MODE_CONFLICT")
        if request.action_type == ActionType.RESTART_LIBVIRT_SERVICE and state.running_vm:
            reasons.append("RUNNING_VM")
        if request.action_type == ActionType.HOST_REBOOT:
            if state.running_vm or state.active_workload: reasons.append("ACTIVE_WORKLOAD")
            if not state.filesystem_healthy: reasons.append("FILESYSTEM_UNHEALTHY")
        if reasons: return PolicyResult(decision=PolicyDecision.BLOCK, reasons=reasons)
        warnings = []
        if request.action_type == ActionType.RESTART_DOCKER_SERVICE and state.active_workload:
            warnings.append("RUNNING_CONTAINERS")
        if state.d_state: warnings.append("D_STATE_PRESENT")
        return PolicyResult(decision=PolicyDecision.WARN if warnings else PolicyDecision.ALLOW, reasons=warnings)
