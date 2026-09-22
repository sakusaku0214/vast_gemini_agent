from __future__ import annotations

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    PackageInstallParameters,
    ServiceParameters,
    VerificationResult,
    VMParameters,
)
from vast_agent.actions.package_catalog import package_definition
from vast_agent.actions.preflight import ProductionPreflightProvider
from vast_agent.models.host import Host


class ProductionActionVerifier:
    """READ-only post-action verification; inability to observe is never success."""

    def __init__(self, preflight: ProductionPreflightProvider) -> None:
        self.preflight = preflight

    def verify(self, host: Host, request: ActionRequest) -> VerificationResult:
        state = self.preflight.collect(host, request)
        params = request.parameters
        if not state.ssh_reachable:
            return VerificationResult(success=False, status="UNAVAILABLE", summary="SSH unavailable")
        if isinstance(params, PackageInstallParameters):
            executable_states = []
            service_states = []
            definition = package_definition(params.package_name)
            if definition:
                for executable in definition.executables:
                    found = self.preflight.executor.execute(
                        host, ("which", "--", executable), 10,
                    ).success
                    executable_states.append(f"{executable}={'present' if found else 'missing'}")
                for service in definition.services:
                    result = self.preflight.executor.execute(
                        host, ("systemctl", "is-active", f"{service}.service"), 10,
                    )
                    service_states.append(f"{service}.service={result.stdout.strip() or 'inactive'}")
            ok = state.package_installed is True
            summary = f"{params.package_name}: {'installed' if ok else 'absent'}"
            if state.package_version: summary += f" version={state.package_version}"
            observations = executable_states + service_states
            if observations: summary += "; " + ", ".join(observations)
            return VerificationResult(success=ok, status="VERIFIED" if ok else "FAILED",
                                      summary=summary)
        if isinstance(params, ServiceParameters):
            ok = state.target_exists and state.current_state == "active"
            return VerificationResult(success=ok, status="VERIFIED" if ok else "FAILED",
                                      summary=f"{params.service}: {state.current_state}")
        if isinstance(params, ContainerParameters):
            ok = state.target_exists and state.current_state.lower().startswith("up")
            return VerificationResult(success=ok, status="VERIFIED" if ok else "FAILED",
                                      summary=f"{params.container}: {state.current_state}")
        if isinstance(params, GPUParameters):
            ok = state.target_exists and state.gpu_mapping_resolved and state.gpu_binding == "nvidia" and not state.gpu_processes
            return VerificationResult(success=ok, status="RECOVERED" if ok else "FAILED",
                                      summary=f"GPU{params.gpu_index}: binding={state.gpu_binding}")
        if isinstance(params, VMParameters):
            ownership_known = state.gpu_mapping_resolved and state.pci_unbound == 0
            ok = state.target_exists and ownership_known and (
                (params.mode == "on" and state.gpu_binding == "vfio" and state.pci_nvidia == 0) or
                (params.mode == "off" and not state.running_vm and
                 state.gpu_binding == "nvidia" and state.pci_vfio == 0)
            )
            return VerificationResult(success=ok, status="VERIFIED" if ok else "FAILED",
                                      summary=f"VM mode {params.mode}: vm_running={state.running_vm}")
        if request.action_type == ActionType.HOST_REBOOT:
            return VerificationResult(success=False, status="UNAVAILABLE", summary="dedicated reboot verifier required")
        return VerificationResult(success=False, status="UNAVAILABLE", summary="unsupported verification")
