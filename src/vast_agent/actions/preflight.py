from __future__ import annotations

import re
from typing import Protocol

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    PreflightSnapshot,
    ServiceParameters,
)
from vast_agent.config import OperationsSettings
from vast_agent.execution.base import Executor, redact
from vast_agent.models.host import Host
from vast_agent.services.inspection import InspectionService


class PreflightProvider(Protocol):
    def collect(self, host: Host, request: ActionRequest) -> PreflightSnapshot: ...


class ProductionPreflightProvider:
    """Collects conservative preflight evidence through existing read-only inspection tools."""

    def __init__(self, inspection: InspectionService, executor: Executor,
                 settings: OperationsSettings, secrets: tuple[str, ...] = ()) -> None:
        self.inspection, self.executor, self.settings = inspection, executor, settings
        self.secrets = secrets

    def collect(self, host: Host, request: ActionRequest) -> PreflightSnapshot:
        if not host.enabled:
            return PreflightSnapshot(host_enabled=False)
        record = self.inspection.inspect_and_record(host, self.executor)
        observation, results = record.observation, record.tool_results
        sudo = self.executor.execute(host, ("sudo", "-n", "true"), 10)
        target_exists, current = self._target(host, request, results, observation)
        gpu_binding = None
        mapping = True
        gpu_processes = bool(results.get("get_gpu_processes") and
                             results["get_gpu_processes"].stdout.strip())
        if isinstance(request.parameters, GPUParameters):
            devices = {device.index: device for device in observation.gpu.devices}
            device = devices.get(request.parameters.gpu_index)
            mapping = bool(observation.gpu.nvml_ok and device and device.pci_bus)
            if mapping:
                pci = next((item for item in observation.gpu.pci_devices
                            if item.device_type == "gpu" and
                            item.pci_address.lower().endswith(str(device.pci_bus).lower())), None)
                mapping = pci is not None
                gpu_binding = ({"nvidia": "nvidia", "vfio-pci": "vfio"}.get(pci.driver, "unbound")
                               if pci else "unknown")
        docker = results.get("get_docker_status")
        docker_lines = docker.stdout.splitlines() if docker and docker.success else []
        containers = [line for line in docker_lines[1:] if line.strip()]
        vm = results.get("get_vm_status")
        vm_text = vm.stdout if vm and vm.success else ""
        running_vm = bool(re.search(r"\brunning\b|qemu-system", vm_text, re.I))
        filesystem = observation.system.filesystem_max_percent
        details = {
            "service_state": current,
            "failed_units": observation.system.failed_units,
            "filesystem_max_percent": filesystem,
            "docker_containers": len(containers),
            "evidence_complete": all(r.success for r in results.values()),
        }
        details = {key: redact(str(value), self.secrets) for key, value in details.items()}
        return PreflightSnapshot(
            host_enabled=True, ssh_reachable=observation.ssh_ok, sudo_available=sudo.success,
            target_exists=target_exists,
            evidence_complete=bool(results) and all(r.success for r in results.values()),
            current_state=current,
            active_workload=bool(containers), running_vm=running_vm,
            d_state=observation.system.d_state_processes > 0,
            filesystem_healthy=filesystem is not None and filesystem < 95,
            gpu_mapping_resolved=mapping, gpu_binding=gpu_binding,
            gpu_processes=gpu_processes,
            docker_running=observation.services.get("docker") == "active",
            signatures=observation.signatures, details=details,
        )

    def _target(self, host, request, results, observation) -> tuple[bool, str]:
        params = request.parameters
        if isinstance(params, ServiceParameters):
            show = results.get("get_service_status")
            exists = bool(show and show.success and f"Id={params.service}" in show.stdout)
            return exists, observation.services.get(params.service.removesuffix(".service"), "unknown")
        if isinstance(params, ContainerParameters):
            result = results.get("get_docker_status")
            line = next((line for line in result.stdout.splitlines()[1:]
                         if len(line.split("|")) > 1 and line.split("|")[1] == params.container), None) if result else None
            return line is not None, line.split("|")[2] if line else "missing"
        if request.action_type == ActionType.GPU_RESET:
            index = request.parameters.gpu_index
            found = next((d for d in observation.gpu.devices if d.index == index), None)
            return found is not None, found.power_state or "unknown" if found else "missing"
        if request.action_type in {ActionType.VM_MODE_ENABLE, ActionType.VM_MODE_DISABLE}:
            if not self.settings.enable_vms_script: return False, "not configured"
            result = self.executor.execute(host, ("test", "-f", self.settings.enable_vms_script), 10)
            return result.success, "configured" if result.success else "missing"
        if request.action_type == ActionType.HOST_REBOOT:
            return True, "up" if observation.ssh_ok else "unknown"
        return False, "unknown"
