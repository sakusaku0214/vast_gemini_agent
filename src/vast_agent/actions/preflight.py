from __future__ import annotations

import re
from dataclasses import dataclass
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


def normalize_pci_bdf(value: str | None) -> str | None:
    """Normalize domain-width variants to the stable bus:device.function suffix."""
    if not value:
        return None
    match = re.search(r"(?:[0-9a-f]{4,8}:)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$",
                      value.strip(), re.I)
    return match.group(1).lower() if match else None


@dataclass(frozen=True)
class DockerWorkloadSummary:
    """Conservative interpretation of ``get_docker_status`` container rows."""

    total_containers: int
    running_containers: int
    unknown_containers: int

    @property
    def has_active_workload(self) -> bool:
        # An unrecognized row must never be interpreted as an idle host.
        return self.running_containers > 0 or self.unknown_containers > 0


def docker_workload_summary(output: str) -> DockerWorkloadSummary:
    """Summarize Docker rows while treating malformed/unknown states as active."""
    lines = output.splitlines()
    rows = [line for line in lines[1:] if line.strip()]
    running = 0
    # The first line is the systemctl state and is required to delimit the rows.
    unknown = int(not lines or not lines[0].strip() or "|" in lines[0])
    for row in rows:
        fields = row.split("|", 3)
        if len(fields) != 4 or not all(field.strip() for field in fields):
            unknown += 1
            continue
        status = fields[2].strip()
        if status.startswith(("Up ", "Restarting")):
            running += 1
        elif not status.startswith(("Exited", "Created", "Dead")):
            unknown += 1
    return DockerWorkloadSummary(
        total_containers=len(rows),
        running_containers=running,
        unknown_containers=unknown,
    )


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
        required_tools = self._required_tools(host, request)
        tolerated_failures = set()
        if (request.action_type == ActionType.VM_MODE_DISABLE
                and observation.gpu.all_bound_to_vfio):
            # nvidia-smi cannot enumerate processes when every GPU is intentionally
            # detached from NVIDIA.  Complete PCI evidence makes that one failure safe.
            tolerated_failures.add("get_gpu_processes")
        evidence_complete = sudo.success and all(
            name in tolerated_failures or (name in results and results[name].success)
            for name in required_tools
        )
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
                target_bdf = normalize_pci_bdf(device.pci_bus)
                pci = next((item for item in observation.gpu.pci_devices
                            if item.device_type == "gpu" and
                            normalize_pci_bdf(item.pci_address) == target_bdf), None)
                mapping = pci is not None
                gpu_binding = ({"nvidia": "nvidia", "vfio-pci": "vfio"}.get(pci.driver, "unbound")
                               if pci else "unknown")
        elif request.action_type in {ActionType.VM_MODE_ENABLE, ActionType.VM_MODE_DISABLE}:
            gpu_devices = [item for item in observation.gpu.pci_devices
                           if item.device_type == "gpu"]
            mapping = bool(gpu_devices) and observation.gpu.unknown_bound == 0 and observation.gpu.unbound == 0
            if mapping and observation.gpu.vfio_bound == len(gpu_devices): gpu_binding = "vfio"
            elif mapping and observation.gpu.nvidia_bound == len(gpu_devices): gpu_binding = "nvidia"
            else: gpu_binding = "unknown"
        docker = results.get("get_docker_status")
        docker_summary = docker_workload_summary(docker.stdout) if docker and docker.success else (
            DockerWorkloadSummary(0, 0, 0)
        )
        vm = results.get("get_vm_status")
        vm_text = vm.stdout if vm and vm.success else ""
        running_vm = bool(re.search(r"\brunning\b|qemu-system", vm_text, re.I))
        filesystem = observation.system.filesystem_max_percent
        details = {
            "service_state": current,
            "failed_units": observation.system.failed_units,
            "filesystem_max_percent": filesystem,
            "docker_total_containers": docker_summary.total_containers,
            "docker_running_containers": docker_summary.running_containers,
            "docker_unknown_containers": docker_summary.unknown_containers,
            "required_tools": ",".join(sorted(required_tools)),
            "evidence_complete": evidence_complete,
        }
        details = {key: redact(str(value), self.secrets) for key, value in details.items()}
        return PreflightSnapshot(
            host_enabled=True, ssh_reachable=observation.ssh_ok, sudo_available=sudo.success,
            target_exists=target_exists,
            evidence_complete=evidence_complete,
            current_state=current,
            active_workload=docker_summary.has_active_workload, running_vm=running_vm,
            d_state=observation.system.d_state_processes > 0,
            filesystem_healthy=filesystem is not None and filesystem < 95,
            gpu_mapping_resolved=mapping, gpu_binding=gpu_binding,
            gpu_processes=gpu_processes,
            nvml_ok=observation.gpu.nvml_ok,
            gpu_present=(
                observation.gpu.pci_count > 0
                if observation.gpu.pci_observed and observation.gpu.pci_ok
                else bool(observation.gpu.devices)
            ),
            pci_nvidia=observation.gpu.nvidia_bound,
            pci_vfio=observation.gpu.vfio_bound,
            pci_unbound=observation.gpu.unbound + observation.gpu.unknown_bound,
            failed_units=observation.system.failed_units,
            vast_state=observation.services.get("vastai"),
            docker_running=observation.services.get("docker") == "active",
            signatures=observation.signatures, details=details,
        )

    @staticmethod
    def _required_tools(host: Host, request: ActionRequest) -> set[str]:
        """Return only evidence relevant to this action and declared host capabilities."""
        common = {"host_ping"}
        action = request.action_type
        if action == ActionType.RESTART_VAST_SERVICE:
            required = common | {"get_vast_status", "get_service_status", "get_system_health"}
            if host.capabilities.docker:
                required.add("get_docker_status")
            if host.capabilities.libvirt:
                required.add("get_vm_status")
            return required
        if action == ActionType.RESTART_DOCKER_SERVICE:
            return common | {"get_service_status", "get_docker_status"}
        if action == ActionType.RESTART_LIBVIRT_SERVICE:
            return common | {"get_service_status", "get_vm_status"}
        if action == ActionType.RESTART_VAST_CONTAINER:
            return common | {"get_docker_status"}
        if action == ActionType.GPU_RESET:
            required = common | {
                "get_gpu_status", "get_gpu_processes", "get_pci_status",
                "get_d_state_processes", "get_kernel_gpu_errors",
            }
            if host.capabilities.libvirt: required.add("get_vm_status")
            if host.capabilities.docker: required.add("get_docker_status")
            return required
        if action in {ActionType.VM_MODE_ENABLE, ActionType.VM_MODE_DISABLE}:
            return common | {
                "get_vm_status", "get_pci_status", "get_gpu_processes",
                "get_d_state_processes",
            }
        # Reboot is broad, but an explicitly absent capability is not missing evidence.
        required = common | {
            "get_system_health", "get_d_state_processes", "get_service_status",
            "get_journal_errors",
        }
        if host.capabilities.nvidia:
            required |= {"get_gpu_status", "get_gpu_processes", "get_pci_status",
                         "get_kernel_gpu_errors"}
        if host.capabilities.vast:
            required |= {"get_vast_status", "get_vast_logs"}
        if host.capabilities.docker: required.add("get_docker_status")
        if host.capabilities.libvirt: required.add("get_vm_status")
        return required

    def _target(self, host, request, results, observation) -> tuple[bool, str]:
        params = request.parameters
        if isinstance(params, ServiceParameters):
            capability = {
                "vastai.service": host.capabilities.vast,
                "docker.service": host.capabilities.docker,
                "libvirtd.service": host.capabilities.libvirt,
            }[params.service]
            show = results.get("get_service_status")
            exists = bool(capability and show and show.success and f"Id={params.service}" in show.stdout)
            return exists, observation.services.get(params.service.removesuffix(".service"), "unknown")
        if isinstance(params, ContainerParameters):
            if not host.capabilities.docker:
                return False, "docker capability unavailable"
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
