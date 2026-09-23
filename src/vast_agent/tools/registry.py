from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from vast_agent.execution.base import Executor
from vast_agent.jobs.cancellation import CancellationToken, current_cancellation
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ErrorCode, ToolResult


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    risk_class: str
    supported_capabilities: tuple[str, ...]
    timeout: int
    command: tuple[str, ...]


# Commands are code-owned constants. No API accepts a caller supplied shell string.
TOOLS: Final[dict[str, ToolMetadata]] = {
    "host_ping": ToolMetadata("host_ping", "Verify SSH connectivity", "read_only", (), 10, ("true",)),
    "get_system_health": ToolMetadata("get_system_health", "Filesystem and failed-unit health", "read_only", (), 20, ("sh", "-c", "df -P; systemctl --failed --no-legend --plain")),
    "get_gpu_status": ToolMetadata("get_gpu_status", "NVIDIA GPU metrics", "read_only", ("nvidia",), 20, ("nvidia-smi", "--query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total,pstate,pci.bus_id,power.draw", "--format=csv,noheader,nounits")),
    "get_gpu_processes": ToolMetadata("get_gpu_processes", "NVIDIA compute processes", "read_only", ("nvidia",), 20, ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits")),
    # PCI discovery remains available when NVML/NVIDIA support is absent (for example VFIO).
    # The parser selects only NVIDIA display and companion audio functions.
    "get_pci_status": ToolMetadata("get_pci_status", "GPU PCI bindings", "read_only", (), 20, ("lspci", "-Dnnk")),
    "get_kernel_gpu_errors": ToolMetadata("get_kernel_gpu_errors", "Kernel GPU errors", "read_only", ("nvidia",), 20, ("journalctl", "-k", "-b", "--no-pager", "-p", "warning")),
    "get_d_state_processes": ToolMetadata("get_d_state_processes", "Uninterruptible processes", "read_only", (), 15, ("ps", "-eo", "stat,pid,comm", "--no-headers")),
    "get_vast_status": ToolMetadata("get_vast_status", "Vast service state", "read_only", ("vast",), 15, ("systemctl", "show", "vastai.service", "--property=ActiveState,MainPID,SubState")),
    "get_vast_logs": ToolMetadata("get_vast_logs", "Recent Vast warnings", "read_only", ("vast",), 20, ("journalctl", "-u", "vastai.service", "-n", "200", "--no-pager", "-p", "warning")),
    "get_docker_status": ToolMetadata("get_docker_status", "Docker service and containers", "read_only", ("docker",), 20, ("sh", "-c", "systemctl is-active docker; docker ps -a --format '{{.ID}}|{{.Names}}|{{.Status}}|{{.Image}}'")),
    "get_vm_status": ToolMetadata("get_vm_status", "Libvirt domains and QEMU processes", "read_only", ("libvirt",), 20, ("sh", "-c", "virsh list --all; pgrep -a qemu-system || true")),
    "get_service_status": ToolMetadata("get_service_status", "Allowlisted service states", "read_only", (), 15, ("systemctl", "show", "vastai.service", "docker.service", "libvirtd.service", "--property=Id,LoadState,ActiveState,SubState,MainPID")),
    "get_journal_errors": ToolMetadata("get_journal_errors", "Recent boot errors", "read_only", (), 20, ("journalctl", "-b", "-p", "err", "-n", "200", "--no-pager")),
    "read_config": ToolMetadata("read_config", "Read allowlisted non-secret Vast config", "read_only", ("vast",), 10, ("cat", "/var/lib/vastai_kaalia/host_enabled")),
}


def run_tool(name: str, host: Host, executor: Executor,
             cancellation: CancellationToken | None = None) -> ToolResult:
    cancellation = cancellation or current_cancellation.get()
    metadata = TOOLS.get(name)
    if metadata is None:
        return ToolResult(success=False, duration_ms=0, error_code=ErrorCode.TOOL_UNSUPPORTED)
    unsupported = [c for c in metadata.supported_capabilities if not getattr(host.capabilities, c)]
    if unsupported:
        return ToolResult(
            success=False, duration_ms=0, error_code=ErrorCode.TOOL_UNSUPPORTED,
            stderr=f"Host does not declare capability: {', '.join(unsupported)}",
        )
    execute_tool = getattr(executor, "execute_tool", None)
    if execute_tool is not None:
        return execute_tool(name, host, metadata.command, metadata.timeout)
    return executor.execute(host, metadata.command, metadata.timeout, cancellation)
