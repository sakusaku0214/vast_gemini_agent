from vast_agent.diagnostics.parser import build_observation
from vast_agent.execution.base import Executor
from vast_agent.models.host import Host
from vast_agent.models.observation import Observation
from vast_agent.tools.registry import run_tool

GROUPS = {
    "gpu": (
        "host_ping", "get_gpu_status", "get_gpu_processes", "get_pci_status",
        "get_kernel_gpu_errors",
    ),
    "pci": ("host_ping", "get_pci_status", "get_kernel_gpu_errors"),
    "vast": ("host_ping", "get_vast_status", "get_vast_logs"),
    "docker": ("host_ping", "get_docker_status"),
    "vm": ("host_ping", "get_vm_status", "get_pci_status"),
    "system": ("host_ping", "get_system_health", "get_d_state_processes", "get_service_status"),
}
FULL = tuple(dict.fromkeys(
    [tool for tools in GROUPS.values() for tool in tools] + ["get_journal_errors"],
))


def inspect_host(host: Host, executor: Executor, group: str | None = None) -> Observation:
    names = GROUPS[group] if group else FULL
    return build_observation(
        host.name,
        {name: run_tool(name, host, executor) for name in names},
        host.expected_gpu_count,
    )
