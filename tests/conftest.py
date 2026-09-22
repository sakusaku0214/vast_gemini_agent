from pathlib import Path

import pytest

from vast_agent.models.host import Capabilities, Host
from vast_agent.models.tool_result import ToolResult


@pytest.fixture
def host() -> Host:
    return Host(name="test-host", address="192.0.2.20", ssh_user="tester",
                capabilities=Capabilities(nvidia=True, docker=True, libvirt=True, vast=True))


def fixture_results(case: str) -> dict[str, ToolResult]:
    root = Path(__file__).parent / "fixtures" / case
    reachable = case != "ssh_failure"
    def read(name: str) -> str:
        path = root / f"{name}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else ""
    gpu = read("get_gpu_status")
    return {
        "host_ping": ToolResult(success=reachable, duration_ms=1),
        "get_gpu_status": ToolResult(success=bool(gpu) and "NVML_ERROR" not in gpu,
                                      stdout=gpu if "NVML_ERROR" not in gpu else "",
                                      stderr=gpu if "NVML_ERROR" in gpu else "", duration_ms=1),
        "get_pci_status": ToolResult(success=reachable, stdout=read("get_pci_status"), duration_ms=1),
        "get_kernel_gpu_errors": ToolResult(success=reachable, stdout=read("get_kernel_gpu_errors"), duration_ms=1),
        "get_vast_logs": ToolResult(success=reachable, stdout=read("get_vast_logs"), duration_ms=1),
        "get_system_health": ToolResult(success=reachable, stdout=read("get_system_health"), duration_ms=1),
        "get_d_state_processes": ToolResult(success=reachable, stdout="S 1 init\n", duration_ms=1),
        "get_service_status": ToolResult(success=reachable, stdout="Id=vastai.service\nActiveState=active\nId=docker.service\nActiveState=active\n", duration_ms=1),
    }
