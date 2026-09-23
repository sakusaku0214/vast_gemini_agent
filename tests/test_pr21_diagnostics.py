from __future__ import annotations

import json

import pytest

from vast_agent.actions.package_catalog import CAPABILITY_REGISTRY
from vast_agent.agent.host_read import HOST_READ_CAPABILITIES, discover, execute_diagnostic
from vast_agent.execution.base import FakeExecutor
from vast_agent.models.host import Capabilities
from vast_agent.models.tool_result import ErrorCode, ToolResult
from vast_agent.tools.registry import TOOLS

PCI_NVIDIA = """00000000:01:00.0 VGA compatible controller: NVIDIA Corporation Device [10de:2684]
\tKernel driver in use: nvidia
\tKernel modules: nvidia
"""


def result(success=True, stdout="", *, error_code=None):
    return ToolResult(success=success, stdout=stdout, duration_ms=1, error_code=error_code)


def gpu_results(pci=PCI_NVIDIA, nvml=None, kernel=""):
    if nvml is None:
        nvml = "0, GPU-1, RTX 4090, 41, 5, 100, 24564, P8, 00000000:01:00.0\n"
    return {
        "host_ping": result(), "get_pci_status": result(stdout=pci),
        "get_gpu_status": result(stdout=nvml),
        "get_kernel_gpu_errors": result(stdout=kernel),
        "get_d_state_processes": result(stdout="D 42 worker\nS 1 init\n"),
    }


def gpu(host, results):
    return execute_diagnostic("query_gpu_diagnostics", host, FakeExecutor(results))


def test_gpu_healthy_and_expected_count_is_structured(host):
    value = gpu(host.model_copy(update={"expected_gpu_count": 1}), gpu_results())
    assert value["status"] == "healthy" and value["ownership_mode"] == "all_nvidia"
    assert value["pci_count"] == value["expected_gpu_count"] == 1
    assert value["d_state_process_count"] == 1
    assert value["devices"][0]["model"] == "RTX 4090"


def test_all_vfio_does_not_turn_expected_nvml_absence_into_failure(host):
    fixtures = gpu_results(pci=PCI_NVIDIA.replace("nvidia\n", "vfio-pci\n"))
    fixtures["get_gpu_status"] = result(False, error_code=ErrorCode.TOOL_UNSUPPORTED)
    value = gpu(host, fixtures)
    assert value["status"] == "healthy" and value["ownership_mode"] == "all_vfio"
    assert value["nvml_status"] == "not_expected"
    assert "NVML_UNAVAILABLE" not in value["signatures"]


def test_mixed_unbound_signatures_mismatch_and_partial_nvml(host):
    second = PCI_NVIDIA.replace("01:00.0", "02:00.0").replace(
        "Kernel driver in use: nvidia", "Kernel driver in use: vfio-pci")
    assert gpu(host, gpu_results(pci=PCI_NVIDIA + second))["ownership_mode"] == "mixed"
    unbound = PCI_NVIDIA.replace("\tKernel driver in use: nvidia\n", "")
    fixtures = gpu_results(pci=unbound, kernel="NVRM: GPU has fallen off the bus\nGSP RPC timeout")
    fixtures["get_gpu_status"] = result(False, error_code=ErrorCode.COMMAND_FAILED)
    value = gpu(host.model_copy(update={"expected_gpu_count": 2}), fixtures)
    assert value["status"] == "warning" and value["ownership_mode"] == "unbound"
    assert {"GPU_UNBOUND", "NVIDIA_FALLEN_OFF_BUS", "NVIDIA_GSP_FAILURE"} <= set(value["signatures"])
    assert any("expected GPU count" in item for item in value["evidence"])
    assert value["devices"] == []


def test_unknown_expected_count_is_not_invented_and_probe_failure_fails_closed(host):
    value = gpu(host, gpu_results())
    assert value["expected_gpu_count"] is None
    assert not any("expected GPU count" in item for item in value["evidence"])
    failed = {name: result(False, error_code=ErrorCode.COMMAND_FAILED) for name in gpu_results()}
    assert gpu(host, failed)["status"] == "error"


def docker(host, service="active", listing=""):
    remote = FakeExecutor({
        "query_docker_diagnostics:service": result(stdout=f"LoadState=loaded\nActiveState={service}\n"),
        "query_docker_diagnostics:containers": result(stdout=listing),
    })
    return execute_diagnostic("query_docker_diagnostics", host, remote), remote


def test_docker_healthy_zero_and_warning_states_are_bounded(host):
    empty, _ = docker(host)
    assert empty["status"] == "healthy" and empty["total_containers"] == 0
    rows = "ok|Up 2 hours (healthy)|running\nloop|Restarting (1) 3 seconds ago|restarting\n"
    rows += "dead-one|Dead|dead\n" + "\n".join(f"c{i}|Up 1 hour|running" for i in range(20))
    value, _ = docker(host, listing=rows)
    assert value["status"] == "warning"
    assert value["restarting_containers"] == value["dead_containers"] == 1
    assert len(value["containers"]) == 16 and value["containers"][0]["health"] == "healthy"
    assert "DOCKER_FAILED" not in value["signatures"]


@pytest.mark.parametrize("listing", [
    "loop|Restarting (1) 3 seconds ago|restarting\n",
    "dead-one|Dead|dead\n",
    "sick|Up 1 minute (unhealthy)|running\n",
])
def test_active_service_container_warning_does_not_mean_docker_failed(host, listing):
    value, _ = docker(host, listing=listing)
    assert value["status"] == "warning"
    assert "DOCKER_FAILED" not in value["signatures"]
    assert value["evidence"]


@pytest.mark.parametrize("service", ["inactive", "failed"])
def test_docker_inactive_preserves_known_service_evidence(host, service):
    value, remote = docker(host, service=service)
    assert value["status"] == "unavailable" and value["service_state"] == service
    assert value["signatures"] == ["DOCKER_FAILED"]
    assert remote.calls == ["query_docker_diagnostics:service"]


def test_malformed_docker_rows_warn_without_leaking_unrequested_fields(host):
    value, remote = docker(host, listing="bad row\nsafe|mystery|unknown\n")
    assert value["status"] == "warning" and value["unknown_containers"] == 2
    serialized = json.dumps(value).casefold()
    assert all(secret not in serialized for secret in ("environment", "token", "mounts", "labels"))
    assert remote.calls == ["query_docker_diagnostics:service", "query_docker_diagnostics:containers"]


def test_read_apis_do_not_need_acquisition_catalog_entries(host):
    definitions = {item.capability_id for item in CAPABILITY_REGISTRY.definitions}
    assert "gpu_diagnostics" not in definitions
    assert "docker_diagnostics" not in definitions
    assert HOST_READ_CAPABILITIES["query_gpu_diagnostics"].risk_class == "READ_ONLY"
    assert HOST_READ_CAPABILITIES["query_docker_diagnostics"].risk_class == "READ_ONLY"
    discovery = discover(host)
    exposed = {item["name"] for item in discovery["capabilities"]}
    assert {"query_gpu_diagnostics", "query_docker_diagnostics"} <= exposed
    assert "not an intelligence boundary" in discovery["discovery_note"].casefold()
    assert TOOLS["get_pci_status"].command == ("lspci", "-Dnnk")


def test_docker_capability_is_unsupported_when_host_does_not_declare_it(host):
    host = host.model_copy(update={"capabilities": Capabilities(nvidia=True)})
    item = next(item for item in discover(host)["capabilities"]
                if item["name"] == "query_docker_diagnostics")
    assert item["available"] is False
