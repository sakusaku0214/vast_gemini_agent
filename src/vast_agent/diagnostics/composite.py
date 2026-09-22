"""Bounded, read-only diagnostic views built from existing probes and parsers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from vast_agent.actions.preflight import docker_workload_summary
from vast_agent.diagnostics.parser import build_observation
from vast_agent.execution.base import Executor
from vast_agent.models.host import Host
from vast_agent.models.observation import GpuDevice
from vast_agent.models.tool_result import ErrorCode, ToolResult
from vast_agent.tools.registry import run_tool

DiagnosticStatus = Literal["healthy", "warning", "degraded", "unavailable", "error"]
MAX_CONTAINERS = 16


class GpuDiagnosticResult(BaseModel):
    status: DiagnosticStatus
    nvml_status: Literal["available", "unavailable", "error", "not_expected"]
    pci_count: int | None = None
    expected_gpu_count: int | None = None
    nvidia_bound: int | None = None
    vfio_bound: int | None = None
    unbound: int | None = None
    unknown_bound: int | None = None
    ownership_mode: Literal["all_nvidia", "all_vfio", "mixed", "unbound", "unknown"]
    devices: list[GpuDevice] = Field(default_factory=list)
    d_state_process_count: int | None = None
    signatures: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    recommended_read_followup: list[str] = Field(default_factory=list)


class ContainerDiagnostic(BaseModel):
    name: str
    state: Literal["running", "restarting", "stopped", "created", "dead", "unknown"]
    health: Literal["healthy", "unhealthy", "starting"] | None = None


class DockerDiagnosticResult(BaseModel):
    status: DiagnosticStatus
    service_state: str
    total_containers: int | None = None
    running_containers: int | None = None
    restarting_containers: int | None = None
    stopped_containers: int | None = None
    created_containers: int | None = None
    dead_containers: int | None = None
    unknown_containers: int | None = None
    containers: list[ContainerDiagnostic] = Field(default_factory=list)
    signatures: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


def _ownership(gpu: object) -> str:
    count = gpu.pci_count
    if not gpu.pci_ok or count == 0 or gpu.unknown_bound:
        return "unknown"
    if gpu.unbound:
        return "unbound" if gpu.unbound == count else "mixed"
    if gpu.nvidia_bound == count:
        return "all_nvidia"
    if gpu.vfio_bound == count:
        return "all_vfio"
    return "mixed"


def gpu_diagnostics(host: Host, executor: Executor) -> GpuDiagnosticResult:
    names = ("host_ping", "get_pci_status", "get_gpu_status", "get_kernel_gpu_errors",
             "get_d_state_processes")
    results = {name: run_tool(name, host, executor) for name in names}
    if not results["host_ping"].success:
        return GpuDiagnosticResult(status="error", nvml_status="error",
            ownership_mode="unknown", evidence=["host connectivity probe failed"])
    observation = build_observation(host.name, results, host.expected_gpu_count)
    gpu = observation.gpu
    ownership = _ownership(gpu)
    nvml_result = results["get_gpu_status"]
    if gpu.nvml_ok:
        nvml_status = "available"
    elif nvml_result.error_code == ErrorCode.TOOL_UNSUPPORTED and ownership == "all_vfio":
        nvml_status = "not_expected"
    elif nvml_result.error_code == ErrorCode.TOOL_UNSUPPORTED:
        nvml_status = "unavailable"
    else:
        nvml_status = "error"
    evidence = [f"PCI enumeration found {gpu.pci_count} NVIDIA GPU(s)"] if gpu.pci_ok else []
    mismatch = host.expected_gpu_count is not None and gpu.pci_ok and gpu.pci_count != host.expected_gpu_count
    if mismatch:
        evidence.append(f"configured expected GPU count is {host.expected_gpu_count}")
    signatures = list(observation.signatures)
    if ownership == "all_vfio":
        signatures = [item for item in signatures
                      if item not in {"NVML_UNAVAILABLE", "GPU_BOUND_VFIO"}]
    else:
        signatures = [item for item in signatures if item != "GPU_BOUND_VFIO"]
    if not gpu.pci_ok and not gpu.nvml_ok:
        status: DiagnosticStatus = "unavailable" if all(
            result.error_code == ErrorCode.TOOL_UNSUPPORTED
            for result in (results["get_pci_status"], nvml_result)
        ) else "error"
    elif signatures or mismatch or gpu.unbound or gpu.unknown_bound:
        status = "warning"
    elif not gpu.pci_ok or (not gpu.nvml_ok and ownership != "all_vfio"):
        status = "degraded"
    else:
        status = "healthy"
    d_result = results["get_d_state_processes"]
    return GpuDiagnosticResult(
        status=status, nvml_status=nvml_status, pci_count=gpu.pci_count if gpu.pci_ok else None,
        expected_gpu_count=host.expected_gpu_count,
        nvidia_bound=gpu.nvidia_bound if gpu.pci_ok else None,
        vfio_bound=gpu.vfio_bound if gpu.pci_ok else None,
        unbound=gpu.unbound if gpu.pci_ok else None,
        unknown_bound=gpu.unknown_bound if gpu.pci_ok else None,
        ownership_mode=ownership, devices=gpu.devices[:16],
        d_state_process_count=observation.system.d_state_processes if d_result.success else None,
        signatures=signatures,
        evidence=evidence,
        recommended_read_followup=["kernel_gpu_errors"] if signatures else [],
    )


def _service_state(result: ToolResult) -> str:
    if not result.success:
        return "unknown"
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return values.get("ActiveState", "unknown").casefold()


def _container(line: str) -> ContainerDiagnostic | None:
    fields = line.split("|", 2)
    if len(fields) != 3 or not fields[0].strip():
        return None
    name, raw_status = fields[0].strip()[:80], fields[1].strip()
    folded = raw_status.casefold()
    if folded.startswith("restarting"):
        state = "restarting"
    elif folded.startswith("up"):
        state = "running"
    elif folded.startswith("exited"):
        state = "stopped"
    elif folded.startswith("created"):
        state = "created"
    elif folded.startswith("dead"):
        state = "dead"
    else:
        state = "unknown"
    health_raw = fields[2].strip().casefold()
    if "(unhealthy)" in folded:
        health_raw = "unhealthy"
    elif "(healthy)" in folded:
        health_raw = "healthy"
    elif "(health: starting)" in folded:
        health_raw = "starting"
    health = health_raw if health_raw in {"healthy", "unhealthy", "starting"} else None
    return ContainerDiagnostic(name=name, state=state, health=health)


def docker_diagnostics(host: Host, executor: Executor) -> DockerDiagnosticResult:
    service = _execute(executor, host, "query_docker_diagnostics:service",
        ("systemctl", "show", "docker.service", "--no-pager", "--property=LoadState,ActiveState,SubState"))
    service_state = _service_state(service)
    if service_state != "active":
        status: DiagnosticStatus = "unavailable" if service_state in {"inactive", "failed"} else "degraded"
        return DockerDiagnosticResult(status=status, service_state=service_state,
            evidence=[f"docker service state is {service_state}"])
    listing = _execute(executor, host, "query_docker_diagnostics:containers",
        ("docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}|{{.State}}"))
    if not listing.success:
        return DockerDiagnosticResult(status="degraded", service_state=service_state,
            evidence=["docker service is active but container listing failed"])
    lines = [line for line in listing.stdout.splitlines() if line.strip()]
    parsed = [_container(line) for line in lines]
    containers = [item for item in parsed if item is not None]
    malformed = len(lines) - len(containers)
    # Reuse the conservative workload classifier used by mutation preflights.
    legacy = "active\n" + "\n".join(
        f"id|{item.name}|{'Restarting' if item.state == 'restarting' else 'Up' if item.state == 'running' else item.state.title()}|image"
        for item in containers
    )
    workload = docker_workload_summary(legacy)
    counts = {state: sum(item.state == state for item in containers) for state in
              ("running", "restarting", "stopped", "created", "dead", "unknown")}
    unknown = counts["unknown"] + malformed
    warning = bool(counts["restarting"] or counts["dead"] or unknown or
                   any(item.health == "unhealthy" for item in containers))
    return DockerDiagnosticResult(
        status="warning" if warning else "healthy", service_state=service_state,
        total_containers=len(lines), running_containers=counts["running"],
        restarting_containers=counts["restarting"], stopped_containers=counts["stopped"],
        created_containers=counts["created"], dead_containers=counts["dead"],
        unknown_containers=max(unknown, workload.unknown_containers),
        containers=containers[:MAX_CONTAINERS],
        signatures=["DOCKER_FAILED"] if warning else [],
        evidence=[f"container details limited to {MAX_CONTAINERS}"] if len(containers) > MAX_CONTAINERS else [],
    )


def _execute(executor: Executor, host: Host, name: str, command: tuple[str, ...]) -> ToolResult:
    method = getattr(executor, "execute_tool", None)
    if method is not None:
        return method(name, host, command, 15)
    return executor.execute(host, command, 15)
