from __future__ import annotations

import re

from vast_agent.models.observation import (
    GpuDevice,
    GpuSummary,
    Observation,
    PciDevice,
    SystemSummary,
)
from vast_agent.models.tool_result import ToolResult


def _integer(value: str) -> int | None:
    try: return int(value.strip())
    except ValueError: return None


def parse_gpu(result: ToolResult) -> tuple[bool, str | None, list[GpuDevice]]:
    if not result.success:
        return False, (result.stderr or "nvidia-smi failed").strip()[:500], []
    devices = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 9: continue
        index = _integer(fields[0])
        if index is None: continue
        devices.append(GpuDevice(
            index=index, uuid=fields[1], model=fields[2], temperature_c=_integer(fields[3]),
            utilization_percent=_integer(fields[4]), vram_used_mb=_integer(fields[5]),
            vram_total_mb=_integer(fields[6]), power_state=fields[7], pci_bus=fields[8],
        ))
    return True, None, devices


def parse_pci(text: str) -> dict[str, object]:
    values: dict[str, object] = {
        "pci_count": 0,
        "nvidia_bound": 0,
        "vfio_bound": 0,
        "unbound": 0,
        "unknown_bound": 0,
        "pci_devices": [],
    }
    blocks = re.split(r"\n(?=\S)", text.strip()) if text.strip() else []
    for block in blocks:
        low = block.lower()
        header = block.splitlines()[0]
        address_match = re.match(r"(\S+)", header)
        is_gpu = "vga" in low or "3d controller" in low
        is_audio = "audio" in low
        if not address_match or not (is_gpu or is_audio):
            continue
        match = re.search(r"Kernel driver in use:\s*(\S+)", block, re.I)
        driver = match.group(1) if match else None
        modules_match = re.search(r"Kernel modules:\s*(.+)", block, re.I)
        modules = [item.strip() for item in modules_match.group(1).split(",")] if modules_match else []
        address = address_match.group(1)
        values["pci_devices"].append(PciDevice(
            pci_address=address,
            device_type="gpu" if is_gpu else "audio",
            driver=driver,
            modules=modules,
            function_group=address.rsplit(".", 1)[0],
        ))
        if not is_gpu:
            continue
        values["pci_count"] += 1
        if not driver: values["unbound"] += 1
        elif driver == "nvidia": values["nvidia_bound"] += 1
        elif driver == "vfio-pci": values["vfio_bound"] += 1
        else: values["unknown_bound"] += 1
    return values


def parse_service_show(text: str) -> dict[str, str]:
    services: dict[str, str] = {}
    current = "vastai"
    for line in text.splitlines():
        if line.startswith("Id="):
            current = line[3:].removesuffix(".service")
        elif line.startswith("ActiveState="):
            services[current] = line.split("=", 1)[1]
    return services


def build_observation(host: str, results: dict[str, ToolResult]) -> Observation:
    ping = results.get("host_ping")
    ssh_ok = bool(ping and ping.success)
    gpu_result = results.get("get_gpu_status", ToolResult(success=False, duration_ms=0))
    nvml_ok, nvml_error, devices = parse_gpu(gpu_result)
    pci = parse_pci(results.get("get_pci_status", ToolResult(success=False, duration_ms=0)).stdout)
    d_text = results.get("get_d_state_processes", ToolResult(success=False, duration_ms=0)).stdout
    d_count = sum(1 for line in d_text.splitlines() if line.lstrip().startswith("D"))
    health = results.get("get_system_health", ToolResult(success=False, duration_ms=0)).stdout
    percentages = [int(x) for x in re.findall(r"\b(\d{1,3})%", health)]
    failed = sum(1 for line in health.splitlines() if "failed" in line.lower())
    services = parse_service_show(results.get("get_service_status", ToolResult(success=False, duration_ms=0)).stdout)
    vast = results.get("get_vast_status")
    if vast: services.update(parse_service_show(vast.stdout))
    docker = results.get("get_docker_status")
    if docker and docker.stdout.strip(): services["docker"] = docker.stdout.splitlines()[0].strip()
    observation = Observation(
        host=host, ssh_ok=ssh_ok,
        gpu=GpuSummary(**pci, nvml_ok=nvml_ok, nvml_error=nvml_error, devices=devices),
        system=SystemSummary(
            d_state_processes=d_count,
            filesystem_max_percent=max(percentages) if percentages else None,
            failed_units=failed,
        ),
        services=services,
        details={
            "kernel_gpu_errors": results.get("get_kernel_gpu_errors", ToolResult(success=False, duration_ms=0)).stdout[:8000],
            "vast_logs": results.get("get_vast_logs", ToolResult(success=False, duration_ms=0)).stdout[:8000],
            "journal_errors": results.get("get_journal_errors", ToolResult(success=False, duration_ms=0)).stdout[:8000],
        },
    )
    from vast_agent.diagnostics.signatures import detect_signatures
    observation.signatures = [item.value for item in detect_signatures(observation)]
    return observation
