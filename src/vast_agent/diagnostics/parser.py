from __future__ import annotations

import re

from vast_agent.models.observation import (
    GpuDevice,
    GpuSummary,
    Observation,
    PciDevice,
    SystemSummary,
)
from vast_agent.models.tool_result import ErrorCode, ToolResult


def _observed(result: ToolResult | None) -> bool:
    """Whether a tool actually ran, as opposed to being unavailable by policy/config."""
    return result is not None and result.error_code != ErrorCode.TOOL_UNSUPPORTED


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
    # ``grep -A`` places ``--`` between non-adjacent matches.  Build blocks from
    # PCI headers instead of treating every unindented line as a new device, so
    # the separator cannot swallow the following device and its driver lines.
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\s", line, re.I):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current and line.strip() != "--":
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    nvidia_gpu_groups = {
        block.splitlines()[0].split()[0].rsplit(".", 1)[0]
        for block in blocks
        if any(token in block.splitlines()[0].lower() for token in ("vga", "3d controller"))
        and ("nvidia corporation" in block.splitlines()[0].lower()
             or re.search(r"\[10de:[0-9a-f]{4}\]", block.splitlines()[0], re.I))
    }
    for block in blocks:
        header = block.splitlines()[0]
        address_match = re.match(r"(\S+)", header)
        if not address_match:
            continue
        address = address_match.group(1)
        group = address.rsplit(".", 1)[0]
        header_low = header.lower()
        is_gpu = group in nvidia_gpu_groups and any(
            token in header_low for token in ("vga", "3d controller")
        )
        is_audio = group in nvidia_gpu_groups and "audio" in header_low
        if not (is_gpu or is_audio):
            continue
        match = re.search(r"Kernel driver in use:\s*(\S+)", block, re.I)
        driver = match.group(1) if match else None
        modules_match = re.search(r"Kernel modules:\s*(.+)", block, re.I)
        modules = [item.strip() for item in modules_match.group(1).split(",")] if modules_match else []
        values["pci_devices"].append(PciDevice(
            pci_address=address,
            device_type="gpu" if is_gpu else "audio",
            driver=driver,
            modules=modules,
            function_group=group,
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
    units: list[dict[str, str]] = []
    current: dict[str, str] = {"Id": "vastai.service"}
    for line in text.splitlines():
        if line.startswith("Id="):
            if len(current) > 1:
                units.append(current)
            current = {"Id": line[3:]}
        elif "=" in line:
            key, value = line.split("=", 1)
            current[key] = value
    if len(current) > 1:
        units.append(current)
    unavailable = {"not-found", "empty", "error"}
    return {
        unit["Id"].removesuffix(".service"): unit["ActiveState"]
        for unit in units
        if unit.get("LoadState", "loaded").casefold() not in unavailable
        and "ActiveState" in unit
    }


def summarize_docker_status(text: str) -> dict[str, int]:
    """Summarize the canonical ``get_docker_status`` output without retaining rows."""
    statuses = [
        fields[2].strip().casefold()
        for line in text.splitlines()[1:]
        if len(fields := line.split("|", 3)) == 4
    ]
    running = sum(status.startswith("up") for status in statuses)
    return {"total": len(statuses), "running": running, "stopped": len(statuses) - running}


def summarize_vm_status(text: str) -> dict[str, object]:
    """Summarize virsh domain rows from ``get_vm_status`` output."""
    known_states = {
        "running", "idle", "paused", "shutdown", "shut off", "crashed", "pmsuspended",
        "blocked",
    }
    domains: list[dict[str, str]] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:-|\d+)\s+(\S+)\s+(.+?)\s*$", line)
        if not match or match.group(2).casefold() not in known_states:
            continue
        domains.append({"name": match.group(1)[:80], "state": match.group(2).casefold()})
    return {
        "total": len(domains),
        "running": sum(domain["state"] == "running" for domain in domains),
        "domains": domains,
    }


def build_observation(
    host: str, results: dict[str, ToolResult], expected_gpu_count: int | None = None,
) -> Observation:
    ping = results.get("host_ping")
    ssh_ok = bool(ping and ping.success)
    gpu_result = results.get("get_gpu_status")
    nvml_ok, nvml_error, devices = (
        parse_gpu(gpu_result) if _observed(gpu_result) else (False, None, [])
    )
    pci_result = results.get("get_pci_status")
    pci = parse_pci(pci_result.stdout) if _observed(pci_result) else parse_pci("")
    d_result = results.get("get_d_state_processes")
    d_text = d_result.stdout if d_result else ""
    d_count = sum(1 for line in d_text.splitlines() if line.lstrip().startswith("D"))
    health_result = results.get("get_system_health")
    health = health_result.stdout if health_result else ""
    percentages = [int(x) for x in re.findall(r"\b(\d{1,3})%", health)]
    failed = sum(1 for line in health.splitlines() if "failed" in line.lower())
    service_result = results.get("get_service_status")
    services = parse_service_show(service_result.stdout) if service_result else {}
    services_observed = [
        name for name in ("get_service_status", "get_vast_status", "get_docker_status")
        if _observed(results.get(name))
    ]
    vast = results.get("get_vast_status")
    if _observed(vast): services.update(parse_service_show(vast.stdout))
    docker = results.get("get_docker_status")
    if _observed(docker) and docker.stdout.strip():
        services["docker"] = docker.stdout.splitlines()[0].strip()
    observation = Observation(
        host=host, ssh_ok=ssh_ok, ssh_observed=ping is not None,
        gpu=GpuSummary(
            **pci,
            expected_gpu_count=expected_gpu_count,
            nvml_observed=_observed(gpu_result),
            pci_observed=_observed(pci_result),
            pci_ok=bool(pci_result and pci_result.success),
            nvml_ok=nvml_ok,
            nvml_error=nvml_error,
            devices=devices,
        ),
        system=SystemSummary(
            health_observed=health_result is not None,
            d_state_observed=d_result is not None,
            d_state_processes=d_count,
            filesystem_max_percent=max(percentages) if percentages else None,
            failed_units=failed,
        ),
        services=services,
        services_observed=services_observed,
        observed_tools=list(results),
        details={
            key: results[tool].stdout[:8000]
            for key, tool in (
                ("kernel_gpu_errors", "get_kernel_gpu_errors"),
                ("vast_logs", "get_vast_logs"),
                ("journal_errors", "get_journal_errors"),
                ("docker_status", "get_docker_status"),
                ("vm_status", "get_vm_status"),
            )
            if tool in results
        },
    )
    from vast_agent.diagnostics.signatures import detect_signatures
    observation.signatures = [item.value for item in detect_signatures(observation)]
    return observation
