from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Any, Final

from pydantic import BaseModel

from vast_agent.actions.package_catalog import CAPABILITY_REGISTRY
from vast_agent.agent.generic_read import run_validated_read
from vast_agent.agent.models import (
    CliArgvArgs,
    CollectEvidenceArgs,
    ExecutableQueryArgs,
    HostArgument,
    InspectHostArgs,
    InterfaceInspectionArgs,
    NetworkInspectionArgs,
    NvmeHealthArgs,
    PackageQueryArgs,
    ProcessQueryArgs,
    RecentIncidentsArgs,
    ServiceQueryArgs,
    TrafficHistoryArgs,
)
from vast_agent.diagnostics.composite import docker_diagnostics, gpu_diagnostics
from vast_agent.execution.base import Executor
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult


@dataclass(frozen=True)
class HostReadCapability:
    name: str
    description: str
    category: str
    argument_model: type[BaseModel]
    executor: str
    required_host_capabilities: tuple[str, ...] = ()
    risk_class: str = "READ_ONLY"
    available: bool = True

    def declaration(self) -> dict[str, object]:
        return {
            "type": "function", "name": self.name, "description": self.description,
            "parameters": self.argument_model.model_json_schema(),
        }


HOST_READ_CAPABILITIES: Final[dict[str, HostReadCapability]] = {
    item.name: item for item in (
        HostReadCapability("inspect_host", "Inspect an allowlisted host scope read-only.", "diagnostics", InspectHostArgs, "legacy"),
        HostReadCapability("collect_evidence", "Collect a fixed type of read-only evidence.", "diagnostics", CollectEvidenceArgs, "legacy"),
        HostReadCapability("get_recent_incidents", "Read compact incident summaries when history is relevant.", "history", RecentIncidentsArgs, "legacy"),
        HostReadCapability("query_package", "Check whether one validated Debian package is installed.", "packages", PackageQueryArgs, "generic"),
        HostReadCapability("query_executable", "Resolve one validated executable name without exposing PATH. Include continuation_argv when PATH discovery should immediately continue the requested READ.", "executables", ExecutableQueryArgs, "generic"),
        HostReadCapability("query_cli_help", "Read bounded CLI help using executable plus argv; supports typed non-interactive sudo, never shell.", "discovery", CliArgvArgs, "generic"),
        HostReadCapability("run_readonly_argv", "Run a validated READ argv, optionally with typed non-interactive sudo; mutations become Proposal candidates.", "discovery", CliArgvArgs, "generic"),
        HostReadCapability("query_service", "Read one validated systemd service state.", "services", ServiceQueryArgs, "generic"),
        HostReadCapability("inspect_network", "Read bounded link, address, or route information.", "network", NetworkInspectionArgs, "generic"),
        HostReadCapability("inspect_interface", "Read bounded details for one validated network interface.", "network", InterfaceInspectionArgs, "generic"),
        HostReadCapability("query_traffic_history", "Read bounded vnStat traffic history in bytes.", "network", TrafficHistoryArgs, "generic"),
        HostReadCapability("query_nvme_health", "Read selected NVMe SMART health fields.", "storage", NvmeHealthArgs, "generic"),
        HostReadCapability("query_process", "Check a process name; returns only count and a few PIDs.", "processes", ProcessQueryArgs, "generic"),
        HostReadCapability("query_gpu_diagnostics", "Return bounded PCI, NVML, and known-signature GPU diagnostics.", "gpu", HostArgument, "diagnostic"),
        HostReadCapability("query_docker_diagnostics", "Return bounded Docker service and container-state diagnostics.", "containers", HostArgument, "diagnostic", ("docker",)),
        HostReadCapability("inspect_os", "Read distribution, kernel, architecture, and uptime facts.", "system", HostArgument, "generic"),
        HostReadCapability("list_host_capabilities", "List registry READ capabilities available for this host.", "discovery", HostArgument, "discovery"),
    )
}

FUNCTION_DECLARATIONS: Final = [
    capability.declaration() for capability in HOST_READ_CAPABILITIES.values() if capability.available
]


def _run(remote: Executor, name: str, host: Host, argv: Sequence[str], timeout: int = 15) -> ToolResult:
    execute_tool = getattr(remote, "execute_tool", None)
    if execute_tool is not None:
        return execute_tool(name, host, tuple(argv), timeout)
    return remote.execute(host, tuple(argv), timeout)


def _error(result: ToolResult) -> dict[str, object]:
    if result.exit_code == 127:
        code = "COMMAND_NOT_AVAILABLE"
    else:
        code = str(result.error_code or "COMMAND_FAILED")
    return {"error": code, "exit_code": result.exit_code}


def _json(result: ToolResult) -> Any:
    if not result.success:
        return _error(result)
    try:
        return json.loads(result.stdout or "[]")
    except (json.JSONDecodeError, TypeError):
        return {"error": "PARSER_FAILED"}


def _resolve_executable(
    remote: Executor, host: Host, executable: str, *, tool_name: str = "query_executable",
) -> tuple[str | None, ToolResult]:
    """Resolve an executable with a bounded, shell-free set of probes.

    This is shared by explicit discovery and deterministic readers so a fast path does
    not become a separate capability silo.  Candidate locations are code-owned, while
    each host is probed independently.
    """
    result = _run(remote, tool_name, host, ("which", "--", executable))
    path = result.stdout.splitlines()[0].strip() if result.success and result.stdout.strip() else None
    if path and (not path.startswith("/") or len(path) > 256 or any(char.isspace() for char in path)):
        path = None
    if path is None:
        candidates = (
            f"/home/{host.ssh_user}/.local/bin/{executable}",
            f"/home/{host.ssh_user}/bin/{executable}",
            f"/usr/local/bin/{executable}", f"/usr/bin/{executable}",
            f"/snap/bin/{executable}",
        )
        for candidate in candidates:
            probe = _run(remote, f"{tool_name}:candidate", host, ("test", "-x", candidate))
            if probe.success:
                return candidate, result
    return path, result


def _traffic_timestamp(row: dict[str, object], timezone: tzinfo) -> datetime | None:
    date = row.get("date")
    if not isinstance(date, dict):
        return None
    try:
        hour = row.get("time") if isinstance(row.get("time"), dict) else {}
        return datetime(
            int(date["year"]), int(date["month"]), int(date["day"]),
            int(hour.get("hour", 0)), int(hour.get("minute", 0)), tzinfo=timezone,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _traffic_history(args: BaseModel, host: Host, remote: Executor) -> dict[str, object]:
    period = args.period  # type: ignore[attr-defined]
    interface = args.interface  # type: ignore[attr-defined]
    mode = "h" if period == "last_24h" else "d"
    argv = ("vnstat", "--json", mode) + (("-i", interface) if interface else ())
    result = _run(remote, "query_traffic_history", host, argv)
    if result.exit_code == 127 or "command not found" in result.stderr.casefold():
        executable, _ = _resolve_executable(
            remote, host, "vnstat", tool_name="query_traffic_history:discovery",
        )
        if executable is None:
            return {"status": "unsupported", "period": period, "interface": interface,
                    "source": "vnstat", "error": "EXECUTABLE_DISCOVERY_FAILED",
                    "failure_kind": "executable_discovery_failed"}
        argv = (executable, *argv[1:])
        result = _run(remote, "query_traffic_history:resolved", host, argv)
    if not result.success:
        status = "unsupported" if result.exit_code == 127 else "error"
        failure_kind = "command_not_found" if result.exit_code == 127 else "command_failed"
        return {"status": status, "period": period, "interface": interface, "source": "vnstat",
                "failure_kind": failure_kind, **_error(result)}
    try:
        payload = json.loads(result.stdout)
        interfaces = payload["interfaces"]
        if not isinstance(interfaces, list):
            raise TypeError
        candidates = [item for item in interfaces if isinstance(item, dict) and isinstance(item.get("name"), str)]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"status": "error", "period": period, "interface": interface, "source": "vnstat", "error": "PARSER_FAILED", "failure_kind": "parser_failed"}
    if interface:
        candidates = [item for item in candidates if item["name"] == interface]
    if not candidates:
        return {"status": "no_data", "period": period, "interface": interface, "source": "vnstat", "failure_kind": "no_data"}
    if len(candidates) != 1:
        return {"status": "error", "period": period, "interface": None, "source": "vnstat", "error": "AMBIGUOUS_INTERFACE", "failure_kind": "ambiguous_interface", "candidates": [item["name"] for item in candidates[:16]]}
    selected = candidates[0]
    traffic = selected.get("traffic")
    rows = traffic.get("hour" if mode == "h" else "day") if isinstance(traffic, dict) else None
    if not isinstance(rows, list):
        return {"status": "error", "period": period, "interface": selected["name"], "source": "vnstat", "error": "UNSUPPORTED_JSON"}
    clock = _run(remote, "query_traffic_history:clock", host, ("date", "--iso-8601=seconds"))
    if not clock.success:
        return {"status": "error", "period": period, "interface": selected["name"],
                "source": "vnstat", "error": "HOST_TIME_UNAVAILABLE", "failure_kind": "host_time_invalid"}
    try:
        value = clock.stdout.strip()
        if len(value) > 64 or "\n" in value or "\r" in value:
            raise ValueError
        now = datetime.fromisoformat(value)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError
    except (ValueError, OverflowError):
        return {"status": "error", "period": period, "interface": selected["name"],
                "source": "vnstat", "error": "HOST_TIME_INVALID", "failure_kind": "host_time_invalid"}
    dated: list[tuple[datetime, dict[str, object]]] = []
    for row in rows:
        if isinstance(row, dict) and (stamp := _traffic_timestamp(row, now.tzinfo)) is not None:
            dated.append((stamp, row))
    if period == "today":
        chosen = [(stamp, row) for stamp, row in dated if stamp.date() == now.date()]
    elif period == "yesterday":
        chosen = [(stamp, row) for stamp, row in dated if stamp.date() == (now - timedelta(days=1)).date()]
    elif period == "last_24h":
        chosen = [(stamp, row) for stamp, row in dated if now - timedelta(hours=24) <= stamp <= now]
    else:
        chosen = [(stamp, row) for stamp, row in dated if now.date() - timedelta(days=6) <= stamp.date() <= now.date()]
    if not chosen:
        return {"status": "no_data", "period": period, "interface": selected["name"], "source": "vnstat", "failure_kind": "no_data"}
    try:
        rx = sum(int(row["rx"]) for _, row in chosen)
        tx = sum(int(row["tx"]) for _, row in chosen)
    except (KeyError, TypeError, ValueError):
        return {"status": "error", "period": period, "interface": selected["name"], "source": "vnstat", "error": "UNSUPPORTED_JSON"}
    stamps = [stamp for stamp, _ in chosen]
    return {"status": "available", "interface": selected["name"], "period": period,
            "rx_bytes": rx, "tx_bytes": tx, "total_bytes": rx + tx,
            "sample_start": min(stamps).isoformat(), "sample_end": max(stamps).isoformat(), "source": "vnstat"}


def _temperature_celsius(value: object) -> float:
    """Normalize nvme-cli's version-dependent Celsius/Kelvin temperature value.

    Values above the plausible Celsius range are accepted only in a conservative Kelvin range.
    Zero and ambiguous/out-of-range values fail closed instead of producing a misleading result.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-numeric temperature")
    temperature = float(value)
    if 0 < temperature <= 200:
        return round(temperature, 2)
    if 250 <= temperature <= 500:
        return round(temperature - 273.15, 2)
    raise ValueError("unsupported temperature value")


def _nvme_health(args: BaseModel, host: Host, remote: Executor) -> dict[str, object]:
    device = args.device  # type: ignore[attr-defined]
    if device is None:
        listing = _run(remote, "query_nvme_health:list", host, ("nvme", "list", "-o", "json"))
        if not listing.success:
            status = "unsupported" if listing.exit_code == 127 else "error"
            return {"status": status, "device": None, **_error(listing)}
        try:
            payload = json.loads(listing.stdout)
            devices = payload["Devices"]
            candidates = sorted({f"/dev/{match.group(1)}" for item in devices
                                 if isinstance(item, dict) and isinstance(item.get("DevicePath"), str)
                                 if (match := re.fullmatch(r"/dev/(nvme\d+)(?:n\d+)?", item["DevicePath"]))})
        except (json.JSONDecodeError, KeyError, TypeError):
            return {"status": "error", "device": None, "error": "PARSER_FAILED"}
        if not candidates:
            return {"status": "unsupported", "device": None, "error": "NO_NVME_DEVICE"}
        if len(candidates) != 1:
            return {"status": "error", "device": None, "error": "AMBIGUOUS_DEVICE", "candidates": candidates[:16]}
        device = candidates[0]
    smart = _run(remote, "query_nvme_health:smart", host, ("nvme", "smart-log", "-o", "json", device))
    if not smart.success:
        status = "unsupported" if smart.exit_code == 127 else "error"
        return {"status": status, "device": device, **_error(smart)}
    try:
        raw = json.loads(smart.stdout)
        fields = {
            "temperature_c": _temperature_celsius(raw["temperature"]),
            "available_spare_percent": raw["avail_spare"],
            "available_spare_threshold_percent": raw["spare_thresh"],
            "percentage_used": raw["percent_used"],
            "data_units_read": raw["data_units_read"],
            "data_units_written": raw["data_units_written"],
            "power_on_hours": raw["power_on_hours"],
            "unsafe_shutdowns": raw["unsafe_shutdowns"],
            "media_errors": raw["media_errors"],
            "num_err_log_entries": raw["num_err_log_entries"],
            "critical_warning": raw["critical_warning"],
        }
        if any(isinstance(value, (dict, list, bool)) for value in fields.values()):
            raise TypeError
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"status": "error", "device": device, "error": "UNSUPPORTED_JSON"}
    return {"status": "available", "device": device, **fields}


def _network_summary(part: str, rows: list[object]) -> list[object]:
    """Keep useful network facts while omitting MACs, aliases, and other raw fields."""
    bounded = [row for row in rows if isinstance(row, dict)][:32]
    if part == "interfaces":
        keys = ("ifindex", "ifname", "flags", "mtu", "operstate", "link_type")
        output = []
        for row in bounded:
            item = {key: row.get(key) for key in keys if key in row}
            statistics = row.get("stats64") or row.get("stats")
            if isinstance(statistics, dict):
                item["statistics"] = {
                    direction: {
                        key: values.get(key) for key in ("bytes", "packets", "errors", "dropped")
                        if key in values
                    }
                    for direction in ("rx", "tx")
                    if isinstance((values := statistics.get(direction)), dict)
                }
            output.append(item)
        return output
    if part == "addresses":
        output = []
        for row in bounded:
            addresses = row.get("addr_info", [])
            safe_addresses = [
                {key: address.get(key) for key in ("family", "local", "prefixlen", "scope") if key in address}
                for address in addresses[:16] if isinstance(address, dict)
            ]
            output.append({"ifname": row.get("ifname"), "operstate": row.get("operstate"), "addresses": safe_addresses})
        return output
    keys = ("dst", "gateway", "dev", "protocol", "scope", "metric", "prefsrc")
    return [{key: row.get(key) for key in keys if key in row} for row in bounded]


def execute_generic(name: str, args: BaseModel, host: Host, remote: Executor) -> dict[str, object]:
    if name in {"query_cli_help", "run_readonly_argv"}:
        return run_validated_read(
            remote, host, args.executable, args.argv,  # type: ignore[attr-defined]
            help_only=name == "query_cli_help",
            requires_sudo=args.requires_sudo,  # type: ignore[attr-defined]
        )
    if name == "query_traffic_history":
        return _traffic_history(args, host, remote)
    if name == "query_nvme_health":
        return _nvme_health(args, host, remote)
    if name == "query_package":
        package = args.package_name  # type: ignore[attr-defined]
        result = _run(remote, name, host, ("dpkg-query", "-W", "-f=${db:Status-Status}\\t${Version}\\n", "--", package))
        if not result.success:
            if result.exit_code == 1:
                return {"package": package, "installed": False, "status": "not-installed", "version": None}
            return {"package": package, "installed": None, **_error(result)}
        status, _, version = result.stdout.strip().partition("\t")
        return {"package": package, "installed": status == "installed", "status": status, "version": version or None}
    if name == "query_executable":
        executable = args.executable_name  # type: ignore[attr-defined]
        path, result = _resolve_executable(remote, host, executable)
        if path is None and result.exit_code not in {0, 1, 127}:
            return {"executable": executable, "exists": None, **_error(result)}
        return {"executable": executable, "exists": bool(path), "path": path}
    if name == "query_service":
        supplied = args.service_name  # type: ignore[attr-defined]
        unit = supplied if supplied.endswith(".service") else f"{supplied}.service"
        result = _run(remote, name, host, ("systemctl", "show", unit, "--no-pager", "--property=LoadState,ActiveState,SubState,MainPID"))
        if not result.success:
            return {"service": unit, "exists": None, **_error(result)}
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        return {"service": unit, "exists": values.get("LoadState") != "not-found", "load_state": values.get("LoadState"), "active_state": values.get("ActiveState"), "sub_state": values.get("SubState"), "main_pid": int(values.get("MainPID", "0") or 0)}
    if name == "inspect_network":
        scope = args.scope  # type: ignore[attr-defined]
        commands = {"interfaces": ("ip", "-j", "link", "show"), "addresses": ("ip", "-j", "address", "show"), "routes": ("ip", "-j", "route", "show")}
        selected = tuple(commands) if scope == "summary" else (scope,)
        output: dict[str, object] = {"scope": scope}
        for part in selected:
            parsed = _json(_run(remote, f"{name}:{part}", host, commands[part]))
            if isinstance(parsed, list):
                parsed = _network_summary(part, parsed)
            output[part] = parsed
        return output
    if name == "inspect_interface":
        interface = args.interface_name  # type: ignore[attr-defined]
        link = _json(_run(remote, f"{name}:link", host, ("ip", "-j", "-s", "link", "show", "dev", interface)))
        if isinstance(link, list):
            link = _network_summary("interfaces", link)
        details = _run(remote, f"{name}:details", host, ("ethtool", interface))
        safe_details = [line.strip() for line in details.stdout.splitlines() if line.strip().startswith(("Speed:", "Duplex:", "Link detected:"))]
        result: dict[str, object] = {"interface": interface, "link": link, "details": safe_details[:8]}
        if not details.success:
            result["details_error"] = _error(details)["error"]
        return result
    if name == "query_process":
        process = args.process_name  # type: ignore[attr-defined]
        result = _run(remote, name, host, ("pgrep", "-x", "--", process))
        if result.exit_code == 1:
            return {"process": process, "running": False, "count": 0, "pids": []}
        if not result.success:
            return {"process": process, "running": None, **_error(result)}
        pids = [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()][:10]
        return {"process": process, "running": bool(pids), "count": len(pids), "pids": pids}
    if name == "inspect_os":
        release = _run(remote, f"{name}:release", host, ("cat", "/etc/os-release"))
        kernel = _run(remote, f"{name}:kernel", host, ("uname", "-srmo"))
        uptime = _run(remote, f"{name}:uptime", host, ("uptime", "-p"))
        if not release.success:
            return _error(release)
        allowed = {"ID", "PRETTY_NAME", "VERSION_ID"}
        facts = {key.lower(): value.strip().strip('"') for line in release.stdout.splitlines() if "=" in line for key, value in [line.split("=", 1)] if key in allowed}
        facts.update({"kernel": kernel.stdout.strip() if kernel.success else None, "uptime": uptime.stdout.strip() if uptime.success else None})
        return facts
    return {"error": "FUNCTION_NOT_ALLOWED", "function": name}


def execute_diagnostic(name: str, host: Host, remote: Executor) -> dict[str, object]:
    if name == "query_gpu_diagnostics":
        return gpu_diagnostics(host, remote).model_dump(mode="json")
    if name == "query_docker_diagnostics":
        return docker_diagnostics(host, remote).model_dump(mode="json")
    return {"status": "error", "error": "FUNCTION_NOT_ALLOWED"}


def discover(host: Host) -> dict[str, object]:
    """Describe callable READ APIs plus optional acquisition metadata.

    This is intentionally not a complete list of concepts the model may understand;
    generic executable/help discovery extends beyond these metadata entries.
    """
    capabilities = []
    for item in HOST_READ_CAPABILITIES.values():
        available = item.available and all(getattr(host.capabilities, cap, False) for cap in item.required_host_capabilities)
        capabilities.append({"name": item.name, "risk_class": item.risk_class,
                             "available": available})
    acquisition_metadata = []
    for definition in CAPABILITY_REGISTRY.definitions:
        backend_name = definition.read_backend.tool_name if definition.read_backend else None
        backend = HOST_READ_CAPABILITIES.get(backend_name) if backend_name else None
        acquisition = definition.acquisition
        acquisition_metadata.append({
            "capability_id": definition.capability_id,
            "backend": backend_name,
            "backend_registered": backend is not None,
            "acquisition": acquisition.package_name if acquisition else None,
        })
    return {"host": host.name,
            # Compatibility key retained; it is the callable READ API surface.
            "capabilities": capabilities, "acquisition_metadata": acquisition_metadata,
            "high_level_capabilities": acquisition_metadata,
            "discovery_note": "Not an intelligence boundary; generic CLI discovery is available."}


# Fail fast if a required declarative backend is absent or unsafe. Optional future backends may
# be registered with required=False and will be exposed as unavailable by discovery.
CAPABILITY_REGISTRY.validate(HOST_READ_CAPABILITIES)
