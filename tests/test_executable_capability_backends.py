from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from vast_agent.actions.capability_backends import CapabilityBackendResolver
from vast_agent.actions.capability_bridge import assess_gap
from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.host_read import HOST_READ_CAPABILITIES
from vast_agent.agent.models import CapabilityGap
from vast_agent.config import HostRegistry
from vast_agent.execution.base import FakeExecutor
from vast_agent.models.tool_result import ErrorCode, ToolResult
from vast_agent.services.inspection import InspectionService
from vast_agent.storage.database import Database


def functions_for(tmp_path, host, results):
    database = Database(tmp_path / "backends.db")
    database.migrate()
    remote = FakeExecutor(results)
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}),
        InspectionService(database, tmp_path / "logs"), database, remote,
    )
    return functions, remote


def evidence(result):
    return result["untrusted_evidence"]


def vnstat_day(day, rx=100, tx=50):
    return {"date": {"year": day.year, "month": day.month, "day": day.day}, "rx": rx, "tx": tx}


def traffic_payload(interfaces):
    return json.dumps({"interfaces": interfaces})


def test_yesterday_traffic_is_structured_in_bytes(tmp_path, host):
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    payload = traffic_payload([{"name": "eth0", "traffic": {"day": [vnstat_day(yesterday)]}}])
    functions, remote = functions_for(tmp_path, host, {
        "query_traffic_history": ToolResult(success=True, stdout=payload, duration_ms=1),
    })
    result = evidence(functions.execute("query_traffic_history", {
        "host": host.name, "period": "yesterday",
    }, host.name))
    assert result["status"] == "available"
    assert (result["rx_bytes"], result["tx_bytes"], result["total_bytes"]) == (100, 50, 150)
    assert result["interface"] == "eth0"
    assert remote.calls == ["query_traffic_history"]


def test_yesterday_without_a_sample_is_no_data_not_zero(tmp_path, host):
    payload = traffic_payload([{"name": "eth0", "traffic": {"day": []}}])
    functions, _ = functions_for(tmp_path, host, {
        "query_traffic_history": ToolResult(success=True, stdout=payload, duration_ms=1),
    })
    result = evidence(functions.execute("query_traffic_history", {
        "host": host.name, "period": "yesterday",
    }, host.name))
    assert result["status"] == "no_data"
    assert "total_bytes" not in result


@pytest.mark.parametrize("tool_result, status", [
    (ToolResult(success=False, exit_code=127, duration_ms=1), "unsupported"),
    (ToolResult(success=True, stdout="not-json", duration_ms=1), "error"),
    (ToolResult(success=False, error_code=ErrorCode.SSH_TIMEOUT, duration_ms=15000), "error"),
])
def test_traffic_failures_do_not_fall_back(tmp_path, host, tool_result, status):
    functions, remote = functions_for(tmp_path, host, {"query_traffic_history": tool_result})
    result = evidence(functions.execute("query_traffic_history", {
        "host": host.name, "period": "today",
    }, host.name))
    assert result["status"] == status
    assert remote.calls == ["query_traffic_history"]


def test_multiple_traffic_interfaces_are_ambiguous(tmp_path, host):
    payload = traffic_payload([
        {"name": "eth0", "traffic": {"day": []}},
        {"name": "eth1", "traffic": {"day": []}},
    ])
    functions, _ = functions_for(tmp_path, host, {
        "query_traffic_history": ToolResult(success=True, stdout=payload, duration_ms=1),
    })
    result = evidence(functions.execute("query_traffic_history", {
        "host": host.name, "period": "last_7d",
    }, host.name))
    assert result["error"] == "AMBIGUOUS_INTERFACE"
    assert result["candidates"] == ["eth0", "eth1"]


def test_traffic_interface_uses_code_owned_exact_argv(tmp_path, host):
    class RecordingExecutor:
        def __init__(self):
            self.commands = []

        def execute(self, actual_host, command, timeout, cancellation=None):
            self.commands.append((tuple(command), timeout))
            return ToolResult(success=True, stdout=traffic_payload([
                {"name": "eth1", "traffic": {"hour": []}},
            ]), duration_ms=1)

    database = Database(tmp_path / "traffic-argv.db")
    database.migrate()
    remote = RecordingExecutor()
    functions = FunctionExecutor(HostRegistry(hosts={host.name: host}),
        InspectionService(database, tmp_path / "logs"), database, remote)
    functions.execute("query_traffic_history", {
        "host": host.name, "period": "last_24h", "interface": "eth1",
    }, host.name)
    assert remote.commands == [(('vnstat', '--json', 'h', '-i', 'eth1'), 15)]
    assert remote.commands[0][0][0] not in {"sh", "bash"}


@pytest.mark.parametrize("interface", [";reboot", "../../x", "eth0\nreboot"])
def test_invalid_traffic_interface_never_executes(tmp_path, host, interface):
    functions, remote = functions_for(tmp_path, host, {})
    result = functions.execute("query_traffic_history", {
        "host": host.name, "period": "today", "interface": interface,
    }, host.name)
    assert result["error"] == "INVALID_ARGUMENTS"
    assert remote.calls == []


def smart_payload():
    return json.dumps({
        "temperature": 310, "avail_spare": 99, "spare_thresh": 10,
        "percent_used": 3, "data_units_read": 1000, "data_units_written": 2000,
        "power_on_hours": 300, "unsafe_shutdowns": 2, "media_errors": 1,
        "num_err_log_entries": 4, "critical_warning": 0,
        "serial_number": "must-not-leak", "model_number": "private-model",
    })


def test_one_nvme_is_discovered_and_smart_fields_are_bounded(tmp_path, host):
    functions, remote = functions_for(tmp_path, host, {
        "query_nvme_health:list": ToolResult(success=True, stdout=json.dumps({
            "Devices": [{"DevicePath": "/dev/nvme1n1", "SerialNumber": "secret"}],
        }), duration_ms=1),
        "query_nvme_health:smart": ToolResult(success=True, stdout=smart_payload(), duration_ms=1),
    })
    result = evidence(functions.execute("query_nvme_health", {"host": host.name}, host.name))
    assert result["status"] == "available" and result["device"] == "/dev/nvme1"
    assert result["percentage_used"] == 3 and result["critical_warning"] == 0
    assert "serial" not in json.dumps(result).casefold()
    assert remote.calls == ["query_nvme_health:list", "query_nvme_health:smart"]


def test_multiple_nvme_devices_are_ambiguous(tmp_path, host):
    functions, remote = functions_for(tmp_path, host, {
        "query_nvme_health:list": ToolResult(success=True, stdout=json.dumps({
            "Devices": [{"DevicePath": "/dev/nvme0n1"}, {"DevicePath": "/dev/nvme1n1"}],
        }), duration_ms=1),
    })
    result = evidence(functions.execute("query_nvme_health", {"host": host.name}, host.name))
    assert result["error"] == "AMBIGUOUS_DEVICE"
    assert remote.calls == ["query_nvme_health:list"]


@pytest.mark.parametrize("device", ["/dev/sda", "../../nvme0", "nvme0;reboot", "nvme0\n"])
def test_invalid_nvme_device_never_executes(tmp_path, host, device):
    functions, remote = functions_for(tmp_path, host, {})
    result = functions.execute("query_nvme_health", {"host": host.name, "device": device}, host.name)
    assert result["error"] == "INVALID_ARGUMENTS"
    assert remote.calls == []


def test_explicit_nvme_uses_exact_argv_and_malformed_json_fails(tmp_path, host):
    class RecordingExecutor:
        def __init__(self):
            self.commands = []

        def execute(self, actual_host, command, timeout, cancellation=None):
            self.commands.append((tuple(command), timeout))
            return ToolResult(success=True, stdout="{}", duration_ms=1)

    database = Database(tmp_path / "record.db")
    database.migrate()
    remote = RecordingExecutor()
    functions = FunctionExecutor(HostRegistry(hosts={host.name: host}),
        InspectionService(database, tmp_path / "logs"), database, remote)
    result = evidence(functions.execute("query_nvme_health", {
        "host": host.name, "device": "nvme1",
    }, host.name))
    assert remote.commands == [(('nvme', 'smart-log', '-o', 'json', '/dev/nvme1'), 15)]
    assert result["status"] == "error"


def test_capability_availability_cross_checks_real_registry():
    available = CapabilityBackendResolver()
    absent = CapabilityBackendResolver(registry={})
    mismatch = CapabilityBackendResolver(registry={"different_name": object()})
    for capability in ("traffic_history", "nvme_health", "network_interface_details"):
        gap = CapabilityGap(capability_id=capability, status="unknown", software_status="available",
            reason="software found", confidence="high")
        assert assess_gap(gap, available).status == "available"
        assert assess_gap(gap, absent).status == "unknown"
    assert not mismatch.available("traffic_history")
    assert "inspect_interface" in HOST_READ_CAPABILITIES
