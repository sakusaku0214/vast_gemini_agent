import json

import pytest

from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.host_read import FUNCTION_DECLARATIONS, HOST_READ_CAPABILITIES
from vast_agent.agent.models import AgentResponse
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.config import GeminiSettings, HostRegistry
from vast_agent.execution.base import FakeExecutor
from vast_agent.models.tool_result import ToolResult
from vast_agent.services.inspection import InspectionService
from vast_agent.storage.database import Database


def make_functions(tmp_path, host, results=None, max_chars=2500):
    database = Database(tmp_path / "read.db")
    database.migrate()
    remote = FakeExecutor(results or {})
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}),
        InspectionService(database, tmp_path / "logs"), database, remote,
        max_chars=max_chars, secrets=("secret-token",),
    )
    return functions, remote, database


def evidence(result):
    return result["untrusted_evidence"]


def test_registry_generates_all_declarations():
    assert {item["name"] for item in FUNCTION_DECLARATIONS} == set(HOST_READ_CAPABILITIES)
    assert all(item.risk_class == "READ_ONLY" for item in HOST_READ_CAPABILITIES.values())


@pytest.mark.parametrize(("function", "field", "value"), [
    ("query_package", "package_name", "--help"),
    ("query_package", "package_name", "vnstat; reboot"),
    ("query_executable", "executable_name", "../bin/sh"),
    ("query_executable", "executable_name", "x\nreboot"),
    ("query_service", "service_name", "; reboot"),
    ("query_service", "service_name", "--help"),
    ("query_service", "service_name", "../docker"),
    ("inspect_interface", "interface_name", "../../etc/passwd"),
    ("inspect_interface", "interface_name", "eth0\nreboot"),
])
def test_invalid_dynamic_values_never_execute(tmp_path, host, function, field, value):
    functions, remote, _ = make_functions(tmp_path, host)
    result = functions.execute(function, {"host": host.name, field: value}, host.name)
    assert result["error"] == "INVALID_ARGUMENTS"
    assert remote.calls == []


def test_package_present_and_absent_are_structured_and_never_install(tmp_path, host):
    functions, remote, _ = make_functions(tmp_path, host, {
        "query_package": ToolResult(success=True, exit_code=0, stdout="installed\t2.10-2\n", duration_ms=1),
    })
    found = evidence(functions.execute("query_package", {"host": host.name, "package_name": "vnstat"}, host.name))
    assert found == {"package": "vnstat", "installed": True, "status": "installed", "version": "2.10-2"}
    remote.results["query_package"] = ToolResult(success=False, exit_code=1, duration_ms=1)
    missing = evidence(functions.execute("query_package", {"host": host.name, "package_name": "vnstat"}, host.name))
    assert missing["installed"] is False
    assert remote.calls == ["query_package", "query_package"]


def test_service_process_and_os_results_are_bounded_structures(tmp_path, host):
    functions, remote, _ = make_functions(tmp_path, host, {
        "query_service": ToolResult(success=True, stdout="LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=42\n", duration_ms=1),
        "query_process": ToolResult(success=True, stdout="10\n11\n", duration_ms=1),
        "inspect_os:release": ToolResult(success=True, stdout='PRETTY_NAME="Ubuntu 24.04"\nVERSION_ID="24.04"\nSECRET=secret-token\n', duration_ms=1),
        "inspect_os:kernel": ToolResult(success=True, stdout="Linux 6.8 x86_64 GNU/Linux\n", duration_ms=1),
        "inspect_os:uptime": ToolResult(success=True, stdout="up 2 days\n", duration_ms=1),
    })
    service = evidence(functions.execute("query_service", {"host": host.name, "service_name": "vnstat"}, host.name))
    process = evidence(functions.execute("query_process", {"host": host.name, "process_name": "vnstatd"}, host.name))
    os_facts = evidence(functions.execute("inspect_os", {"host": host.name}, host.name))
    assert service["active_state"] == "active" and service["main_pid"] == 42
    assert process == {"process": "vnstatd", "running": True, "count": 2, "pids": [10, 11]}
    assert "SECRET" not in json.dumps(os_facts)
    assert "secret-token" not in json.dumps(os_facts)
    assert remote.calls == ["query_service", "query_process", "inspect_os:release", "inspect_os:kernel", "inspect_os:uptime"]


def test_network_can_lead_to_interface_followup_without_mac_leak(tmp_path, host):
    link = json.dumps([{"ifname": "eth0", "operstate": "DOWN", "mtu": 1500, "address": "aa:bb:cc:dd:ee:ff"}])
    functions, remote, database = make_functions(tmp_path, host, {
        "inspect_network:interfaces": ToolResult(success=True, stdout=link, duration_ms=1),
        "inspect_network:addresses": ToolResult(success=True, stdout="[]", duration_ms=1),
        "inspect_network:routes": ToolResult(success=True, stdout="[]", duration_ms=1),
        "inspect_interface:link": ToolResult(success=True, stdout=link, duration_ms=1),
        "inspect_interface:details": ToolResult(success=False, exit_code=127, duration_ms=1),
    })
    responses = [
        AgentResponse(function_calls=[], steps=[{"type": "function_call", "name": "inspect_network", "arguments": {"host": host.name, "scope": "summary"}, "id": "1"}]),
        AgentResponse(function_calls=[], steps=[{"type": "function_call", "name": "inspect_interface", "arguments": {"host": host.name, "interface_name": "eth0"}, "id": "2"}]),
        AgentResponse(output_text=json.dumps({"summary": "eth0 is down", "findings": [], "signatures": [], "confidence": "high", "recommended_action": "NONE", "missing_evidence": []})),
    ]
    # Scripted client derives calls from function_call steps.
    client = ScriptedGeminiClient(responses)
    agent = InvestigationAgent(client, functions, database, GeminiSettings())
    result = agent.investigate(host.name, "ネットワークを調べて")
    assert result.summary == "eth0 is down"
    assert remote.calls == ["inspect_network:interfaces", "inspect_network:addresses", "inspect_network:routes", "inspect_interface:link", "inspect_interface:details"]
    assert "aa:bb:cc:dd:ee:ff" not in json.dumps(client.requests, default=str)


def test_unavailable_command_requires_bounded_path_discovery_before_absence(tmp_path, host):
    functions, remote, _ = make_functions(tmp_path, host, {
        "query_executable": ToolResult(success=False, exit_code=127, stderr="missing", duration_ms=1),
    })
    result = evidence(functions.execute("query_executable", {"host": host.name, "executable_name": "vnstat"}, host.name))
    assert result == {"executable": "vnstat", "exists": False, "path": None}
    assert remote.calls == ["query_executable", *["query_executable:candidate"] * 5]


def test_capability_discovery_comes_from_registry_without_remote_execution(tmp_path, host):
    functions, remote, _ = make_functions(tmp_path, host)
    result = evidence(functions.execute("list_host_capabilities", {"host": host.name}, host.name))
    assert {item["name"] for item in result["capabilities"]} == set(HOST_READ_CAPABILITIES)
    assert remote.calls == []


def test_dynamic_value_is_one_argv_element_with_no_shell(tmp_path, host):
    class RecordingExecutor:
        def __init__(self):
            self.commands = []

        def execute(self, actual_host, command, timeout, cancellation=None):
            self.commands.append((actual_host.name, tuple(command), timeout, cancellation))
            return ToolResult(success=True, exit_code=0, stdout="installed\t2.10\n", duration_ms=1)

    database = Database(tmp_path / "argv.db")
    database.migrate()
    remote = RecordingExecutor()
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}),
        InspectionService(database, tmp_path / "logs"), database, remote,
    )
    functions.execute("query_package", {"host": host.name, "package_name": "vnstat"}, host.name)
    assert remote.commands[0][1] == (
        "dpkg-query", "-W", "-f=${db:Status-Status}\\t${Version}\\n", "--", "vnstat",
    )
    assert remote.commands[0][2] == 15
    assert remote.commands[0][1][0] not in {"sh", "bash"}
