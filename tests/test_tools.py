from vast_agent.execution.base import FakeExecutor, redact
from vast_agent.models.tool_result import ToolResult
from vast_agent.tools.registry import TOOLS, run_tool


def test_all_required_read_only_tools_registered():
    expected = {"host_ping", "get_system_health", "get_gpu_status", "get_gpu_processes",
                "get_pci_status", "get_kernel_gpu_errors", "get_d_state_processes",
                "get_vast_status", "get_vast_logs", "get_docker_status", "get_vm_status",
                "get_service_status", "get_journal_errors", "read_config"}
    assert expected == set(TOOLS)
    assert all(tool.risk_class == "read_only" for tool in TOOLS.values())


def test_fake_executor_and_no_external_command(host):
    fake = FakeExecutor({
        "host_ping": ToolResult(success=True, exit_code=0, stdout="ok", duration_ms=1),
    })
    assert run_tool("host_ping", host, fake).stdout == "ok"
    assert run_tool("not-a-tool", host, fake).error_code == "TOOL_UNSUPPORTED"


def test_fake_executor_distinguishes_tools_with_same_command(host):
    fake = FakeExecutor({
        "get_system_health": ToolResult(success=True, stdout="health", duration_ms=1),
        "get_docker_status": ToolResult(success=True, stdout="docker", duration_ms=1),
        "get_vm_status": ToolResult(success=True, stdout="vm", duration_ms=1),
    })
    assert run_tool("get_system_health", host, fake).stdout == "health"
    assert run_tool("get_docker_status", host, fake).stdout == "docker"
    assert run_tool("get_vm_status", host, fake).stdout == "vm"


def test_redaction():
    assert redact("token=hunter2", ["hunter2"]) == "token=[REDACTED]"
