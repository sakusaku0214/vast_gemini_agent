import json

from vast_agent.actions.package_catalog import capability_definition
from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.models import AgentResponse
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.config import GeminiSettings, HostRegistry
from vast_agent.models.tool_result import ToolResult
from vast_agent.services.inspection import InspectionService
from vast_agent.storage.database import Database


def call(name, arguments, call_id):
    return AgentResponse(steps=[{
        "type": "function_call", "name": name, "arguments": arguments, "id": call_id,
    }])


def answer():
    return AgentResponse(output_text=json.dumps({
        "summary": "Vast machine is rented", "findings": ["machine 42 rented=true"],
        "signatures": [], "confidence": "high", "recommended_action": "NONE",
        "missing_evidence": [], "capability_gaps": [], "stop_reason": "ANSWERABLE",
    }))


class CliRemote:
    def __init__(self):
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        command = tuple(command)
        self.calls.append(command)
        outputs = {
            ("which", "--", "vastai"): "/usr/bin/vastai\n",
            ("vastai", "--help"): "show  Display resources\n",
            ("vastai", "show", "machines"): "id=42 rented=true\n",
        }
        if command not in outputs:
            return ToolResult(success=False, exit_code=127, duration_ms=1)
        return ToolResult(success=True, exit_code=0, stdout=outputs[command], duration_ms=1)


class UserPathCliRemote(CliRemote):
    def execute(self, host, command, timeout, cancellation=None):
        command = tuple(command)
        self.calls.append(command)
        path = f"/home/{host.ssh_user}/.local/bin/vastai"
        if command == ("which", "--", "vastai"):
            return ToolResult(success=False, exit_code=1, duration_ms=1)
        if command == ("test", "-x", path):
            return ToolResult(success=True, exit_code=0, duration_ms=1)
        if command in {(path, "--help"), (path, "show", "machines")}:
            output = "show Display resources\n" if command[-1] == "--help" else "id=42 rented=true\n"
            return ToolResult(success=True, exit_code=0, stdout=output, duration_ms=1)
        return ToolResult(success=False, exit_code=1, duration_ms=1)


def test_failed_which_continues_bounded_user_path_discovery(tmp_path, host):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    remote = UserPathCliRemote()
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}), InspectionService(db, tmp_path / "logs"), db, remote,
    )
    path = f"/home/{host.ssh_user}/.local/bin/vastai"
    client = ScriptedGeminiClient([
        call("query_executable", {"host": host.name, "executable_name": "vastai"}, "1"),
        call("query_cli_help", {"host": host.name, "executable": path, "argv": ["--help"],
                                "reason": "discover syntax"}, "2"),
        call("run_readonly_argv", {"host": host.name, "executable": path,
                                    "argv": ["show", "machines"], "reason": "read rental"}, "3"),
        answer(),
    ])
    result = InvestigationAgent(
        client, functions, db, GeminiSettings(max_llm_calls=4, max_agent_steps=4),
    ).investigate(host.name, "X570にvast cli入ってるからレント状況見て。分からなかったらhelp見て")
    assert result.summary == "Vast machine is rented"
    assert ("test", "-x", path) in remote.calls
    assert (path, "show", "machines") in remote.calls


def test_discovered_path_immediately_continues_requested_read_without_reasoning_round(tmp_path, host):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    remote = UserPathCliRemote()
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}), InspectionService(db, tmp_path / "logs"), db, remote,
    )
    path = f"/home/{host.ssh_user}/.local/bin/vastai"
    client = ScriptedGeminiClient([
        call("query_executable", {
            "host": host.name, "executable_name": "vastai",
            "continuation_argv": ["show", "machines"],
        }, "1"),
        answer(),
    ])

    result = InvestigationAgent(
        client, functions, db, GeminiSettings(max_llm_calls=2, max_agent_steps=2),
    ).investigate(host.name, "X570にvast cli入ってるからレント状況見て")

    assert result.summary == "Vast machine is rented"
    assert (path, "show", "machines") in remote.calls
    assert client.calls == 2


def test_user_to_help_driven_vast_read_result(tmp_path, host):
    assert capability_definition("vast_cli") is None
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    remote = CliRemote()
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}), InspectionService(db, tmp_path / "logs"), db, remote,
    )
    client = ScriptedGeminiClient([
        call("query_executable", {"host": host.name, "executable_name": "vastai"}, "1"),
        call("query_cli_help", {
            "host": host.name, "executable": "vastai", "argv": ["--help"],
            "reason": "discover Vast CLI syntax",
        }, "2"),
        call("run_readonly_argv", {
            "host": host.name, "executable": "vastai", "argv": ["show", "machines"],
            "reason": "read rental state",
        }, "3"),
        answer(),
    ])
    agent = InvestigationAgent(client, functions, db, GeminiSettings(max_llm_calls=4, max_agent_steps=4))

    result = agent.investigate(host.name, "vast cliでレント状況を確認。分からなければhelp見て")

    assert result.summary == "Vast machine is rented"
    assert remote.calls == [
        ("which", "--", "vastai"), ("vastai", "--help"),
        ("vastai", "show", "machines"),
    ]
    assert agent.last_session.trace()["selected_tools"] == [
        "query_executable", "query_cli_help", "run_readonly_argv",
    ]


class AdaptiveRemote:
    def __init__(self):
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        command = tuple(command)
        self.calls.append(command)
        if command == ("novel-observer", "snapshot", "--json"):
            return ToolResult(
                success=False, exit_code=2, stderr="unknown option --json\nUsage: snapshot",
                duration_ms=1,
            )
        if command == ("novel-observer", "snapshot", "--help"):
            return ToolResult(success=True, exit_code=0, stdout="Usage: snapshot [--compact]\n", duration_ms=1)
        if command == ("novel-observer", "snapshot", "--compact"):
            return ToolResult(success=True, exit_code=0, stdout="healthy nodes=3\n", duration_ms=1)
        return ToolResult(success=False, exit_code=127, stderr="command not found", duration_ms=1)


def test_unregistered_cli_executes_first_then_adapts_from_failure(tmp_path, host):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    remote = AdaptiveRemote()
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}), InspectionService(db, tmp_path / "logs"), db, remote,
    )
    def read(argv, call_id):
        return call("run_readonly_argv", {
            "host": host.name, "executable": "novel-observer", "argv": argv,
            "reason": "inspect the requested novel subsystem",
        }, call_id)
    client = ScriptedGeminiClient([
        read(["snapshot", "--json"], "1"),
        call("query_cli_help", {
            "host": host.name, "executable": "novel-observer",
            "argv": ["snapshot", "--help"], "reason": "syntax error requires bounded help",
        }, "2"),
        read(["snapshot", "--compact"], "3"),
        answer(),
    ])
    agent = InvestigationAgent(
        client, functions, db, GeminiSettings(max_llm_calls=4, max_agent_steps=4),
    )

    result = agent.investigate(host.name, "新しいobserverでnode状態を調べて")

    assert result.summary == "Vast machine is rented"
    assert remote.calls == [
        ("novel-observer", "snapshot", "--json"),
        ("novel-observer", "snapshot", "--help"),
        ("novel-observer", "snapshot", "--compact"),
    ]
    assert "query_executable" not in agent.last_session.trace()["selected_tools"]


class PermissionRetryRemote:
    def __init__(self):
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        command = tuple(command)
        self.calls.append(command)
        if command == ("private-observer", "status"):
            return ToolResult(success=False, exit_code=126, stderr="Permission denied", duration_ms=1)
        if command == ("sudo", "-n", "true"):
            return ToolResult(success=True, exit_code=0, duration_ms=1)
        if command == ("sudo", "-n", "private-observer", "status"):
            return ToolResult(success=True, exit_code=0, stdout="healthy\n", duration_ms=1)
        return ToolResult(success=False, exit_code=1, duration_ms=1)


def test_permission_failure_can_adapt_to_same_typed_sudo_read(tmp_path, host):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    remote = PermissionRetryRemote()
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}), InspectionService(db, tmp_path / "logs"), db, remote,
    )
    arguments = {
        "host": host.name, "executable": "private-observer", "argv": ["status"],
        "reason": "inspect private subsystem status",
    }
    client = ScriptedGeminiClient([
        call("run_readonly_argv", arguments, "1"),
        call("run_readonly_argv", {**arguments, "requires_sudo": True}, "2"),
        answer(),
    ])
    result = InvestigationAgent(
        client, functions, db, GeminiSettings(max_llm_calls=3, max_agent_steps=3),
    ).investigate(host.name, "private subsystemの状態を調べて")

    assert result.summary == "Vast machine is rented"
    assert remote.calls == [
        ("private-observer", "status"),
        ("sudo", "-n", "true"),
        ("sudo", "-n", "private-observer", "status"),
    ]
