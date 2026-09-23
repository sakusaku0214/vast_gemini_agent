import json

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


def test_user_to_help_driven_vast_read_result(tmp_path, host):
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
