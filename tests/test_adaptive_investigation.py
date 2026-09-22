import json

from vast_agent.agent.evidence import compact_evidence
from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.investigation_session import InvestigationSession, StopReason
from vast_agent.agent.models import AgentResponse
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.config import GeminiSettings, HostRegistry
from vast_agent.execution.base import FakeExecutor
from vast_agent.models.tool_result import ToolResult
from vast_agent.services.inspection import InspectionService
from vast_agent.storage.database import Database


def make_agent(tmp_path, host, responses, results=None, **limits):
    database = Database(tmp_path / "planner.db")
    database.migrate()
    remote = FakeExecutor(results or {})
    functions = FunctionExecutor(
        HostRegistry(hosts={host.name: host}),
        InspectionService(database, tmp_path / "logs"), database, remote,
    )
    client = ScriptedGeminiClient(responses)
    return InvestigationAgent(client, functions, database, GeminiSettings(**limits)), client, remote


def call(name, arguments, call_id="1"):
    return AgentResponse(steps=[{
        "type": "function_call", "name": name, "arguments": arguments, "id": call_id,
    }])


def answer(summary="done"):
    return AgentResponse(output_text=json.dumps({
        "summary": summary, "findings": [], "signatures": [], "confidence": "medium",
        "recommended_action": "NONE", "missing_evidence": [], "capability_gaps": [],
        "stop_reason": "ANSWERABLE",
    }))


def test_two_different_reads_compose_and_stop_when_answerable(tmp_path, host):
    agent, client, remote = make_agent(tmp_path, host, [
        call("query_gpu_diagnostics", {"host": host.name}),
        call("query_docker_diagnostics", {"host": host.name}, "2"),
        answer("combined"),
    ])
    result = agent.investigate(host.name, "GPUとdocker両方変じゃない？")
    assert result.summary == "combined"
    assert result.stop_reason == "ANSWERABLE"
    assert client.calls == 3
    assert remote.calls  # both composite backends gathered allowlisted evidence


def test_answerable_after_first_read_does_not_call_more(tmp_path, host):
    agent, client, _ = make_agent(tmp_path, host, [
        call("query_nvme_health", {"host": host.name}), answer("SSD evidence is sufficient"),
    ], {"query_nvme_health:list": ToolResult(success=True, stdout="[]", duration_ms=1)})
    assert agent.investigate(host.name, "SSD怪しくない？").summary == "SSD evidence is sufficient"
    assert client.calls == 2


def test_exact_repeat_is_deduplicated(tmp_path, host):
    repeated = {"host": host.name, "scope": "routes"}
    agent, _, remote = make_agent(tmp_path, host, [
        call("inspect_network", repeated), call("inspect_network", repeated, "2"),
    ])
    result = agent.investigate(host.name, "ネット遅くない？")
    assert result.stop_reason == "NO_NEW_EVIDENCE"
    assert remote.calls.count("inspect_network:routes") == 1


def test_unknown_and_write_tools_fail_closed_without_execution(tmp_path, host):
    for name in ("unknown_tool", "PACKAGE_INSTALL", "restart_service"):
        agent, _, remote = make_agent(tmp_path, host, [call(name, {"host": host.name})])
        result = agent.investigate(host.name, "調べて")
        assert result.stop_reason == "ERROR"
        assert remote.calls == []


def test_invalid_arguments_and_host_mismatch_are_compact_missing_evidence(tmp_path, host):
    for arguments in ({"host": host.name, "scope": "invalid"}, {"host": "other", "scope": "routes"}):
        agent, client, remote = make_agent(tmp_path, host, [
            call("inspect_network", arguments), answer(),
        ])
        agent.investigate(host.name, "ネットを調べて")
        sent = json.dumps(client.requests[1], ensure_ascii=False)
        assert "evidence_missing" in sent
        assert remote.calls == []


def test_tool_failure_is_not_capability_gap(tmp_path, host):
    record = compact_evidence("query_nvme_health", {"error": "COMMAND_FAILED"}, 200)
    assert record.status == "evidence_missing"
    assert "capability" not in record.status


def test_session_bounds_cache_and_evidence_compaction():
    session = InvestigationSession(
        target_host="host", goal="goal", max_tool_calls=1, max_rounds=1,
        max_total_evidence_chars=220,
    )
    args = {"host": "host"}
    record = compact_evidence("read", {"untrusted_evidence": {"text": "x" * 1000}}, 100)
    assert session.add("read", args, record)
    assert session.has_call("read", args)
    assert not session.can_call()
    assert session.evidence_chars <= session.max_total_evidence_chars
    assert not session.add("read", args, record)
    assert session.stop_reason == StopReason.NO_NEW_EVIDENCE


def test_prompt_declares_no_mutation_tools(tmp_path, host):
    agent, client, _ = make_agent(tmp_path, host, [answer()])
    agent.investigate(host.name, "不安定じゃない？")
    names = {item["name"] for item in client.requests[0]["tools"]}
    assert not names & {"PACKAGE_INSTALL", "restart_service", "reboot", "gpu_reset"}
    assert all("command" not in item["parameters"].get("properties", {}) for item in client.requests[0]["tools"])
