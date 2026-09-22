import json

from vast_agent.actions.capability_bridge import ground_capability_gap, request_for_gap
from vast_agent.agent.evidence import compact_evidence
from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.investigation_session import InvestigationSession, StopReason
from vast_agent.agent.models import AgentResponse, CapabilityGap
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


def traffic_gap(candidate="curl"):
    return CapabilityGap(
        capability_id="traffic_history", status="missing", software_status="missing",
        reason="model says missing", candidate_package=candidate, confidence="high",
    )


def add_record(session, source, arguments, facts, status="available"):
    record = compact_evidence(source, {"untrusted_evidence": facts}, 1000)
    if status != "available":
        record = record.model_copy(update={"status": status})
    assert session.add(source, arguments, record)


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


def test_last_round_is_reserved_for_synthesis_after_tool_call(tmp_path, host):
    agent, client, _ = make_agent(tmp_path, host, [
        call("query_nvme_health", {"host": host.name}), answer("used final evidence"),
    ], {"query_nvme_health:list": ToolResult(success=True, stdout="[]", duration_ms=1)},
        max_agent_steps=2, max_llm_calls=2)
    result = agent.investigate(host.name, "SSDを確認して")
    assert result.summary == "used final evidence"
    assert result.stop_reason == "ANSWERABLE"
    assert client.calls == 2


def test_exact_repeat_is_deduplicated(tmp_path, host):
    repeated = {"host": host.name, "scope": "routes"}
    agent, _, remote = make_agent(tmp_path, host, [
        call("inspect_network", repeated), call("inspect_network", repeated, "2"),
        answer("used retained network evidence"),
    ])
    result = agent.investigate(host.name, "ネット遅くない？")
    assert result.summary == "used retained network evidence"
    assert result.stop_reason == "ANSWERABLE"
    assert remote.calls.count("inspect_network:routes") == 1


def test_tool_budget_stops_pending_read_and_uses_tool_free_synthesis(tmp_path, host):
    agent, client, remote = make_agent(tmp_path, host, [
        call("inspect_network", {"host": host.name, "scope": "routes"}),
        call("query_nvme_health", {"host": host.name}, "2"),
        answer("partial answer from routes"),
    ], max_tool_calls=1)

    result = agent.investigate(host.name, "ネットとSSDを確認して")

    assert result.summary == "partial answer from routes"
    assert result.stop_reason == "ANSWERABLE"
    assert client.calls == 3
    assert client.requests[-1]["tools"] == []
    assert "No more tools are available" in json.dumps(client.requests[-1]["inputs"])
    assert remote.calls == ["inspect_network:routes"]


def test_final_synthesis_error_preserves_existing_evidence(tmp_path, host):
    agent, client, remote = make_agent(tmp_path, host, [
        call("inspect_network", {"host": host.name, "scope": "routes"}),
        call("inspect_network", {"host": host.name, "scope": "routes"}, "2"),
        RuntimeError("provider unavailable"),
    ])

    result = agent.investigate(host.name, "ネットを確認して")

    assert result.stop_reason == "ERROR"
    assert result.findings
    assert client.requests[-1]["tools"] == []
    assert remote.calls == ["inspect_network:routes"]


def test_function_call_from_tool_free_synthesis_fails_closed(tmp_path, host):
    agent, client, remote = make_agent(tmp_path, host, [
        call("inspect_network", {"host": host.name, "scope": "routes"}),
        call("inspect_network", {"host": host.name, "scope": "routes"}, "2"),
        call("query_nvme_health", {"host": host.name}, "3"),
    ])

    result = agent.investigate(host.name, "ネットを確認して")

    assert result.stop_reason == "BOUND_REACHED"
    assert result.findings
    assert client.requests[-1]["tools"] == []
    assert remote.calls == ["inspect_network:routes"]


def test_evidence_limit_rejection_still_allows_synthesis(tmp_path, host):
    agent, client, _ = make_agent(tmp_path, host, [answer("bounded evidence conclusion")])
    session = InvestigationSession(
        target_host=host.name, goal="bounded", max_total_evidence_chars=220,
    )
    add_record(session, "first", {"host": host.name}, {"state": "known"})
    oversized = compact_evidence("second", {"untrusted_evidence": {"x": "x" * 1000}}, 1000)
    assert not session.add("second", {"host": host.name}, oversized)
    assert len(session.evidence) == 1

    result = agent._synthesize_from_evidence(
        session, [_user_input_for_test("bounded")], StopReason.BOUND_REACHED,
    )

    assert result.summary == "bounded evidence conclusion"
    assert client.requests[-1]["tools"] == []
    assert "first" in json.dumps(client.requests[-1]["inputs"])


def _user_input_for_test(text):
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


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


def test_model_missing_without_prerequisite_reads_cannot_propose():
    session = InvestigationSession(target_host="host", goal="history")
    grounded = ground_capability_gap(traffic_gap(), session)
    assert grounded.software_status == "unknown"
    assert request_for_gap("host", grounded, consent=True) is None


def test_installed_or_failed_package_read_cannot_propose():
    installed = InvestigationSession(target_host="host", goal="history")
    add_record(installed, "query_package", {"host": "host", "package_name": "vnstat"},
               {"package": "vnstat", "installed": True})
    add_record(installed, "query_executable", {"host": "host", "executable_name": "vnstat"},
               {"executable": "vnstat", "exists": True})
    assert request_for_gap("host", ground_capability_gap(traffic_gap(), installed), consent=True) is None

    failed = InvestigationSession(target_host="host", goal="history")
    add_record(failed, "query_package", {"host": "host", "package_name": "vnstat"},
               {"error": "COMMAND_FAILED"}, "evidence_missing")
    add_record(failed, "query_executable", {"host": "host", "executable_name": "vnstat"},
               {"executable": "vnstat", "exists": False})
    grounded = ground_capability_gap(traffic_gap(), failed)
    assert grounded.software_status == "unknown"
    assert request_for_gap("host", grounded, consent=True) is None


def test_package_and_executable_absence_ground_catalog_package_only():
    session = InvestigationSession(target_host="host", goal="history")
    add_record(session, "query_package", {"host": "host", "package_name": "vnstat"},
               {"package": "vnstat", "installed": False})
    add_record(session, "query_executable", {"host": "host", "executable_name": "vnstat"},
               {"executable": "vnstat", "exists": False})
    grounded = ground_capability_gap(traffic_gap(candidate="curl"), session)
    request = request_for_gap("host", grounded, consent=True)
    assert grounded.software_status == "missing"
    assert request is not None
    assert request.parameters.package_name == "vnstat"
    assert "curl" not in request.model_dump_json()
    assert request_for_gap("host", grounded, consent=False) is None


def test_agent_sanitizes_model_gap_from_actual_session_reads(tmp_path, host):
    final = AgentResponse(output_text=json.dumps({
        "summary": "history unavailable", "findings": [], "signatures": [],
        "confidence": "high", "recommended_action": "NONE", "missing_evidence": [],
        "capability_gaps": [traffic_gap().model_dump(mode="json")],
    }))
    agent, _, _ = make_agent(tmp_path, host, [
        call("query_package", {"host": host.name, "package_name": "vnstat"}),
        call("query_executable", {"host": host.name, "executable_name": "vnstat"}, "2"),
        final,
    ], {
        "query_package": ToolResult(success=False, exit_code=1, duration_ms=1),
        "query_executable": ToolResult(success=False, exit_code=1, duration_ms=1),
    })
    result = agent.investigate(host.name, "昨日の通信量。必要なら入れて")
    assert len(result.capability_gaps) == 1
    request = request_for_gap(host.name, result.capability_gaps[0], consent=True)
    assert request is not None
    assert request.parameters.package_name == "vnstat"


def test_inactive_service_is_degraded_not_missing():
    session = InvestigationSession(target_host="host", goal="history")
    add_record(session, "query_package", {"host": "host", "package_name": "vnstat"},
               {"package": "vnstat", "installed": True})
    add_record(session, "query_executable", {"host": "host", "executable_name": "vnstat"},
               {"executable": "vnstat", "exists": True})
    add_record(session, "query_service", {"host": "host", "service_name": "vnstat"},
               {"service": "vnstat.service", "exists": True, "active_state": "inactive"})
    grounded = ground_capability_gap(traffic_gap(), session)
    assert grounded.status == "degraded"
    assert grounded.software_status == "available"
    assert request_for_gap("host", grounded, consent=True) is None


def test_grounding_rejects_cross_host_and_other_capability_evidence():
    session = InvestigationSession(target_host="host", goal="history")
    session.evidence.append(compact_evidence(
        "query_package", {"untrusted_evidence": {"package": "vnstat", "installed": False}}, 1000,
    ).model_copy(update={
        "target_host": "other", "arguments": {"host": "other", "package_name": "vnstat"},
    }))
    add_record(session, "query_package", {"host": "host", "package_name": "nvme-cli"},
               {"package": "nvme-cli", "installed": False})
    add_record(session, "query_executable", {"host": "host", "executable_name": "nvme"},
               {"executable": "nvme", "exists": False})
    grounded = ground_capability_gap(traffic_gap(), session)
    assert grounded.software_status == "unknown"
    assert request_for_gap("host", grounded, consent=True) is None
