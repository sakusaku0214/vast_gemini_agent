import json

import pytest

from vast_agent.actions.capability_bridge import ground_capability_gap, request_for_gap
from vast_agent.agent.evidence import compact_evidence
from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.investigation_session import InvestigationSession, StopReason
from vast_agent.agent.models import AgentResponse, CapabilityGap, InvestigationResult
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.agent.prompts import SYSTEM_PROMPT
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
    assert agent.last_session.trace()["selected_calls"] == [
        {"tool": "query_gpu_diagnostics", "arguments": {}},
        {"tool": "query_docker_diagnostics", "arguments": {}},
    ]
    assert len(agent.last_session.tool_calls) == 2
    assert "collect_evidence" not in agent.last_session.trace()["selected_tools"]
    assert "get_recent_incidents" not in agent.last_session.trace()["selected_tools"]


def test_typed_trace_arguments_omit_host_and_untrusted_evidence(tmp_path, host):
    agent, _, _ = make_agent(tmp_path, host, [
        call("collect_evidence", {
            "host": host.name, "evidence_type": "kernel_gpu_errors",
        }),
        answer(),
    ])

    agent.investigate(host.name, "GPU診断の不足証拠を確認")

    trace = agent.last_session.trace()
    assert trace["selected_calls"] == [{
        "tool": "collect_evidence", "arguments": {"evidence_type": "kernel_gpu_errors"},
    }]
    assert host.name not in json.dumps(trace)
    assert "untrusted_evidence" not in json.dumps(trace)


def test_degraded_specialized_read_can_have_relevant_follow_up(tmp_path, host):
    agent, _, _ = make_agent(tmp_path, host, [
        call("query_gpu_diagnostics", {"host": host.name}),
        call("collect_evidence", {
            "host": host.name, "evidence_type": "kernel_gpu_errors",
        }, "2"),
        answer("follow-up used"),
    ])

    result = agent.investigate(host.name, "GPUとDockerの状態")

    assert result.summary == "follow-up used"
    assert agent.last_session.trace()["selected_tools"] == [
        "query_gpu_diagnostics", "collect_evidence",
    ]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("recommended_action", "RESTART_NOW"), ("confidence", "certain")],
)
def test_final_synthesis_logs_safe_literal_validation_details(
    tmp_path, host, caplog, field, invalid,
):
    payload = json.loads(answer().output_text)
    payload[field] = invalid
    raw_marker = "RAW_SECRET_MARKER"
    payload["summary"] = raw_marker
    agent, client, _ = make_agent(tmp_path, host, [
        AgentResponse(output_text=json.dumps(payload)),
    ])
    session = InvestigationSession(target_host=host.name, goal="safe diagnostics")

    with caplog.at_level("WARNING"):
        result = agent._synthesize_from_evidence(
            session, [_user_input_for_test("safe")], StopReason.BOUND_REACHED,
        )

    assert result.stop_reason == "BOUND_REACHED"
    assert field in caplog.text
    assert "literal_error" in caplog.text
    assert raw_marker not in caplog.text
    assert client.requests[-1]["tools"] == []


@pytest.mark.parametrize("raw", ["{not valid JSON SECRET_BODY", "```json\n{}\n``` SECRET_BODY"])
def test_final_synthesis_malformed_output_logs_only_shape(tmp_path, host, caplog, raw):
    agent, _, _ = make_agent(tmp_path, host, [AgentResponse(output_text=raw)])
    session = InvestigationSession(target_host=host.name, goal="safe diagnostics")

    with caplog.at_level("WARNING"):
        result = agent._synthesize_from_evidence(
            session, [_user_input_for_test("safe")], StopReason.BOUND_REACHED,
        )

    assert result.stop_reason == "BOUND_REACHED"
    assert "output_chars" in caplog.text
    assert "starts_with_json_object" in caplog.text
    assert "contains_code_fence" in caplog.text
    assert "SECRET_BODY" not in caplog.text


def test_planning_loop_logs_safe_structured_validation_details(tmp_path, host, caplog):
    payload = json.loads(answer().output_text)
    payload["confidence"] = "certain"
    payload["summary"] = "PLANNING_RAW_SECRET"
    agent, _, _ = make_agent(tmp_path, host, [
        AgentResponse(output_text=json.dumps(payload)),
    ])

    with caplog.at_level("WARNING"):
        result = agent.investigate(host.name, "現在の状態")

    assert result.stop_reason == "ERROR"
    assert "confidence" in caplog.text
    assert "literal_error" in caplog.text
    assert "PLANNING_RAW_SECRET" not in caplog.text


def test_prompt_distinguishes_current_health_from_historical_relevance():
    assert "by default for a current-state check" in SYSTEM_PROMPT
    assert "explicitly asks about recent/history/instability" in SYSTEM_PROMPT
    assert "specific evidence_type resolves an explicit uncertainty" in SYSTEM_PROMPT


def test_long_gpu_clock_output_keeps_late_locked_clock_sections():
    unrelated = "\n".join(f"Unrelated field {index}: value" for index in range(200))
    output = {"untrusted_evidence": {
        "status": "completed", "executable": "nvidia-smi", "argv": ["-q", "-d", "CLOCK"],
        "stdout": (
            f"GPU 00000000:01:00.0\n{unrelated}\n"
            "    GPU Locked Clocks\n        Min : 1200 MHz\n        Max : 1800 MHz\n"
            "    Max Clocks\n        Graphics : 2100 MHz\n"
        ),
    }}

    record = compact_evidence("run_readonly_argv", output, 1800)

    assert "GPU Locked Clocks" in record.summary
    assert "Min : 1200 MHz" in record.relevant_excerpt
    assert "Max Clocks" in record.relevant_excerpt
    assert len(record.summary) <= 1800


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
    agent, client, remote = make_agent(tmp_path, host, [
        call("inspect_network", repeated), call("inspect_network", repeated, "2"),
        answer("used retained network evidence"),
    ])
    result = agent.investigate(host.name, "ネット遅くない？")
    assert result.summary == "used retained network evidence"
    assert result.stop_reason == "ANSWERABLE"
    assert remote.calls.count("inspect_network:routes") == 1
    duplicate = _function_result_payload(client.requests[-1]["inputs"], "2")
    assert duplicate == {"read_executed": False, "reason": "DUPLICATE_READ"}


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
    blocked = _function_result_payload(client.requests[-1]["inputs"], "2")
    assert blocked["read_executed"] is False


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


def test_evidence_bound_truthfully_replays_executed_and_pending_reads(
    tmp_path, host, monkeypatch,
):
    first = {"type": "function_call", "name": "inspect_network",
             "arguments": {"host": host.name, "scope": "routes"}, "id": "executed"}
    second = {"type": "function_call", "name": "query_nvme_health",
              "arguments": {"host": host.name}, "id": "pending"}
    agent, client, remote = make_agent(tmp_path, host, [
        AgentResponse(steps=[first, second]), answer("synthesized after evidence bound"),
    ])
    original_add = InvestigationSession.add

    def reject_for_evidence_bound(session, tool_name, arguments, evidence, **kwargs):
        session.max_total_evidence_chars = 0
        return original_add(session, tool_name, arguments, evidence, **kwargs)

    monkeypatch.setattr(InvestigationSession, "add", reject_for_evidence_bound)

    result = agent.investigate(host.name, "ネットとSSDを確認して")

    assert result.summary == "synthesized after evidence bound"
    assert result.stop_reason == "ANSWERABLE"
    assert remote.calls == ["inspect_network:routes"]
    assert client.requests[-1]["tools"] == []
    executed = _function_result_payload(client.requests[-1]["inputs"], "executed")
    pending = _function_result_payload(client.requests[-1]["inputs"], "pending")
    assert executed == {
        "read_executed": True,
        "result_retained": False,
        "reason": "EVIDENCE_BOUND_REACHED",
    }
    assert pending == {"read_executed": False, "reason": "EVIDENCE_BOUND_REACHED"}


def _user_input_for_test(text):
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def _function_result_payload(inputs, call_id):
    step = next(item for item in inputs
                if item.get("type") == "function_result" and item.get("call_id") == call_id)
    return json.loads(step["result"][0]["text"])


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


def test_failed_read_is_missing_evidence_with_adaptive_failure_kind():
    record = compact_evidence("run_readonly_argv", {"untrusted_evidence": {
        "status": "failed", "failure_kind": "permission_denied",
        "suggested_next_read": "retry with requires_sudo=true",
    }}, 500)
    assert record.status == "evidence_missing"
    assert record.missing_evidence == ["permission_denied"]
    assert "requires_sudo=true" in record.summary


def test_cli_evidence_strips_ansi_and_deduplicates_display(tmp_path, host):
    output = {"untrusted_evidence": {
        "status": "completed", "executable": "/usr/bin/tool", "argv": ["show"],
        "stdout": "\x1b[40m\x1b[97mid  state\x1b[0m\n42  rented\n",
    }}
    record = compact_evidence("run_readonly_argv", output, 1800)
    assert record.relevant_excerpt == "id  state\n42  rented"
    assert "\x1b" not in record.summary

    session = InvestigationSession(target_host=host.name, goal="show")
    assert session.add("run_readonly_argv", {"argv": ["show"]}, record)
    assert session.add("run_readonly_argv", {"argv": ["list"]}, record)
    result = InvestigationResult(summary="done")
    InvestigationAgent._attach_validated_grounding(result, session)
    assert result._display_evidence.count("42  rented") == 1


def test_display_evidence_selects_primary_cli_not_supporting_or_alternate_reads(host):
    session = InvestigationSession(
        target_host=host.name, goal="widgetctl list nodes の結果を共有して",
    )
    reads = (
        ("query_executable", [], "/opt/tools/widgetctl", "executable discovery"),
        ("query_cli_help", ["--help"], "HELP giant usage", "syntax discovery"),
        ("run_readonly_argv", ["list", "nodes", "--json"],
         '[{"node": "alpha", "state": "ready"}]', "parser fallback"),
        ("run_readonly_argv", ["list", "nodes"], "NODE   STATE\nalpha  ready", "requested data"),
    )
    for source, argv, stdout, reason in reads:
        executable = "widgetctl"
        payload = {
            "status": "completed", "executable": executable, "argv": argv, "stdout": stdout,
        }
        if source == "query_executable":
            payload = {"status": "available", "path": stdout}
        output = {"untrusted_evidence": payload}
        record = compact_evidence(source, output, 1800)
        if source == "query_executable":
            record = record.model_copy(update={"relevant_excerpt": stdout})
        assert session.add(source, {
            "executable": executable, "argv": argv, "reason": reason,
        }, record)

    result = InvestigationResult(summary="1 machine is rented")
    InvestigationAgent._attach_validated_grounding(result, session)

    assert result._display_evidence == "NODE   STATE\nalpha  ready"
    assert "HELP" not in result._display_evidence
    assert '"node"' not in result._display_evidence
    assert "/opt/tools" not in result._display_evidence


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
