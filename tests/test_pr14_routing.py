from __future__ import annotations

import asyncio
import json

from vast_agent.agent.functions import FUNCTION_DECLARATIONS
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.models import AgentResponse, InvestigationResult, Route
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.agent.router import route_intent
from vast_agent.config import GeminiSettings, HostRegistry
from vast_agent.conversation.state import ConversationStore
from vast_agent.jobs.manager import JobManager
from vast_agent.models.host import Host
from vast_agent.models.observation import Observation, SystemSummary
from vast_agent.models.tool_result import ToolResult
from vast_agent.services.agent_service import AgentService
from vast_agent.storage.database import Database


class StubFunctions:
    def execute(self, *args):
        raise AssertionError("general conversation must not expose host functions")


class RecordingAgent:
    def __init__(self):
        self.investigations = []
        self.general = []

    def investigate(self, host, question):
        self.investigations.append((host, question))
        return InvestigationResult(summary="現在のREAD toolでは直接確認できません")

    def answer_general(self, question):
        self.general.append(question)
        return "一般回答"


def make_service(tmp_path, agent=None, inspection=None):
    db = Database(tmp_path / "db.sqlite")
    host = Host(name="garage-mag", aliases=["mag"], address="192.0.2.1", ssh_user="agent")
    registry = HostRegistry(hosts={host.name: host})
    return AgentService(
        registry, inspection or object(), object(), JobManager(db), ConversationStore(db), agent=agent,
    )


def test_unknown_host_read_and_general_questions_take_separate_fallbacks(tmp_path):
    agent = RecordingAgent()
    service = make_service(tmp_path, agent)

    host_reply = asyncio.run(service.handle_question("magにvnstat入ってる？", 1, 2))
    general_reply = asyncio.run(service.handle_question("自然言語理解できる？", 1, 2))

    assert host_reply.job_id is not None
    assert agent.investigations == [("garage-mag", "magにvnstat入ってる？")]
    assert general_reply.text == "一般回答"
    assert agent.general == ["自然言語理解できる？"]


def test_general_agent_has_no_tools_records_usage_and_is_bounded(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    client = ScriptedGeminiClient([AgentResponse(output_text="x" * 1300, usage={"total_tokens": 3})])
    agent = InvestigationAgent(client, StubFunctions(), db, GeminiSettings())

    assert len(agent.answer_general("VFIOって何？")) == 1200
    assert client.requests[0]["tools"] == []
    assert client.requests[0]["store"] is False
    with db.connect() as connection:
        assert connection.execute("SELECT purpose,total_tokens FROM token_usage").fetchone() == ("general", 3)


def test_read_followup_keeps_host_across_general_turn(tmp_path):
    agent = RecordingAgent()
    service = make_service(tmp_path, agent)

    asyncio.run(service.handle_question("magにvnstat入ってる？", 1, 2))
    asyncio.run(service.handle_question("VFIOって何？", 1, 2))
    followup = asyncio.run(service.handle_question("状態は？", 1, 2))

    assert followup.job_id is not None
    assert agent.investigations[-1] == ("garage-mag", "状態は？")
    assert service.conversations.get(1, 2).last_host == "garage-mag"


def test_bare_write_never_inherits_read_host(tmp_path):
    agent = RecordingAgent()
    service = make_service(tmp_path, agent)
    asyncio.run(service.handle_question("magにvnstat入ってる？", 1, 2))

    reply = asyncio.run(service.handle_question("再起動して", 1, 2))

    assert reply.job_id is None
    assert "WRITE" in reply.text
    assert len(agent.investigations) == 1


def test_reboot_advice_is_read_only_investigation(tmp_path):
    agent = RecordingAgent()
    service = make_service(tmp_path, agent)

    reply = asyncio.run(service.handle_question("magは再起動した方がいい？", 1, 2))

    assert reply.proposal is None
    assert reply.job_id is not None
    assert agent.investigations == [("garage-mag", "magは再起動した方がいい？")]


def test_realtime_questions_reach_general_agent(tmp_path):
    agent = RecordingAgent()
    service = make_service(tmp_path, agent)

    weather = asyncio.run(service.handle_question("今日の天気は？", 1, 2))
    fx = asyncio.run(service.handle_question("ドル円いくら？", 1, 2))

    assert weather.text == "一般回答"
    assert fx.text == "一般回答"
    assert agent.general == ["今日の天気は？", "ドル円いくら？"]


def test_system_formatter_does_not_render_gpu_template():
    observation = Observation(
        host="garage-mag", ssh_ok=True,
        system=SystemSummary(filesystem_max_percent=71, failed_units=1, d_state_processes=2),
        services={"vastai": "active"}, signatures=["DISK_PRESSURE"],
    )

    rendered = AgentService._format_observation(observation, "system")

    assert "Filesystem max usage: 71%" in rendered
    assert "Failed units: 1" in rendered
    assert "GPU:" not in rendered


def test_service_formatters_use_canonical_observation_keys():
    observation = Observation(
        host="garage-mag", ssh_ok=True, services={"vastai": "active", "docker": "active"},
    )

    assert "vastai: active" in AgentService._format_observation(observation, "vast")
    assert "unknown" not in AgentService._format_observation(observation, "vast")
    assert "docker: active" in AgentService._format_observation(observation, "docker")


def test_docker_and_vm_tool_output_produces_bounded_summaries():
    from vast_agent.diagnostics.parser import build_observation

    def ok(stdout=""):
        return ToolResult(success=True, stdout=stdout, duration_ms=1)
    observation = build_observation("garage-mag", {
        "host_ping": ok(),
        "get_docker_status": ok(
            "active\nabc|worker|Up 2 hours|image:latest\ndef|old|Exited (0) 1 day ago|image:old\n",
        ),
        "get_vm_status": ok(
            " Id   Name       State\n---------------------------\n"
            " 1    compute    running\n -    archive    shut off\n"
            "1234 /usr/bin/qemu-system-x86_64 -name guest=compute\n",
        ),
    })

    assert "docker_status" in observation.details
    assert "vm_status" in observation.details
    docker = AgentService._format_observation(observation, "docker")
    vm = AgentService._format_observation(observation, "vm")
    assert "total=2, running=1, stopped=1" in docker
    assert "not available" not in docker
    assert "VMs: total=2, running=1" in vm
    assert "compute=running" in vm and "archive=shut off" in vm
    assert "not available" not in vm


def test_restarting_docker_container_counts_as_running():
    from vast_agent.diagnostics.parser import summarize_docker_status

    summary = summarize_docker_status(
        "active\nabc|worker|Restarting (1) 5 seconds ago|image:latest\n",
    )

    assert summary == {"total": 1, "running": 1, "stopped": 0}


def test_routing_order_and_gemini_function_boundary(tmp_path):
    service = make_service(tmp_path)
    registry = service.registry

    assert route_intent("magのGPU温度", registry).route == Route.DETERMINISTIC
    disk = route_intent("magのディスク状況確認できる？", registry)
    assert disk.route == Route.DETERMINISTIC and disk.scope == "system"
    assert route_intent("magにvnstat入ってる？", registry).route == Route.AGENT
    assert route_intent("自然言語理解できる？", registry).route == Route.UNKNOWN
    assert route_intent("magを再起動して", registry).route == Route.UNSUPPORTED_WRITE
    assert {item["name"] for item in FUNCTION_DECLARATIONS} == {
        "inspect_host", "collect_evidence", "get_recent_incidents",
    }
    assert "restart" not in json.dumps(FUNCTION_DECLARATIONS).casefold()
