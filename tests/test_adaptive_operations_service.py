from __future__ import annotations

import asyncio

import pytest

from vast_agent.agent.models import InvestigationResult
from vast_agent.config import HostRegistry
from vast_agent.conversation.state import ConversationStore
from vast_agent.jobs.manager import JobManager
from vast_agent.models.host import Host
from vast_agent.services.agent_service import AgentService
from vast_agent.storage.database import Database


class ContextAgent:
    def __init__(self):
        self.calls = []

    def investigate(self, host, goal):
        self.calls.append((host, goal))
        return InvestigationResult(summary=f"{host}: evidence collected", confidence="high")

    def answer_general(self, goal):
        return "general"


class ContextAwareAgent(ContextAgent):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def investigate(self, host, goal, context=None):
        self.calls.append((host, goal))
        self.contexts.append(context or {})
        if "最終ブート" in goal:
            return InvestigationResult(
                summary="last -x とjournalから前回bootを確認", findings=["abnormal shutdown marker"],
                missing_evidence=["直前のkernel log"], confidence="medium",
            )
        return InvestigationResult(summary="前回の証拠を踏まえて継続調査", confidence="high")


class UptimeFleetAgent(ContextAgent):
    def investigate(self, host, goal):
        self.calls.append((host, goal))
        hours = 120 if "X570" in host else 48
        return InvestigationResult(summary=f"uptime_seconds={hours * 3600}", confidence="high")

    def synthesize_fleet(self, goal, hosts, results):
        ranked = sorted(
            ((name, int(result.summary.split("=")[1])) for name, result in results.items()),
            key=lambda item: item[1], reverse=True,
        )
        return "|順位|Host|Uptime|\n|---:|---|---:|\n" + "\n".join(
            f"|{rank}|{name}|{seconds // 3600}h|"
            for rank, (name, seconds) in enumerate(ranked, 1)
        )


class TrafficFunctions:
    def __init__(self, totals):
        self.totals = totals
        self.calls = []

    def execute(self, name, arguments, target_host):
        self.calls.append((name, arguments, target_host))
        value = self.totals[target_host]
        if value is None:
            return {"untrusted_evidence": {"status": "error"}}
        return {"untrusted_evidence": {
            "status": "available", "period": "yesterday", "interface": "eth0",
            "rx_bytes": value // 4, "tx_bytes": value - value // 4,
            "total_bytes": value,
        }}


class FleetAgent(ContextAgent):
    def __init__(self, totals):
        super().__init__()
        self.functions = TrafficFunctions(totals)


def make_service(tmp_path, agent, names=("Garage-X570", "garage-h12ssl-nt")):
    db = Database(tmp_path / "db.sqlite")
    aliases = {"Garage-X570": ["X570"], "garage-h12ssl-nt": ["h12ssl"]}
    hosts = {
        name: Host(name=name, aliases=aliases[name], address=f"192.0.2.{index + 1}", ssh_user="agent")
        for index, name in enumerate(names)
    }
    return AgentService(
        HostRegistry(hosts=hosts), object(), object(), JobManager(db), ConversationStore(db),
        agent=agent, max_parallel_hosts=2,
    )


def test_production_cli_discovery_and_single_host_followups_reach_one_agent(tmp_path):
    agent = ContextAgent()
    service = make_service(tmp_path, agent)

    first = asyncio.run(service.handle_question(
        "X570にvast cli入ってるからレント状況見て。分からなかったらhelp見て", 1, 2,
    ))
    second = asyncio.run(service.handle_question("NVTOP入ってる？", 1, 2))
    third = asyncio.run(service.handle_question("nvidia-smiの結果見せて", 1, 2))

    assert all(reply.job_id is not None for reply in (first, second, third))
    assert [host for host, _ in agent.calls] == ["Garage-X570"] * 3
    assert "host operation intent is uncertain" not in "".join(
        reply.text for reply in (first, second, third)
    )


def test_two_gpu_followup_uses_immediate_host_context(tmp_path):
    agent = ContextAgent()
    service = make_service(tmp_path, agent)
    asyncio.run(service.handle_question("h12sslのGPU状態見て", 1, 2))
    reply = asyncio.run(service.handle_question("二枚あるでしょう？", 1, 2))
    assert reply.job_id is not None
    assert agent.calls[-1] == ("garage-h12ssl-nt", "二枚あるでしょう？")


def test_fleet_vnstat_is_ranked_and_partial_failure_is_preserved(tmp_path):
    totals = {"Garage-X570": 5 * 1024**3, "garage-h12ssl-nt": None}
    agent = FleetAgent(totals)
    service = make_service(tmp_path, agent)

    reply = asyncio.run(service.handle_question(
        "全台の昨日の通信量をvnstatで拾って多い順", 1, 2,
    ))

    assert "| 1 | Garage-X570" in reply.text
    assert "5.0 GiB" in reply.text
    assert "Unavailable" in reply.text and "garage-h12ssl-nt" in reply.text
    assert all(call[0] == "query_traffic_history" for call in agent.functions.calls)
    assert all(call[1]["period"] == "yesterday" for call in agent.functions.calls)
    assert service.conversations.get(1, 2).last_host is None


def test_fleet_result_cannot_authorize_batch_write(tmp_path):
    agent = FleetAgent({"Garage-X570": 1, "garage-h12ssl-nt": 2})
    service = make_service(tmp_path, agent)
    asyncio.run(service.handle_question("全台の昨日の通信量をvnstatで拾って多い順", 1, 2))
    reply = asyncio.run(service.handle_question("全部再起動して", 1, 2))
    assert reply.proposal is None
    assert "fleet WRITE" in reply.text


def test_docker_fix_is_not_captured_by_deterministic_read(tmp_path):
    agent = ContextAgent()
    service = make_service(tmp_path, agent)
    reply = asyncio.run(service.handle_question("X570のdocker状態直して", 1, 2))
    assert reply.job_id is not None
    assert agent.calls == [("Garage-X570", "X570のdocker状態直して")]


def test_generic_fleet_uptime_uses_per_host_agent_and_synthesis(tmp_path):
    agent = UptimeFleetAgent()
    service = make_service(tmp_path, agent)
    reply = asyncio.run(service.handle_question(
        "全台のuptimeを長い順にソートして表で一括表示", 1, 2,
    ))
    assert "|1|Garage-X570|120h|" in reply.text
    assert "|2|garage-h12ssl-nt|48h|" in reply.text
    assert {host for host, _ in agent.calls} == {"Garage-X570", "garage-h12ssl-nt"}


def test_compact_investigation_context_supports_natural_followups(tmp_path):
    agent = ContextAwareAgent()
    service = make_service(tmp_path, agent)
    asyncio.run(service.handle_question(
        "garage-h12ssl-nt の最終ブートは手動か不具合か", 1, 2,
    ))
    concise = asyncio.run(service.handle_question("つまり？", 1, 2))
    logs = asyncio.run(service.handle_question("ログ見て理由までわからないか", 1, 2))
    assert concise.job_id is not None and logs.job_id is not None
    assert agent.calls[-2:] == [
        ("garage-h12ssl-nt", "つまり？"),
        ("garage-h12ssl-nt", "ログ見て理由までわからないか"),
    ]
    assert agent.contexts[1]["previous_goal"].startswith("garage-h12ssl-nt")
    assert "abnormal shutdown marker" in agent.contexts[1]["key_findings"]
    assert "untrusted_evidence" not in str(agent.contexts[1])


def test_old_agent_docker_state_request_keeps_useful_fast_read(tmp_path):
    class DockerInspection:
        def inspect_and_record(self, host, executor, scope, token):
            from types import SimpleNamespace

            observation = SimpleNamespace(
                host=host.name, ssh_ok=True, signatures=[],
                gpu=SimpleNamespace(nvml_ok=False, devices=[]),
                services={"docker": "active"},
                details={"docker_status": "service=active\nabc|worker|Up 2 hours|image"},
            )
            return SimpleNamespace(observation=observation)

    agent = ContextAgent()
    service = make_service(tmp_path, agent)
    service.inspection = DockerInspection()
    reply = asyncio.run(service.handle_question("X570のdocker状態確認", 1, 2))
    assert "docker: active" in reply.text
    assert "running=1" in reply.text


def test_bare_all_means_all_gpus_not_all_hosts(tmp_path):
    agent = ContextAgent()
    service = make_service(tmp_path, agent)
    asyncio.run(service.handle_question("h12sslのGPU状態見て", 1, 2))
    reply = asyncio.run(service.handle_question("GPU全部見せて", 1, 2))
    assert reply.job_id is not None
    assert agent.calls[-1] == ("garage-h12ssl-nt", "GPU全部見せて")


class FleetMutationAgent(ContextAgent):
    def investigate(self, host, goal):
        self.calls.append((host, goal))
        return InvestigationResult(
            summary="current state inspected", mutation_requested=True,
            mutation_goal=goal, confidence="high",
        )


@pytest.mark.parametrize("user_text", [
    "全台のdocker止めて", "全台にnvitop入れて", "全台のGPUクロック下げて",
])
def test_fleet_mutation_language_never_creates_batch_proposal(tmp_path, user_text):
    agent = FleetMutationAgent()
    service = make_service(tmp_path, agent)
    reply = asyncio.run(service.handle_question(user_text, 1, 2))
    assert reply.proposal is None
    assert "fleet WRITE" in reply.text
    assert len(agent.calls) == 2
