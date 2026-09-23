import asyncio

from vast_agent.actions.coordinator import ActionCoordinator
from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import PreflightSnapshot, VerificationResult
from vast_agent.actions.operation_plan import OperationPlan
from vast_agent.agent.models import InvestigationResult
from vast_agent.config import HostRegistry, OperationsSettings
from vast_agent.conversation.state import ConversationStore
from vast_agent.jobs.manager import JobManager
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult
from vast_agent.services.agent_service import AgentService
from vast_agent.storage.database import Database


class Preflight:
    def collect(self, host, request):
        return PreflightSnapshot()


class Verifier:
    def verify(self, host, request):
        return VerificationResult(success=False, status="UNAVAILABLE", summary="unused")


class Remote:
    def __init__(self):
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        command = tuple(command)
        self.calls.append(command)
        if command[:2] == ("pgrep", "-f"):
            return ToolResult(success=True, exit_code=0, stdout="123 SRBMiner-MULTI\n", duration_ms=1)
        if command[:2] == ("virsh", "list"):
            return ToolResult(success=True, exit_code=0, stdout="", duration_ms=1)
        return ToolResult(success=True, exit_code=0, duration_ms=1)


class PlanningAgent:
    def __init__(self):
        self.investigations = []
        self.plans = []

    def investigate(self, host, question):
        self.investigations.append((host, question))
        return InvestigationResult(
            summary="GPU0: SRBMiner active, 340W, 1800MHz, 70C; nvidia-smi help supports clock lock",
            findings=["GPU0 process SRBMiner-MULTI", "power 340 W", "clock 1800 MHz"],
            confidence="high", recommended_action="NONE",
        )

    def plan_operation(self, host, host_source, message, investigation):
        self.plans.append((host, host_source, message, investigation.summary))
        return OperationPlan(
            host=host, host_source=host_source, executable="nvidia-smi",
            argv=["-i", "0", "--lock-gpu-clocks=1500,1500"], requires_sudo=True,
            target="GPU 0", reason="reduce loaded GPU toward the requested 300 W target",
            expected_effect="apply one bounded GPU clock restriction",
            known_side_effects=["active SRBMiner workload performance changes"],
            verification_plan="re-read GPU clock, power, temperature, utilization, and process",
            verification_kind="gpu_state", verification_target="0",
            command_source="current user request + CLI help + READ evidence",
            current_relevant_state="340 W, 1800 MHz, 70 C",
            active_workload="SRBMiner-MULTI active (materially affected)", running_vm="none",
            rollback="new Proposal for nvidia-smi --reset-gpu-clocks",
        )

    def answer_general(self, question):
        return "general"


def service(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    host = Host(name="garage-h12ssl-nt", aliases=["h12ssl"], address="192.0.2.8", ssh_user="agent")
    hosts = HostRegistry(hosts={host.name: host})
    settings = OperationsSettings(enabled=True, generic_operations_enabled=False)
    remote = Remote()
    actions = ActionCoordinator(
        db, hosts, settings, Preflight(), TypedActionExecutor(remote, settings), Verifier(), 7, 9,
    )
    agent = PlanningAgent()
    return AgentService(
        hosts, object(), remote, JobManager(db), ConversationStore(db), agent=agent, actions=actions,
    ), agent, db, remote


def test_gpu_clock_write_inherits_high_confidence_host_and_proposes(tmp_path):
    app, agent, db, remote = service(tmp_path)
    first = asyncio.run(app.handle_question("h12sslの状況は？GPU動いてる？", 7, 9))
    assert first.job_id is not None

    message = "GPU0がマイナーか、クロック制限掛けて消費電力を300W位まで絞って、PLじゃなくクロック制限の方ね"
    reply = asyncio.run(app.handle_question(message, 7, 9))

    assert reply.proposal is not None
    assert reply.proposal.plan.host == "garage-h12ssl-nt"
    assert reply.proposal.plan.host_source == "conversation context"
    assert "Host source: conversation context" in reply.text
    assert "SRBMiner-MULTI" in reply.text
    assert "generic operation execution is disabled" in reply.text
    assert db.pending_approval_count() == 1
    assert not any("--lock-gpu-clocks" in " ".join(call) for call in remote.calls)
    assert agent.plans[0][:3] == ("garage-h12ssl-nt", "conversation context", message)


def test_context_does_not_invent_target_or_reuse_fleet_context(tmp_path):
    app, _, _, _ = service(tmp_path)
    asyncio.run(app.handle_question("h12sslの状況は？GPU動いてる？", 7, 9))
    vague = asyncio.run(app.handle_question("そっち適当に絞って", 7, 9))
    assert vague.proposal is None
    assert "確認" in vague.text

    state = app.conversations.get(7, 9)
    state.last_host = None  # fleet results deliberately clear single-host authority
    state.last_scope = "gpu"
    app.conversations.save(7, 9, state)
    ambiguous = asyncio.run(app.handle_question("GPU0をクロック制限して", 7, 9))
    assert ambiguous.proposal is None
