import asyncio

from vast_agent.actions.coordinator import ActionCoordinator
from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import PreflightSnapshot, VerificationResult
from vast_agent.actions.operation_plan import OperationPlan
from vast_agent.agent.models import InvestigationResult
from vast_agent.agent.router import WRITE_WORDS
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

    def investigate(self, host, question, context=None):
        self.investigations.append((host, question))
        if "開放" in question:
            result = InvestigationResult(
                summary="GPU0 has locked clocks 1200-1800 MHz",
                findings=["GPU 0 Locked Clocks Min 1200 MHz Max 1800 MHz"],
                confidence="high", mutation_requested=True, mutation_goal=question,
            )
            result.ground_from_validated_evidence(
                "GPU 0 Locked Clocks Min 1200 MHz Max 1800 MHz",
            )
            return result
        return InvestigationResult(
            summary="GPU0: SRBMiner active, 340W, 1800MHz, 70C; nvidia-smi help supports clock lock",
            findings=[
                "GPU0 process SRBMiner-MULTI", "power 340 W", "clock 1800 MHz",
                "validated supported clock 1500 MHz",
            ],
            confidence="high", recommended_action="NONE",
            mutation_requested=("抑えて" in question or "動かし直して" in question),
            mutation_goal=question if ("抑えて" in question or "動かし直して" in question) else None,
        )

    def plan_operation(self, host, host_source, message, investigation):
        self.plans.append((host, host_source, message, investigation.summary))
        if "foo.service" in message:
            return None
        release = "開放" in message
        return OperationPlan(
            host=host, host_source=host_source, executable="nvidia-smi",
            argv=["-i", "0", "--reset-gpu-clocks"] if release else
                 ["-i", "0", "--lock-gpu-clocks=1500,1500"], requires_sudo=True,
            target="GPU 0", reason=("release the observed GPU clock lock" if release else
                                    "reduce loaded GPU toward the requested 300 W target"),
            expected_effect=("release GPU clock restriction" if release else
                             "apply one bounded GPU clock restriction"),
            known_side_effects=["active SRBMiner workload performance changes"],
            verification_plan="re-read GPU clock, power, temperature, utilization, and process",
            verification_kind="gpu_state", verification_target="0",
            command_source="bounded value derived from request + CLI help + READ evidence",
            current_relevant_state=("GPU0 locked at 1200-1800 MHz" if release else
                                    "340 W, 1800 MHz, 70 C"),
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

    message = "GPU0ちょっと抑えて。PLじゃなくて300Wくらいにしたい"
    assert not any(word in message.casefold() for word in WRITE_WORDS)
    reply = asyncio.run(app.handle_question(message, 7, 9))

    assert reply.proposal is not None
    assert reply.proposal.plan.host == "garage-h12ssl-nt"
    assert reply.proposal.plan.host_source == "conversation context"
    assert "Host source: conversation context" in reply.text
    assert "SRBMiner-MULTI" in reply.text
    assert "generic operation execution is disabled" not in reply.text
    assert db.pending_approval_count() == 1
    assert not any("--lock-gpu-clocks" in " ".join(call) for call in remote.calls)
    assert agent.plans[0][:3] == ("garage-h12ssl-nt", "conversation context", message)


def test_natural_clock_release_investigates_inherited_host_before_proposal(tmp_path):
    app, agent, db, remote = service(tmp_path)
    first = asyncio.run(app.handle_question("h12sslのGPU状態見て", 7, 9))
    assert first.job_id is not None

    reply = asyncio.run(app.handle_question(
        "クロック制限入れてるから開放しておいて。どんなコマンド使ったか教えて", 7, 9,
    ))

    assert reply.proposal is not None
    assert reply.proposal.plan.host_source == "conversation context"
    assert reply.proposal.plan.argv == ["-i", "0", "--reset-gpu-clocks"]
    assert "GPU0 locked at 1200-1800 MHz" in reply.text
    assert "target grounding" not in reply.text
    assert agent.investigations == [(
        "garage-h12ssl-nt", "クロック制限入れてるから開放しておいて。どんなコマンド使ったか教えて",
    )]
    assert db.pending_approval_count() == 1
    assert not any("--lock-gpu-clocks" in " ".join(call) for call in remote.calls)


def test_application_write_intent_proposes_when_model_flag_is_false(tmp_path):
    app, agent, db, remote = service(tmp_path)
    asyncio.run(app.handle_question("h12sslのGPU状態見て", 7, 9))
    original = agent.investigate

    def without_model_write_flag(host, question, context=None):
        result = original(host, question, context)
        result.mutation_requested = False
        return result

    agent.investigate = without_model_write_flag
    reply = asyncio.run(app.handle_question(
        "クロック制限入れてるから開放しておいて。どんなコマンド使うか教えて", 7, 9,
    ))

    assert reply.proposal is not None
    assert reply.proposal.plan.argv == ["-i", "0", "--reset-gpu-clocks"]
    assert db.pending_approval_count() == 1
    assert not any("--reset-gpu-clocks" in call for call in remote.calls)


def test_command_hint_continues_pending_write_goal(tmp_path):
    app, agent, _, _ = service(tmp_path)
    asyncio.run(app.handle_question("h12sslのGPU状態見て", 7, 9))
    original_plan = agent.plan_operation
    agent.plan_operation = lambda *args: None
    unresolved = asyncio.run(app.handle_question("クロック制限を開放しておいて", 7, 9))
    assert unresolved.proposal is None

    agent.plan_operation = original_plan
    reply = asyncio.run(app.handle_question("sudo nvidia-smi -rgc だったかな", 7, 9))

    assert reply.proposal is not None
    assert "Pending requested operation" in agent.investigations[-1][1]
    assert "sudo nvidia-smi -rgc" in agent.investigations[-1][1]


def test_semantic_advice_never_creates_proposal(tmp_path):
    app, agent, db, _ = service(tmp_path)
    reply = asyncio.run(app.handle_question("h12sslのGPU0少し下げた方がいい？", 7, 9))
    assert reply.proposal is None
    assert reply.job_id is not None
    assert db.pending_approval_count() == 0
    assert agent.plans == []


def test_explicit_host_unknown_service_write_uses_semantic_fallback(tmp_path):
    app, agent, _, _ = service(tmp_path)
    # The fake planner uses a GPU plan and is correctly rejected because its host is fixed but
    # its GPU target was not grounded in this service request. Routing still reached planning.
    reply = asyncio.run(app.handle_question("h12sslのfoo.serviceを動かし直して", 7, 9))
    assert reply.proposal is None
    assert "曖昧" in reply.text
    assert agent.investigations[-1][0] == "garage-h12ssl-nt"


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
