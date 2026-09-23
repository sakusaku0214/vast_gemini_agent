import asyncio
import json

from vast_agent.actions.coordinator import ActionCoordinator
from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import PreflightSnapshot, VerificationResult
from vast_agent.actions.operation_plan import OperationPlan
from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.models import AgentResponse, InvestigationResult
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.agent.router import WRITE_WORDS
from vast_agent.config import GeminiSettings, HostRegistry, OperationsSettings
from vast_agent.conversation.state import ConversationStore
from vast_agent.jobs.manager import JobManager
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult
from vast_agent.services.agent_service import AgentService
from vast_agent.services.inspection import InspectionService
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


def test_clock_release_proposal_uses_real_fresh_read_path(tmp_path):
    """Regression for production orchestration, not a pre-grounded InvestigationResult."""
    db = Database(tmp_path / "real-path.sqlite")
    db.migrate()
    host = Host(
        name="garage-h12ssl-nt", aliases=["h12ssl"], address="192.0.2.8", ssh_user="agent",
    )
    registry = HostRegistry(hosts={host.name: host})

    class CommandRemote(Remote):
        def execute(self, host, command, timeout, cancellation=None):
            del host, timeout, cancellation
            command = tuple(command)
            self.calls.append(command)
            output = {
                ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"): "0\n1\n",
                ("nvidia-smi", "-i", "0", "-q", "-d", "CLOCK"): (
                    "GPU 00000000:01:00.0\n    GPU Locked Clocks\n"
                    "        Min : 1200 MHz\n        Max : 1800 MHz\n"
                ),
                ("nvidia-smi", "-i", "1", "-q", "-d", "CLOCK"): (
                    "GPU 00000000:02:00.0\n    GPU Locked Clocks\n"
                    "        Min : N/A\n        Max : N/A\n"
                ),
                ("nvidia-smi", "--help"): "-rgc, --reset-gpu-clocks  Reset locked GPU clocks\n",
            }.get(command, "")
            return ToolResult(success=True, exit_code=0, stdout=output, duration_ms=1)

    def tool_calls():
        specs = [
            ("run_readonly_argv", {
                "host": host.name, "executable": "nvidia-smi",
                "argv": ["--query-gpu=index", "--format=csv,noheader,nounits"],
                "reason": "enumerate GPU indexes before checking every clock lock",
            }),
            ("run_readonly_argv", {
                "host": host.name, "executable": "nvidia-smi",
                "argv": ["-i", "0", "-q", "-d", "CLOCK"],
                "reason": "read GPU 0 current clock-lock state",
            }),
            ("run_readonly_argv", {
                "host": host.name, "executable": "nvidia-smi",
                "argv": ["-i", "1", "-q", "-d", "CLOCK"],
                "reason": "read GPU 1 current clock-lock state",
            }),
            ("query_cli_help", {
                "host": host.name, "executable": "nvidia-smi", "argv": ["--help"],
                "reason": "confirm bounded reset syntax",
            }),
        ]
        return AgentResponse(steps=[{
            "type": "function_call", "name": name, "arguments": args, "id": str(index),
        } for index, (name, args) in enumerate(specs, 1)])

    conclusion = AgentResponse(output_text=json.dumps({
        "summary": "GPU 0だけに1200-1800 MHzのクロック制限があります。GPU 1は制限なしです。",
        "findings": ["GPU indexes: 0, 1", "GPU 0 Locked Clocks 1200-1800 MHz", "GPU 1 N/A"],
        "signatures": [], "confidence": "high", "recommended_action": "NONE",
        "missing_evidence": [], "capability_gaps": [], "mutation_requested": True,
        "mutation_goal": "クロック制限を開放", "stop_reason": "ANSWERABLE",
    }))
    expected_plan = OperationPlan(
        host=host.name, host_source="conversation context", executable="nvidia-smi",
        argv=["-i", "0", "--reset-gpu-clocks"], requires_sudo=True, target="GPU 0",
        reason="fresh READ shows GPU 0 has locked clocks", expected_effect="release GPU 0 clock lock",
        known_side_effects=["GPU clocks return to driver defaults"],
        verification_plan="re-read GPU 0 clock state", verification_kind="gpu_state",
        verification_target="0", command_source="fresh clock query and bounded nvidia-smi help",
        current_relevant_state="GPU 0 Locked Clocks Min 1200 MHz Max 1800 MHz",
        active_workload="unknown", running_vm="unknown",
        rollback="a separately approved clock-lock Proposal",
    )
    client = ScriptedGeminiClient([
        tool_calls(), conclusion,
        AgentResponse(output_text=expected_plan.model_dump_json()),
    ])
    remote = CommandRemote()
    inspection = InspectionService(db, tmp_path / "logs")
    functions = FunctionExecutor(registry, inspection, db, remote)
    agent = InvestigationAgent(client, functions, db, GeminiSettings())
    settings = OperationsSettings(enabled=True, generic_operations_enabled=False)
    actions = ActionCoordinator(
        db, registry, settings, Preflight(), TypedActionExecutor(remote, settings), Verifier(), 7, 9,
    )
    app = AgentService(
        registry, inspection, remote, JobManager(db), ConversationStore(db),
        agent=agent, actions=actions,
    )
    state = app.conversations.get(7, 9)
    state.last_host = host.name
    state.last_scope = "gpu"
    state.write_context_host = host.name
    state.current_hosts = (host.name,)
    state.context_kind = "single"
    state.context_source = "explicit"
    app.conversations.save(7, 9, state)

    reply = asyncio.run(app.handle_question(
        "クロック制限入れてるから開放しておいて。どんなコマンド使うか教えて", 7, 9,
    ))

    assert reply.proposal is not None
    assert reply.proposal.plan == expected_plan
    assert reply.proposal.plan.argv == ["-i", "0", "--reset-gpu-clocks"]
    assert remote.calls[:4] == [
        ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
        ("nvidia-smi", "-i", "0", "-q", "-d", "CLOCK"),
        ("nvidia-smi", "-i", "1", "-q", "-d", "CLOCK"),
        ("nvidia-smi", "--help"),
    ]
    assert not any("--reset-gpu-clocks" in command for command in remote.calls)


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
