from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vast_agent.actions.coordinator import ActionCoordinator
from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    PackageInstallParameters,
    PreflightSnapshot,
    RebootParameters,
    VerificationResult,
)
from vast_agent.actions.policies import PolicyEngine
from vast_agent.actions.preflight import PACKAGE_TOOLS, ProductionPreflightProvider
from vast_agent.actions.resolver import ActionIntentResolver
from vast_agent.agent.models import CapabilityGap, InvestigationResult
from vast_agent.config import HostRegistry, OperationsSettings
from vast_agent.conversation.state import ConversationStore
from vast_agent.jobs.manager import JobManager
from vast_agent.models.host import Host
from vast_agent.models.observation import Observation, SystemSummary
from vast_agent.models.tool_result import ToolResult
from vast_agent.services.agent_service import AgentService
from vast_agent.storage.database import Database


class Preflight:
    def __init__(self, state):
        self.state = state
        self.calls = 0

    def collect(self, host, request):
        self.calls += 1
        return self.state


class Remote:
    def __init__(self, success=True):
        self.success = success
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        self.calls.append(tuple(command))
        return ToolResult(success=self.success, exit_code=0 if self.success else 1, duration_ms=1)


class Verify:
    def __init__(self, success=True):
        self.success = success

    def verify(self, host, request):
        return VerificationResult(
            success=self.success,
            status="VERIFIED" if self.success else "FAILED",
            summary=f"{request.parameters.package_name}: installed" if self.success else "absent",
        )


class ProductionRemote:
    """Command-aware remote for exercising the real package preflight collector."""

    def __init__(self, *, dpkg_exit=1, dpkg_stdout="", fuser_exit=1):
        self.dpkg_exit = dpkg_exit
        self.dpkg_stdout = dpkg_stdout
        self.fuser_exit = fuser_exit
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        argv = tuple(command)
        self.calls.append(argv)
        if argv[:2] == ("sudo", "-n") and argv[2:] == ("true",):
            return ToolResult(success=True, exit_code=0, duration_ms=1)
        if argv[0] == "dpkg-query":
            return ToolResult(
                success=self.dpkg_exit == 0, exit_code=self.dpkg_exit,
                stdout=self.dpkg_stdout, duration_ms=1,
            )
        if argv[:2] == ("apt-cache", "policy"):
            return ToolResult(
                success=True, exit_code=0, stdout="  Candidate: 2.12-1\n", duration_ms=1,
            )
        if argv[:3] == ("sudo", "-n", "fuser"):
            return ToolResult(
                success=self.fuser_exit == 0, exit_code=self.fuser_exit, duration_ms=1,
            )
        if argv[:2] == ("which", "--") and argv[2] in PACKAGE_TOOLS:
            return ToolResult(
                success=True, exit_code=0, stdout=f"/usr/bin/{argv[2]}\n", duration_ms=1,
            )
        raise AssertionError(f"unexpected argv: {argv}")


class ProductionInspection:
    def inspect_and_record(self, host, executor):
        observation = Observation(
            host=host.name, ssh_ok=True,
            system=SystemSummary(filesystem_max_percent=10, failed_units=0),
        )
        ok = ToolResult(success=True, exit_code=0, duration_ms=1)
        return SimpleNamespace(
            observation=observation,
            tool_results={"host_ping": ok, "get_system_health": ok},
        )


def production_state(remote):
    settings = OperationsSettings(enabled=True, allowed_actions=["PACKAGE_INSTALL"])
    provider = ProductionPreflightProvider(ProductionInspection(), remote, settings)
    host = Host(name="garage-mag", address="192.0.2.8", ssh_user="agent")
    return provider.collect(host, request()), remote


def safe_state(**updates):
    state = PreflightSnapshot(
        ssh_reachable=True, sudo_available=True, target_exists=True, evidence_complete=True,
        package_installed=False, package_candidate="2.12-1", package_manager_busy=False,
    )
    return state.model_copy(update=updates)


def request(package="vnstat"):
    return ActionRequest(
        host="garage-mag", action_type=ActionType.PACKAGE_INSTALL,
        parameters=PackageInstallParameters(
            package_name=package, expected_capability="traffic_history",
            reason="traffic history capability is unavailable",
        ),
    )


def coordinator(tmp_path, state=None, command_success=True, verify=True):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    hosts = HostRegistry(hosts={
        "garage-mag": Host(name="garage-mag", address="192.0.2.8", ssh_user="agent"),
    })
    settings = OperationsSettings(enabled=True, allowed_actions=["PACKAGE_INSTALL"])
    remote = Remote(command_success)
    preflight = Preflight(state or safe_state())
    actions = ActionCoordinator(
        db, hosts, settings, preflight, TypedActionExecutor(remote, settings), Verify(verify), 7, 9,
    )
    return db, actions, remote, preflight


def test_exact_argv_and_approval_executes_once(tmp_path):
    _, actions, remote, preflight = coordinator(tmp_path)
    proposal = actions.propose(request(), "7")
    assert remote.calls == []
    assert actions.approve(proposal.id, user_id=7, channel_id=9)[0]
    assert remote.calls == [(
        "sudo", "-n", "apt-get", "install", "-y", "--no-install-recommends", "--", "vnstat",
    )]
    assert preflight.calls == 2
    assert not actions.approve(proposal.id, user_id=7, channel_id=9)[0]
    assert len(remote.calls) == 1


@pytest.mark.parametrize(("change", "reason"), [
    ({"package_installed": True}, "PACKAGE_ALREADY_INSTALLED"),
    ({"package_candidate": None}, "PACKAGE_NOT_AVAILABLE"),
    ({"active_workload": True}, "ACTIVE_WORKLOAD"),
    ({"running_vm": True}, "RUNNING_VM"),
    ({"sudo_available": False}, "SUDO_NOT_AVAILABLE"),
    ({"package_manager_busy": True}, "PACKAGE_MANAGER_BUSY"),
])
def test_unsafe_package_preflight_blocks_without_mutation(tmp_path, change, reason):
    db, actions, remote, _ = coordinator(tmp_path, safe_state(**change))
    proposal = actions.propose(request(), "7")
    assert proposal.status == "BLOCKED"
    decision = PolicyEngine().evaluate(request(), safe_state(**change), actions.settings)
    assert reason in decision.reasons
    assert remote.calls == []


@pytest.mark.parametrize("value", [
    "--help", ";reboot", "../x", "vnstat\nreboot", "Apt", "foo;rm", "foo bar",
    "foo/bar", "$(id)", "`id`", "foo|bar", "foo>bar",
])
def test_invalid_package_names_never_reach_executor(value):
    with pytest.raises(ValidationError):
        PackageInstallParameters(package_name=value)


def test_catalog_absence_does_not_reject_explicit_package(tmp_path):
    _, actions, remote, preflight = coordinator(tmp_path)
    unknown = ActionRequest(
        host="garage-mag", action_type=ActionType.PACKAGE_INSTALL,
        parameters=PackageInstallParameters(package_name="unknown-safe-name"),
    )
    proposal = actions.propose(unknown, "7")
    assert proposal.status == "PENDING"
    assert proposal.parameters.expected_capability is None
    assert preflight.calls == 1
    assert remote.calls == []


def test_catalog_backed_acquisition_still_requires_matching_metadata(tmp_path):
    _, actions, remote, preflight = coordinator(tmp_path)
    inconsistent = ActionRequest(
        host="garage-mag", action_type=ActionType.PACKAGE_INSTALL,
        parameters=PackageInstallParameters(
            package_name="nvitop", expected_capability="traffic_history",
        ),
    )
    with pytest.raises(ValueError, match="do not match"):
        actions.propose(inconsistent, "7")
    assert preflight.calls == 0
    assert remote.calls == []


def test_command_failure_and_postcheck_failure_are_not_retried(tmp_path):
    db, actions, remote, _ = coordinator(tmp_path, command_success=False)
    proposal = actions.propose(request(), "7")
    assert not actions.approve(proposal.id, user_id=7, channel_id=9)[0]
    assert len(remote.calls) == 1
    assert db.get_action_proposal(proposal.id)["status"] == "FAILED"

    db2, actions2, remote2, _ = coordinator(tmp_path / "post", verify=False)
    proposal2 = actions2.propose(request(), "7")
    assert not actions2.approve(proposal2.id, user_id=7, channel_id=9)[0]
    assert len(remote2.calls) == 1
    assert db2.get_action_proposal(proposal2.id)["status"] == "FAILED_VERIFY"


def test_natural_language_catalog_and_conditional_present(tmp_path):
    db, actions, remote, preflight = coordinator(
        tmp_path, safe_state(package_installed=True, package_version="2.12-1"),
    )
    service = AgentService(
        actions.hosts, object(), remote, JobManager(db), ConversationStore(db), actions=actions,
    )
    reply = asyncio.run(service.handle_question("garage-magにvnstatなければ入れて", 7, 9))
    assert reply.proposal is None
    assert "既に導入済み" in reply.text
    assert db.pending_approval_count() == 0
    assert remote.calls == []
    assert preflight.calls == 1


def test_resolver_emits_exact_explicit_package_and_gemini_has_no_write_tool():
    hosts = HostRegistry(hosts={
        "garage-mag": Host(
            name="garage-mag", aliases=["mag", "X570"],
            address="192.0.2.8", ssh_user="agent",
        ),
    })
    resolver = ActionIntentResolver(hosts)
    # Goal-shaped acquisition is deliberately left for Gemini capability reasoning;
    # only an explicit catalog package remains on this deterministic path.
    assert resolver.resolve("garage-magの昨日の通信量見たい。必要なら入れて") is None
    resolved = resolver.resolve("garage-magにvnstat入れて")
    assert resolved is not None
    assert resolved.action_type == ActionType.PACKAGE_INSTALL
    assert resolved.parameters.package_name == "vnstat"
    assert resolved.parameters.expected_capability is None
    assert resolved.parameters.reason == "explicit user request"
    for text, package in (
        ("X570にnvitop入れておいて", "nvitop"),
        ("magにhtopインストールして", "htop"),
        ("garage-magへiotop入れといて", "iotop"),
        ("garage-magにnvtop入れて", "nvtop"),
    ):
        dynamic = resolver.resolve(text)
        assert dynamic is not None
        assert dynamic.parameters.package_name == package
        assert dynamic.parameters.expected_capability is None

    assert resolver.resolve("garage-magにGPU監視に便利なの入れて") is None

    for invalid in ("foo;rm", "foo bar", "foo/bar", "$(id)", "`id`", "foo|bar", "foo>bar"):
        with pytest.raises(ValueError):
            resolver.resolve(f"garage-magに{invalid}入れて")

    from vast_agent.agent.functions import FUNCTION_DECLARATIONS
    declarations = str(FUNCTION_DECLARATIONS).casefold()
    assert "package_install" not in declarations
    assert "apt-get" not in declarations


def test_production_preflight_uses_direct_which_argv_and_completes_when_tools_present():
    state, remote = production_state(ProductionRemote(dpkg_exit=1, fuser_exit=1))

    assert state.evidence_complete is True
    assert state.package_installed is False
    assert state.package_manager_busy is False
    presence_calls = [call for call in remote.calls if call[0] == "which"]
    assert presence_calls == [("which", "--", tool) for tool in PACKAGE_TOOLS]
    assert not any(call[0] in {"command", "sh", "bash"} for call in remote.calls)


def test_production_preflight_installed_and_unexpected_dpkg_failure_are_distinct():
    installed, _ = production_state(ProductionRemote(
        dpkg_exit=0, dpkg_stdout="install ok installed\t2.12-1\n", fuser_exit=1,
    ))
    assert installed.package_installed is True
    assert installed.package_version == "2.12-1"
    assert installed.evidence_complete is True

    unknown, _ = production_state(ProductionRemote(dpkg_exit=2, fuser_exit=1))
    assert unknown.package_installed is None
    assert unknown.current_state == "unknown"
    assert unknown.evidence_complete is False
    decision = PolicyEngine().evaluate(request(), unknown, OperationsSettings(
        enabled=True, allowed_actions=["PACKAGE_INSTALL"],
    ))
    assert decision.decision == "BLOCK"
    assert "PREFLIGHT_INCOMPLETE" in decision.reasons


@pytest.mark.parametrize(("busy", "label"), [
    (None, "unknown"),
    (False, "idle"),
    (True, "busy"),
])
def test_proposal_renders_three_package_manager_states(tmp_path, busy, label):
    _, actions, _, _ = coordinator(tmp_path, safe_state(package_manager_busy=busy))
    proposal = actions.propose(request(), "7")
    rendered = AgentService._proposal_text(proposal, "BLOCK" if busy is not False else "ALLOW")
    assert f"package-manager={label}" in rendered
    if busy is None:
        decision = PolicyEngine().evaluate(request(), proposal.preflight_summary, actions.settings)
        assert decision.decision == "BLOCK"
        assert "PACKAGE_MANAGER_BUSY" in decision.reasons


def test_production_preflight_unexpected_fuser_failure_is_unknown_and_blocked():
    state, _ = production_state(ProductionRemote(dpkg_exit=1, fuser_exit=2))
    assert state.package_manager_busy is None
    assert state.details["package_manager"] == "unknown"
    assert state.evidence_complete is False
    decision = PolicyEngine().evaluate(request(), state, OperationsSettings(
        enabled=True, allowed_actions=["PACKAGE_INSTALL"],
    ))
    assert decision.decision == "BLOCK"
    assert "PACKAGE_MANAGER_BUSY" in decision.reasons


class GapAgent:
    def __init__(self, status="missing"):
        self.status = status

    def investigate(self, host, text):
        return InvestigationResult(
            summary="traffic history check", confidence="high", recommended_action="NONE",
            capability_gaps=[CapabilityGap(
                capability_id="traffic_history", status=self.status,
                software_status=self.status,
                reason="package and executable checked", evidence=[f"vnstat: {self.status}"],
                candidate_package="curl", confidence="high",
            )],
        )


class FlexibleInstallAgent:
    def __init__(self):
        self.calls = []

    def interpret_action(self, host, text):
        self.calls.append((host, text))
        return ActionRequest(
            host=host, action_type=ActionType.PACKAGE_INSTALL,
            parameters=PackageInstallParameters(
                package_name="nvitop", reason="explicit user request",
            ),
        )


class RebootIntentAgent:
    def __init__(self):
        self.interpretations = []
        self.investigations = []

    def interpret_action(self, host, text):
        self.interpretations.append((host, text))
        return ActionRequest(
            host=host, action_type=ActionType.HOST_REBOOT,
            parameters=RebootParameters(assessment="HOST_REBOOT_CANDIDATE"),
        )

    def investigate(self, host, text):
        self.investigations.append((host, text))
        return InvestigationResult(
            summary="assessment only", recommended_action="HOST_REBOOT_CANDIDATE",
        )


def test_service_does_not_run_a_pre_agent_action_interpreter(tmp_path):
    db, actions, remote, preflight = coordinator(tmp_path)
    agent = FlexibleInstallAgent()
    service = AgentService(
        actions.hosts, object(), remote, JobManager(db), ConversationStore(db),
        agent=agent, actions=actions,
    )

    reply = asyncio.run(service.handle_question(
        "garage-magへnvitopを導入しておいて", 7, 9,
    ))

    assert agent.calls == []
    assert reply.proposal is None
    assert preflight.calls == 0
    assert remote.calls == []


def test_reboot_fallback_requires_current_turn_host_and_preserves_advice_assessment(tmp_path):
    db, actions, remote, _ = coordinator(tmp_path)
    agent = RebootIntentAgent()
    service = AgentService(
        actions.hosts, object(), remote, JobManager(db), ConversationStore(db),
        agent=agent, actions=actions,
    )

    hostless = asyncio.run(service.handle_question("再起動して", 7, 9))
    advice = asyncio.run(service.handle_question("garage-magは再起動した方がいい？", 7, 9))

    assert hostless.proposal is None
    assert "WRITE" in hostless.text
    assert agent.interpretations == []
    assert agent.investigations == [("garage-mag", "garage-magは再起動した方がいい？")]
    assert advice.proposal is None
    assert "assessment only" in advice.text
    assert db.pending_approval_count() == 0
    assert remote.calls == []


@pytest.mark.parametrize(("text", "status", "proposal_count"), [
    ("garage-magの昨日の通信量見て", "missing", 0),
    # A stubbed model gap has no session READ evidence and must fail closed.
    ("garage-magの昨日の通信量見たい。必要なら入れて", "missing", 0),
    ("garage-magの昨日の通信量見たい。必要なら入れて", "available", 0),
    ("garage-magの昨日の通信量見たい。必要なら入れて", "unknown", 0),
])
def test_goal_gap_bridge_proposes_only_confirmed_missing_with_consent(
    tmp_path, text, status, proposal_count,
):
    db, actions, remote, _ = coordinator(tmp_path)
    service = AgentService(
        actions.hosts, object(), remote, JobManager(db), ConversationStore(db),
        agent=GapAgent(status), actions=actions,
    )
    reply = asyncio.run(service.handle_question(text, 7, 9))
    assert db.pending_approval_count() == proposal_count
    assert (reply.proposal is not None) is bool(proposal_count)
    assert remote.calls == []
    if proposal_count:
        assert reply.proposal.parameters.package_name == "vnstat"


def test_bare_acquisition_followup_never_reuses_read_last_host(tmp_path):
    db, actions, remote, _ = coordinator(tmp_path)
    service = AgentService(
        actions.hosts, object(), remote, JobManager(db), ConversationStore(db),
        agent=GapAgent(), actions=actions,
    )
    asyncio.run(service.handle_question("garage-magの昨日の通信量見て", 7, 9))
    reply = asyncio.run(service.handle_question("じゃあ必要なら入れて", 7, 9))
    assert "hostを明示" in reply.text
    assert reply.proposal is None
    assert db.pending_approval_count() == 0
