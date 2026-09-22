from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from vast_agent.actions.coordinator import ActionCoordinator
from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    PackageInstallParameters,
    PreflightSnapshot,
    VerificationResult,
)
from vast_agent.actions.policies import PolicyEngine
from vast_agent.actions.resolver import ActionIntentResolver
from vast_agent.config import HostRegistry, OperationsSettings
from vast_agent.conversation.state import ConversationStore
from vast_agent.jobs.manager import JobManager
from vast_agent.models.host import Host
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
    ({"package_candidate": None}, "PACKAGE_CANDIDATE_NOT_FOUND"),
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


@pytest.mark.parametrize("value", ["--help", ";reboot", "../x", "vnstat\nreboot", "Apt"])
def test_invalid_package_names_never_reach_executor(value):
    with pytest.raises(ValidationError):
        PackageInstallParameters(package_name=value)


def test_catalog_boundary_rejects_unknown_package_before_preflight(tmp_path):
    _, actions, remote, preflight = coordinator(tmp_path)
    unknown = ActionRequest(
        host="garage-mag", action_type=ActionType.PACKAGE_INSTALL,
        parameters=PackageInstallParameters(package_name="unknown-safe-name"),
    )
    with pytest.raises(ValueError, match="do not match"):
        actions.propose(unknown, "7")
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


def test_resolver_only_emits_catalog_package_and_gemini_has_no_write_tool():
    hosts = HostRegistry(hosts={
        "garage-mag": Host(name="garage-mag", address="192.0.2.8", ssh_user="agent"),
    })
    resolver = ActionIntentResolver(hosts)
    resolved = resolver.resolve("garage-magの昨日の通信量見たい。必要なら入れて")
    assert resolved is not None
    assert resolved.action_type == ActionType.PACKAGE_INSTALL
    assert resolved.parameters.package_name == "vnstat"
    assert resolver.resolve("garage-magにcurl入れて") is None

    from vast_agent.agent.functions import FUNCTION_DECLARATIONS
    declarations = str(FUNCTION_DECLARATIONS).casefold()
    assert "package_install" not in declarations
    assert "apt-get" not in declarations
