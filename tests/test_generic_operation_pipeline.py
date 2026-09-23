from vast_agent.actions.coordinator import ActionCoordinator
from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import PreflightSnapshot, VerificationResult
from vast_agent.actions.operation_plan import OperationPlan
from vast_agent.config import HostRegistry, OperationsSettings
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult
from vast_agent.storage.database import Database


class UnusedTypedPreflight:
    def collect(self, host, request):
        return PreflightSnapshot()


class UnusedVerifier:
    def verify(self, host, request):
        return VerificationResult(success=False, status="UNAVAILABLE", summary="unused")


class GenericRemote:
    def __init__(self):
        self.calls = []
        self.workload = ""

    def execute(self, host, command, timeout, cancellation=None):
        command = tuple(command)
        self.calls.append(command)
        if command == ("true",) or command == ("sudo", "-n", "true"):
            return ToolResult(success=True, exit_code=0, duration_ms=1)
        if command[:2] == ("pgrep", "-f"):
            return ToolResult(success=bool(self.workload), exit_code=0 if self.workload else 1,
                              stdout=self.workload, duration_ms=1)
        if command[:2] == ("virsh", "list"):
            return ToolResult(success=True, exit_code=0, stdout="", duration_ms=1)
        if command[:2] == ("systemctl", "is-active"):
            return ToolResult(success=True, exit_code=0, stdout="active\n", duration_ms=1)
        return ToolResult(success=True, exit_code=0, duration_ms=1)

    @property
    def mutations(self):
        return [call for call in self.calls if call == (
            "sudo", "-n", "systemctl", "restart", "vastai.service",
        )]


def build(tmp_path, *, generic=True):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    host = Host(name="garage-x570", address="192.0.2.1", ssh_user="agent")
    hosts = HostRegistry(hosts={host.name: host})
    settings = OperationsSettings(enabled=True, generic_operations_enabled=generic)
    remote = GenericRemote()
    coordinator = ActionCoordinator(
        db, hosts, settings, UnusedTypedPreflight(), TypedActionExecutor(remote, settings),
        UnusedVerifier(), owner_id=7, channel_id=9,
    )
    return db, coordinator, remote


def plan():
    return OperationPlan(
        host="garage-x570", host_source="current message", executable="systemctl",
        argv=["restart", "vastai.service"], requires_sudo=True, target="vastai.service",
        reason="daemon is unresponsive", expected_effect="restart one service",
        verification_plan="read systemd active state", verification_kind="service_state",
        verification_target="vastai.service", command_source="CLI help + Agent planning",
        current_relevant_state="failed", active_workload="none", running_vm="none",
    )


def test_reject_and_unauthorized_approval_execute_zero(tmp_path):
    _, coordinator, remote = build(tmp_path)
    proposal = coordinator.propose_operation(plan(), "7")
    assert not coordinator.approve(proposal.id, user_id=8, channel_id=9)[0]
    assert remote.mutations == []
    assert coordinator.reject(proposal.id, user_id=7, channel_id=9) == (True, "REJECTED")
    assert remote.mutations == []


def test_approve_executes_exact_argv_once_and_verifies(tmp_path):
    db, coordinator, remote = build(tmp_path)
    proposal = coordinator.propose_operation(plan(), "7")
    assert coordinator.approve(proposal.id, user_id=7, channel_id=9)[0]
    assert remote.mutations == [("sudo", "-n", "systemctl", "restart", "vastai.service")]
    assert ("systemctl", "is-active", "--", "vastai.service") in remote.calls
    assert db.get_action_proposal(proposal.id)["status"] == "SUCCEEDED"


def test_generic_execution_default_deny(tmp_path):
    db, coordinator, remote = build(tmp_path, generic=False)
    proposal = coordinator.propose_operation(plan(), "7")
    assert coordinator.approve(proposal.id, user_id=7, channel_id=9) == (
        False, "GENERIC_OPERATIONS_DISABLED",
    )
    assert remote.mutations == []
    assert db.get_action_proposal(proposal.id)["status"] == "INVALIDATED"


def test_fresh_state_change_invalidates_before_execution(tmp_path):
    db, coordinator, remote = build(tmp_path)
    proposal = coordinator.propose_operation(plan(), "7")
    remote.workload = "123 SRBMiner-MULTI\n"
    assert coordinator.approve(proposal.id, user_id=7, channel_id=9) == (
        False, "PREFLIGHT_CHANGED",
    )
    assert remote.mutations == []
    assert db.get_action_proposal(proposal.id)["status"] == "INVALIDATED"


def test_persisted_fingerprint_tamper_invalidates(tmp_path):
    db, coordinator, remote = build(tmp_path)
    proposal = coordinator.propose_operation(plan(), "7")
    with db.connect() as connection:
        connection.execute(
            "UPDATE action_proposals SET preflight_fingerprint='tampered' WHERE id=?",
            (proposal.id,),
        )
    assert coordinator.approve(proposal.id, user_id=7, channel_id=9) == (
        False, "PREFLIGHT_CHANGED",
    )
    assert remote.mutations == []
