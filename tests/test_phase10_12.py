from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from vast_agent.actions.coordinator import ActionCoordinator
from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    PreflightSnapshot,
    ServiceParameters,
    VerificationResult,
    VMParameters,
)
from vast_agent.actions.reboot import RebootCoordinator, RebootResult
from vast_agent.config import HostRegistry, OperationsSettings
from vast_agent.execution.base import Executor
from vast_agent.jobs.locks import HostLockManager, LockClass
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult
from vast_agent.storage.database import Database


class FakePreflight:
    def __init__(self, state): self.state = state
    def collect(self, host, request): return self.state


class RecordingExecutor(Executor):
    def __init__(self): self.calls = []
    def execute(self, host, command, timeout, cancellation=None):
        self.calls.append(tuple(command)); return ToolResult(success=True, exit_code=0, duration_ms=1)


class Verifier:
    def __init__(self, success=True): self.success = success
    def verify(self, host, request):
        return VerificationResult(success=self.success, status="VERIFIED" if self.success else "FAILED", summary="checked")


def setup(tmp_path, *, enabled=True, state=None, verifier=True):
    db = Database(tmp_path / "db.sqlite"); db.migrate()
    host = Host(name="torrent", address="192.0.2.1", ssh_user="agent")
    hosts = HostRegistry(hosts={host.name: host})
    settings = OperationsSettings(enabled=enabled)
    state = state or PreflightSnapshot(ssh_reachable=True, sudo_available=True, target_exists=True)
    remote = RecordingExecutor()
    coordinator = ActionCoordinator(db, hosts, settings, FakePreflight(state),
        TypedActionExecutor(remote, settings), Verifier(verifier), 7, 9)
    return db, coordinator, remote


def service_request():
    return ActionRequest(host="torrent", action_type=ActionType.RESTART_VAST_SERVICE,
                         parameters=ServiceParameters(service="vastai.service"))


@pytest.mark.parametrize("kwargs", [
    {"user_id": 8, "channel_id": 9}, {"user_id": 7, "channel_id": 10},
    {"user_id": 7, "channel_id": 9, "is_bot": True},
])
def test_approval_guard_executes_zero(tmp_path, kwargs):
    _, coordinator, remote = setup(tmp_path)
    proposal = coordinator.propose(service_request(), "7")
    assert coordinator.approve(proposal.id, **kwargs)[0] is False
    assert remote.calls == []


def test_double_and_concurrent_approval_exactly_once(tmp_path):
    _, coordinator, remote = setup(tmp_path)
    proposal = coordinator.propose(service_request(), "7")
    threads = [threading.Thread(target=coordinator.approve,
               kwargs={"proposal_id": proposal.id, "user_id": 7, "channel_id": 9}) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert remote.calls == [("sudo", "-n", "systemctl", "restart", "vastai.service")]


def test_expired_disabled_changed_and_verify_fail_never_retry(tmp_path):
    _, coordinator, remote = setup(tmp_path)
    old = datetime.now(UTC) - timedelta(hours=1)
    proposal = coordinator.propose(service_request(), "7", old)
    assert not coordinator.approve(proposal.id, user_id=7, channel_id=9)[0]
    assert not remote.calls
    _, disabled, remote2 = setup(tmp_path / "two", enabled=False)
    p2 = disabled.propose(service_request(), "7")
    assert not disabled.approve(p2.id, user_id=7, channel_id=9)[0]
    assert not remote2.calls
    db, failed, remote3 = setup(tmp_path / "three", verifier=False)
    p3 = failed.propose(service_request(), "7")
    assert not failed.approve(p3.id, user_id=7, channel_id=9)[0]
    assert len(remote3.calls) == 1
    assert db.get_action_proposal(p3.id)["status"] == "FAILED_VERIFY"


def test_typed_parameter_injection_rejected():
    with pytest.raises(ValidationError): ContainerParameters(container="C.123;reboot")
    with pytest.raises(ValidationError): GPUParameters(gpu_index="0; reboot")
    with pytest.raises(ValidationError): ServiceParameters(service="evil.service")
    with pytest.raises(ValidationError): OperationsSettings(enable_vms_script='"; reboot')


def test_gpu_and_vm_policy_blocks(tmp_path):
    unsafe = PreflightSnapshot(ssh_reachable=True, sudo_available=True, target_exists=True,
                               gpu_binding="vfio", gpu_processes=True)
    _, c, remote = setup(tmp_path, state=unsafe)
    request = ActionRequest(host="torrent", action_type=ActionType.GPU_RESET,
                            parameters=GPUParameters(gpu_index=0))
    assert c.propose(request, "7").status == "BLOCKED"
    assert not remote.calls
    request = ActionRequest(host="torrent", action_type=ActionType.VM_MODE_ENABLE,
                            parameters=VMParameters(mode="on"))
    assert c.propose(request, "7").status == "BLOCKED"


class RebootRemote:
    def __init__(self, sequence, post=True): self.sequence, self.calls, self.post = iter(sequence), 0, post
    def dispatch(self, host): self.calls += 1; return ToolResult(success=False, duration_ms=1)
    def reachable(self, host): return next(self.sequence)
    def postcheck(self, host): return self.post


@pytest.mark.parametrize(("sequence", "expected"), [
    ([True, False, True], RebootResult.SUCCEEDED),
    ([True, True, True], RebootResult.REBOOT_NOT_OBSERVED),
    ([False, False, False], RebootResult.REBOOT_RECOVERY_FAILED),
])
def test_reboot_state_machine(sequence, expected):
    now = [0.0]
    remote = RebootRemote(sequence)
    monitor = RebootCoordinator(remote, timeout=3, poll_seconds=1,
        clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert monitor.execute_approved(Host(name="x", address="192.0.2.2", ssh_user="a")).result == expected
    assert remote.calls == 1


def test_lock_write_excludes_read():
    locks, entered, release, order = HostLockManager(), threading.Event(), threading.Event(), []
    def writer():
        with locks.acquire("h", LockClass.WRITE): entered.set(); release.wait(); order.append("write")
    def reader():
        entered.wait()
        with locks.acquire("h", LockClass.READ): order.append("read")
    a, b = threading.Thread(target=writer), threading.Thread(target=reader)
    a.start(); b.start(); entered.wait(); assert order == []; release.set(); a.join(); b.join()
    assert order == ["write", "read"]
