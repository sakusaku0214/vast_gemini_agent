from __future__ import annotations

import asyncio
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
    state = state or PreflightSnapshot(ssh_reachable=True, sudo_available=True, target_exists=True, evidence_complete=True)
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
    unsafe = PreflightSnapshot(ssh_reachable=True, sudo_available=True, target_exists=True, evidence_complete=True,
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

@pytest.mark.parametrize(("action_type", "parameters"), [
    (ActionType.RESTART_VAST_SERVICE, GPUParameters(gpu_index=0)),
    (ActionType.RESTART_DOCKER_SERVICE, ContainerParameters(container="C.1")),
    (ActionType.RESTART_LIBVIRT_SERVICE, VMParameters(mode="off")),
    (ActionType.RESTART_VAST_CONTAINER, GPUParameters(gpu_index=0)),
    (ActionType.GPU_RESET, ContainerParameters(container="C.1")),
    (ActionType.VM_MODE_ENABLE, VMParameters(mode="off")),
    (ActionType.VM_MODE_DISABLE, VMParameters(mode="on")),
    (ActionType.HOST_REBOOT, ServiceParameters(service="vastai.service")),
])
def test_every_action_parameter_mismatch_rejected_before_proposal(tmp_path, action_type, parameters):
    _, coordinator, remote = setup(tmp_path)
    request = ActionRequest(host="torrent", action_type=action_type, parameters=parameters)
    with pytest.raises(ValueError, match="do not match"):
        coordinator.propose(request, "7")
    with pytest.raises(ValueError, match="do not match"):
        coordinator.executor.execute(coordinator.hosts.resolve("torrent"), request)
    assert remote.calls == []


def test_preflight_change_invalidates_without_write(tmp_path):
    db, coordinator, remote = setup(tmp_path)
    proposal = coordinator.propose(service_request(), "7")
    coordinator.preflight.state = coordinator.preflight.state.model_copy(update={"current_state": "changed"})
    assert coordinator.approve(proposal.id, user_id=7, channel_id=9) == (False, "PREFLIGHT_CHANGED")
    assert remote.calls == []
    assert db.get_action_proposal(proposal.id)["status"] == "INVALIDATED"


def test_incomplete_evidence_is_blocked(tmp_path):
    state = PreflightSnapshot(ssh_reachable=True, sudo_available=True, target_exists=True,
                              evidence_complete=False)
    _, coordinator, remote = setup(tmp_path, state=state)
    proposal = coordinator.propose(service_request(), "7")
    assert proposal.status == "BLOCKED"
    assert remote.calls == []


def test_reboot_approval_uses_dedicated_pipeline_once(tmp_path):
    from vast_agent.actions.reboot import RebootCoordinator
    db, coordinator, _ = setup(tmp_path)
    now = [0.0]
    reboot_remote = RebootRemote([True, False, True])
    coordinator.reboot = RebootCoordinator(reboot_remote, timeout=3, poll_seconds=1,
        clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    request = ActionRequest(host="torrent", action_type=ActionType.HOST_REBOOT,
        parameters={"kind": "reboot", "assessment": "HOST_REBOOT_CANDIDATE"})
    proposal = coordinator.propose(request, "7")
    assert coordinator.approve(proposal.id, user_id=7, channel_id=9)[0]
    assert reboot_remote.calls == 1
    assert db.get_action_proposal(proposal.id)["status"] == "SUCCEEDED"


def test_reboot_definite_dispatch_failure_does_not_monitor():
    from vast_agent.actions.reboot import RebootCoordinator
    from vast_agent.models.tool_result import ErrorCode
    class Denied(RebootRemote):
        def dispatch(self, host):
            self.calls += 1
            return ToolResult(success=False, duration_ms=1, stderr="sudo: a password is required",
                              error_code=ErrorCode.COMMAND_FAILED)
        def reachable(self, host): raise AssertionError("must not monitor")
    remote = Denied([])
    result = RebootCoordinator(remote).execute_approved(
        Host(name="x", address="192.0.2.2", ssh_user="a"))
    assert result.result == RebootResult.DISPATCH_FAILED
    assert remote.calls == 1

def test_agent_service_write_request_returns_proposal_without_write(tmp_path):
    from vast_agent.conversation.state import ConversationStore
    from vast_agent.jobs.manager import JobManager
    from vast_agent.services.agent_service import AgentService
    db, coordinator, remote = setup(tmp_path)
    service = AgentService(coordinator.hosts, object(), remote, JobManager(db),
                           ConversationStore(db), actions=coordinator)
    reply = asyncio.run(service.handle_question("torrentのvastai再起動して", 7, 9))
    assert reply.proposal is not None
    assert "Proposal #" in reply.text
    assert "RESTART_VAST_SERVICE" in reply.text
    assert remote.calls == []


def test_agent_service_fleet_write_creates_no_proposal(tmp_path):
    from vast_agent.conversation.state import ConversationStore
    from vast_agent.jobs.manager import JobManager
    from vast_agent.services.agent_service import AgentService
    db, coordinator, remote = setup(tmp_path)
    service = AgentService(coordinator.hosts, object(), remote, JobManager(db),
                           ConversationStore(db), actions=coordinator)
    reply = asyncio.run(service.handle_question("全台再起動して", 7, 9))
    assert reply.proposal is None
    assert "fleet WRITE" in reply.text
    assert db.pending_approval_count() == 0
    assert remote.calls == []


@pytest.mark.parametrize("recommendation", ["CONTINUE_OBSERVING", "PHYSICAL_CHECK_REQUIRED"])
def test_reboot_non_candidate_creates_no_proposal(tmp_path, recommendation):
    from types import SimpleNamespace

    from vast_agent.conversation.state import ConversationStore
    from vast_agent.jobs.manager import JobManager
    from vast_agent.services.agent_service import AgentService
    class Agent:
        def investigate(self, host, text):
            return SimpleNamespace(summary="safe", signatures=[], recommended_action=recommendation,
                                   confidence="high")
    db, coordinator, remote = setup(tmp_path)
    service = AgentService(coordinator.hosts, object(), remote, JobManager(db),
                           ConversationStore(db), agent=Agent(), actions=coordinator)
    reply = asyncio.run(service.handle_question("torrent再起動して", 7, 9))
    assert reply.proposal is None
    assert db.pending_approval_count() == 0
    assert remote.calls == []


def test_reboot_gemini_unavailable_creates_no_proposal(tmp_path):
    from vast_agent.conversation.state import ConversationStore
    from vast_agent.jobs.manager import JobManager
    from vast_agent.services.agent_service import AgentService
    db, coordinator, remote = setup(tmp_path)
    service = AgentService(coordinator.hosts, object(), remote, JobManager(db),
                           ConversationStore(db), actions=coordinator)
    reply = asyncio.run(service.handle_question("torrent再起動して", 7, 9))
    assert reply.text == "REBOOT_ASSESSMENT_UNAVAILABLE"
    assert reply.proposal is None and not remote.calls


def test_action_restart_reconciliation_never_replays(tmp_path):
    db, coordinator, remote = setup(tmp_path)
    pending = coordinator.propose(service_request(), "7")
    assert db.reconcile_actions() == 1
    assert db.get_action_proposal(pending.id)["status"] == "INTERRUPTED"
    assert coordinator.approve(pending.id, user_id=7, channel_id=9)[0] is False
    assert remote.calls == []


def test_action_preflight_and_verification_are_redacted_in_database(tmp_path):
    secret = "super-secret-token"
    db = Database(tmp_path / "db.sqlite"); db.migrate()
    host = Host(name="torrent", address="192.0.2.1", ssh_user="agent")
    hosts = HostRegistry(hosts={host.name: host})
    settings = OperationsSettings(enabled=True)
    state = PreflightSnapshot(ssh_reachable=True, sudo_available=True, target_exists=True,
                              evidence_complete=True, details={"diagnostic": secret})
    remote = RecordingExecutor()
    class SecretVerifier:
        def verify(self, host, request):
            return VerificationResult(success=True, status="VERIFIED", summary=secret)
    coordinator = ActionCoordinator(db, hosts, settings, FakePreflight(state),
        TypedActionExecutor(remote, settings), SecretVerifier(), 7, 9, secrets=(secret,))
    proposal = coordinator.propose(service_request(), "7")
    assert secret not in str(db.get_action_proposal(proposal.id))
    assert coordinator.approve(proposal.id, user_id=7, channel_id=9)[0]
    with db.connect() as connection:
        run = connection.execute("SELECT verify_summary FROM action_runs").fetchone()[0]
    assert secret not in run and "[REDACTED]" in run


def test_required_natural_language_actions_and_advisories():
    from vast_agent.actions.resolver import ActionIntentResolver
    taichi = Host(name="taichi", aliases=["Taichi"], address="192.0.2.3", ssh_user="a")
    torrent = Host(name="torrent", address="192.0.2.4", ssh_user="a")
    resolver = ActionIntentResolver(HostRegistry(hosts={"taichi": taichi, "torrent": torrent}))
    cases = {
        "torrentのvastai再起動して": ActionType.RESTART_VAST_SERVICE,
        "C.51729761再起動して": ActionType.RESTART_VAST_CONTAINER,
        "TaichiのGPU0 resetして": ActionType.GPU_RESET,
        "TaichiのVMをonにして": ActionType.VM_MODE_ENABLE,
        "TaichiのVMをoffにして": ActionType.VM_MODE_DISABLE,
        "torrent再起動して": ActionType.HOST_REBOOT,
    }
    for text, expected in cases.items():
        request = resolver.resolve(text, "torrent")
        assert request is not None and request.action_type == expected
    assert resolver.resolve("torrentはrebootすべき？") is None
    assert resolver.resolve("GPU reset必要？") is None
    assert resolver.resolve("GPU resetして") is None
    assert resolver.resolve("全台再起動して") is None


def test_production_service_shares_job_and_write_lock(tmp_path):
    from vast_agent.app import _service
    from vast_agent.config import initialize_config
    from vast_agent.paths import RuntimePaths
    paths = RuntimePaths(tmp_path / "runtime")
    initialize_config(paths)
    host = Host(name="torrent", address="192.0.2.4", ssh_user="a")
    service = _service(paths, HostRegistry(hosts={"torrent": host}), RecordingExecutor(), 7, 9)
    assert service.actions is not None
    assert service.actions.locks is service.jobs.locks


def test_unique_followup_creates_proposal_but_never_immediate_write(tmp_path):
    from types import SimpleNamespace

    from vast_agent.conversation.state import ConversationStore
    from vast_agent.jobs.manager import JobManager
    from vast_agent.services.agent_service import AgentService
    class Agent:
        def investigate(self, host, text):
            return SimpleNamespace(summary="candidate", signatures=[],
                recommended_action="HOST_REBOOT_CANDIDATE", confidence="high")
    db, coordinator, remote = setup(tmp_path)
    service = AgentService(coordinator.hosts, object(), remote, JobManager(db),
                           ConversationStore(db), agent=Agent(), actions=coordinator)
    advisory = asyncio.run(service.handle_question("torrentはrebootすべき？", 7, 9))
    assert advisory.proposal is None
    followup = asyncio.run(service.handle_question("それやって", 7, 9))
    assert followup.proposal is not None
    assert followup.proposal.action_type == ActionType.HOST_REBOOT
    assert remote.calls == []


def test_pci_bdf_normalization_accepts_domain_width_variants():
    from vast_agent.actions.preflight import normalize_pci_bdf
    expected = "01:00.0"
    assert normalize_pci_bdf("00000000:01:00.0") == expected
    assert normalize_pci_bdf("0000:01:00.0") == expected
    assert normalize_pci_bdf("01:00.0") == expected
    assert normalize_pci_bdf("not-a-bdf") is None


def test_reboot_postcheck_requires_actual_fault_recovery():
    before = PreflightSnapshot(
        ssh_reachable=True, target_exists=True, evidence_complete=True,
        signatures=["NVIDIA_FALLEN_OFF_BUS", "NVML_UNAVAILABLE"], nvml_ok=False,
        gpu_present=False, pci_nvidia=0, pci_unbound=1, failed_units=0,
        vast_state="active",
    )
    unchanged = before.model_copy(update={"ssh_reachable": True})
    recovered = before.model_copy(update={
        "signatures": [], "nvml_ok": True, "gpu_present": True,
        "pci_nvidia": 1, "pci_unbound": 0,
    })
    assert not RebootCoordinator._recovered(before, unchanged)
    assert RebootCoordinator._recovered(before, recovered)
    assert not RebootCoordinator._recovered(
        before, recovered.model_copy(update={"evidence_complete": False}))


def test_vm_verify_requires_unambiguous_pci_ownership():
    from vast_agent.actions.verifier import ProductionActionVerifier
    request_on = ActionRequest(host="torrent", action_type=ActionType.VM_MODE_ENABLE,
                               parameters=VMParameters(mode="on"))
    safe = PreflightSnapshot(ssh_reachable=True, target_exists=True, evidence_complete=True,
        gpu_mapping_resolved=True, gpu_binding="vfio", pci_vfio=1, pci_nvidia=0)
    provider = FakePreflight(safe)
    verifier = ProductionActionVerifier(provider)
    assert verifier.verify(Host(name="torrent", address="192.0.2.1", ssh_user="a"), request_on).success
    provider.state = safe.model_copy(update={"gpu_binding": "unknown", "pci_unbound": 1})
    assert not verifier.verify(Host(name="torrent", address="192.0.2.1", ssh_user="a"), request_on).success


def test_slow_approval_is_acknowledged_before_action_finishes():
    from types import SimpleNamespace

    from vast_agent.discord_app.approval_logic import handle_approval
    started, release, events = threading.Event(), threading.Event(), []
    class Response:
        async def defer(self): events.append("defer")
        async def send_message(self, content, ephemeral=False): events.append("reject")
    class Interaction:
        user = SimpleNamespace(id=7, bot=False)
        channel_id = 9
        response = Response()
        async def edit_original_response(self, **kwargs): events.append(kwargs["content"])
    class SlowCoordinator:
        owner_id, channel_id = 7, 9
        def approve(self, proposal_id, **kwargs):
            started.set(); release.wait(); events.append("complete")
            return True, "SUCCEEDED"
    async def scenario():
        task = asyncio.create_task(handle_approval(Interaction(), SlowCoordinator(), 41,
                                                   lambda: events.append("disable")))
        await asyncio.to_thread(started.wait, 1)
        assert events[0] == "defer"
        assert "complete" not in events
        release.set()
        await task
    asyncio.run(scenario())
    assert events.count("complete") == 1
    assert events[-1] == "SUCCEEDED"


def test_action_specific_evidence_ignores_unrelated_unsupported_capability():
    from types import SimpleNamespace

    from vast_agent.actions.preflight import ProductionPreflightProvider
    from vast_agent.models.host import Capabilities
    from vast_agent.models.observation import Observation, SystemSummary
    from vast_agent.models.tool_result import ErrorCode
    host = Host(name="h", address="192.0.2.8", ssh_user="a",
        capabilities=Capabilities(nvidia=True, vast=True, docker=True, libvirt=False))
    def ok(stdout=""):
        return ToolResult(success=True, stdout=stdout, duration_ms=1)
    unsupported = ToolResult(success=False, duration_ms=1, error_code=ErrorCode.TOOL_UNSUPPORTED)
    results = {
        "host_ping": ok(), "get_vast_status": ok("ActiveState=active"),
        "get_service_status": ok("Id=vastai.service\nActiveState=active"),
        "get_system_health": ok("/ 10%"), "get_vm_status": unsupported,
    }
    observation = Observation(host="h", ssh_ok=True,
        system=SystemSummary(filesystem_max_percent=10), services={"vastai": "active"})
    class Inspection:
        def inspect_and_record(self, host, executor):
            return SimpleNamespace(observation=observation, tool_results=results)
    class Remote:
        def execute(self, host, command, timeout, cancellation=None): return ok()
    provider = ProductionPreflightProvider(Inspection(), Remote(), OperationsSettings(enabled=True))
    request = ActionRequest(host="h", action_type=ActionType.RESTART_VAST_SERVICE,
                            parameters=ServiceParameters(service="vastai.service"))
    assert provider.collect(host, request).evidence_complete

    libvirt = ActionRequest(host="h", action_type=ActionType.RESTART_LIBVIRT_SERVICE,
                            parameters=ServiceParameters(service="libvirtd.service"))
    state = provider.collect(host, libvirt)
    assert not state.evidence_complete and not state.target_exists


def test_action_specific_gpu_and_reboot_required_failures_are_incomplete():
    from vast_agent.actions.preflight import ProductionPreflightProvider
    from vast_agent.models.host import Capabilities
    host = Host(name="h", address="192.0.2.8", ssh_user="a",
        capabilities=Capabilities(nvidia=True, vast=False, docker=True, libvirt=False))
    gpu = ActionRequest(host="h", action_type=ActionType.GPU_RESET,
                        parameters=GPUParameters(gpu_index=0))
    required = ProductionPreflightProvider._required_tools(host, gpu)
    assert "get_pci_status" in required and "get_gpu_status" in required
    assert "get_vm_status" not in required
    reboot = ActionRequest(host="h", action_type=ActionType.HOST_REBOOT,
        parameters={"kind": "reboot", "assessment": "HOST_REBOOT_CANDIDATE"})
    reboot_required = ProductionPreflightProvider._required_tools(host, reboot)
    assert "get_docker_status" in reboot_required
    assert "get_vast_status" not in reboot_required

    # A required declared-capability tool failure remains incomplete.
    results = {name: ToolResult(success=True, duration_ms=1) for name in reboot_required}
    results["get_docker_status"] = ToolResult(success=False, duration_ms=1)
    assert not all(results[name].success for name in reboot_required)


def test_vm_disable_tolerates_only_all_vfio_gpu_process_failure():
    from types import SimpleNamespace

    from conftest import fixture_results

    from vast_agent.actions.preflight import ProductionPreflightProvider
    from vast_agent.diagnostics.parser import build_observation

    def ok(stdout=""):
        return ToolResult(success=True, stdout=stdout, duration_ms=1)

    def collect(case, action_type, expected_gpu_count=1):
        results = fixture_results(case)
        if case == "mixed_binding":
            results["get_pci_status"] = results["get_pci_status"].model_copy(update={
                "stdout": results["get_pci_status"].stdout + (
                    "\n0000:02:00.0 3D controller: NVIDIA Corporation Example [10de:0003]\n"
                    "\tKernel driver in use: vfio-pci\n"
                ),
            })
        results.update({
            "get_vm_status": ok("shut off"),
            "get_gpu_processes": ToolResult(
                success=False, stderr="No devices were found", duration_ms=1,
            ),
            "get_d_state_processes": ok("S 1 init"),
        })
        observation = build_observation("h", results, expected_gpu_count)

        class Inspection:
            def inspect_and_record(self, host, executor):
                return SimpleNamespace(observation=observation, tool_results=results)

        class Remote:
            def execute(self, host, command, timeout, cancellation=None):
                return ok()

        settings = OperationsSettings(enabled=True, enable_vms_script="/opt/enable_vms.py")
        provider = ProductionPreflightProvider(Inspection(), Remote(), settings)
        request = ActionRequest(
            host="h", action_type=action_type,
            parameters=VMParameters(mode="off" if action_type == ActionType.VM_MODE_DISABLE else "on"),
        )
        host = Host(
            name="h", address="192.0.2.8", ssh_user="a",
            expected_gpu_count=expected_gpu_count,
        )
        return provider.collect(host, request)

    disabled = collect("full_vfio", ActionType.VM_MODE_DISABLE)
    assert disabled.evidence_complete
    assert disabled.gpu_present
    assert disabled.gpu_mapping_resolved and disabled.gpu_binding == "vfio"

    assert not collect(
        "full_vfio", ActionType.VM_MODE_DISABLE, expected_gpu_count=2,
    ).evidence_complete
    assert not collect(
        "full_vfio", ActionType.VM_MODE_DISABLE, expected_gpu_count=None,
    ).evidence_complete

    # The same NVIDIA process evidence failure must remain fatal for enabling VM mode.
    assert not collect("full_vfio", ActionType.VM_MODE_ENABLE).evidence_complete

    # Mixed ownership is never treated as the all-VFIO exception.
    mixed = collect("mixed_binding", ActionType.VM_MODE_DISABLE, expected_gpu_count=2)
    assert not mixed.evidence_complete
    assert mixed.gpu_mapping_resolved and mixed.gpu_binding == "unknown"
    from vast_agent.actions.policies import PolicyDecision, PolicyEngine
    request = ActionRequest(
        host="h", action_type=ActionType.VM_MODE_DISABLE,
        parameters=VMParameters(mode="off"),
    )
    decision = PolicyEngine().evaluate(
        request, mixed.model_copy(update={"evidence_complete": True}),
        OperationsSettings(enabled=True, enable_vms_script="/opt/enable_vms.py"),
    )
    assert decision.decision == PolicyDecision.BLOCK
    assert "VM_GPU_OWNERSHIP_UNSAFE" in decision.reasons


def test_gpu_reset_without_nvml_index_mapping_remains_blocked(tmp_path):
    state = PreflightSnapshot(
        ssh_reachable=True, sudo_available=True, target_exists=True,
        evidence_complete=True, gpu_present=True, gpu_mapping_resolved=False,
        gpu_binding="nvidia",
    )
    _, coordinator, remote = setup(tmp_path, state=state)
    request = ActionRequest(
        host="torrent", action_type=ActionType.GPU_RESET,
        parameters=GPUParameters(gpu_index=0),
    )
    proposal = coordinator.propose(request, "7")
    assert proposal.status == "BLOCKED"
    assert remote.calls == []


def test_operations_remain_disabled_by_default():
    assert OperationsSettings().enabled is False
