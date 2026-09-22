from __future__ import annotations

import pytest

from vast_agent.actions.coordinator import ActionCoordinator
from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    GPUParameters,
    PreflightSnapshot,
    RebootParameters,
    ServiceParameters,
    VerificationResult,
)
from vast_agent.actions.policies import PolicyDecision, PolicyEngine
from vast_agent.config import (
    ConfigError,
    HostRegistry,
    OperationsSettings,
    load_operations_settings,
)
from vast_agent.execution.base import Executor
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult
from vast_agent.storage.database import Database

SAFE_PREFLIGHT = PreflightSnapshot(
    ssh_reachable=True,
    sudo_available=True,
    target_exists=True,
    evidence_complete=True,
    gpu_mapping_resolved=True,
    gpu_binding="nvidia",
    filesystem_healthy=True,
)


def _request(action_type: ActionType) -> ActionRequest:
    if action_type == ActionType.GPU_RESET:
        parameters = GPUParameters(gpu_index=0)
    elif action_type == ActionType.HOST_REBOOT:
        parameters = RebootParameters(assessment="HOST_REBOOT_CANDIDATE")
    else:
        service = {
            ActionType.RESTART_VAST_SERVICE: "vastai.service",
            ActionType.RESTART_DOCKER_SERVICE: "docker.service",
        }[action_type]
        parameters = ServiceParameters(service=service)
    return ActionRequest(host="torrent", action_type=action_type, parameters=parameters)


@pytest.mark.parametrize("action_type", list(ActionType))
def test_operations_are_disabled_and_empty_by_default(action_type):
    settings = OperationsSettings()
    result = PolicyEngine().evaluate(
        _request_for_any_action(action_type), SAFE_PREFLIGHT, settings, execution=True
    )
    assert settings.enabled is False
    assert settings.allowed_actions == []
    assert result.decision == PolicyDecision.BLOCK
    assert "OPERATIONS_DISABLED" in result.reasons


def _request_for_any_action(action_type: ActionType) -> ActionRequest:
    from vast_agent.actions.models import (
        ContainerParameters,
        PackageInstallParameters,
        VMParameters,
    )

    parameters = {
        ActionType.RESTART_VAST_SERVICE: ServiceParameters(service="vastai.service"),
        ActionType.RESTART_DOCKER_SERVICE: ServiceParameters(service="docker.service"),
        ActionType.RESTART_LIBVIRT_SERVICE: ServiceParameters(service="libvirtd.service"),
        ActionType.RESTART_VAST_CONTAINER: ContainerParameters(container="C.1"),
        ActionType.GPU_RESET: GPUParameters(gpu_index=0),
        ActionType.VM_MODE_ENABLE: VMParameters(mode="on"),
        ActionType.VM_MODE_DISABLE: VMParameters(mode="off"),
        ActionType.HOST_REBOOT: RebootParameters(assessment="HOST_REBOOT_CANDIDATE"),
        ActionType.PACKAGE_INSTALL: PackageInstallParameters(
            package_name="vnstat", expected_capability="traffic_history",
        ),
    }[action_type]
    return ActionRequest(host="torrent", action_type=action_type, parameters=parameters)


@pytest.mark.parametrize("action_type", list(ActionType))
def test_enabled_with_empty_allowlist_blocks_every_action(action_type):
    result = PolicyEngine().evaluate(
        _request_for_any_action(action_type),
        SAFE_PREFLIGHT,
        OperationsSettings(enabled=True),
        execution=True,
    )
    assert result.decision == PolicyDecision.BLOCK
    assert "ACTION_NOT_ALLOWED" in result.reasons


@pytest.mark.parametrize(
    ("action_type", "expected"),
    [
        (ActionType.RESTART_VAST_SERVICE, PolicyDecision.ALLOW),
        (ActionType.RESTART_DOCKER_SERVICE, PolicyDecision.BLOCK),
        (ActionType.GPU_RESET, PolicyDecision.BLOCK),
        (ActionType.HOST_REBOOT, PolicyDecision.BLOCK),
    ],
)
def test_phase_one_allowlist_only_allows_vast_restart(action_type, expected):
    settings = OperationsSettings(
        enabled=True,
        allowed_actions=[ActionType.RESTART_VAST_SERVICE],
    )
    result = PolicyEngine().evaluate(
        _request(action_type),
        SAFE_PREFLIGHT,
        settings,
        execution=True,
    )
    assert result.decision == expected
    if expected == PolicyDecision.BLOCK:
        assert "ACTION_NOT_ALLOWED" in result.reasons


def test_unknown_allowed_action_is_a_config_error(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "operations:\n  enabled: true\n  allowed_actions: [RUN_ARBITRARY_SHELL]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Invalid operations configuration"):
        load_operations_settings(path)


class ChangingPreflight:
    def __init__(self):
        self.calls = 0

    def collect(self, host, request):
        self.calls += 1
        return SAFE_PREFLIGHT


class RecordingExecutor(Executor):
    def __init__(self):
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        self.calls.append(tuple(command))
        return ToolResult(success=True, exit_code=0, duration_ms=1)


class SuccessfulVerifier:
    def verify(self, host, request):
        return VerificationResult(success=True, status="VERIFIED", summary="checked")


def test_fresh_preflight_rechecks_allowlist_after_proposal(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    database.migrate()
    host = Host(name="torrent", address="192.0.2.1", ssh_user="agent")
    settings = OperationsSettings(
        enabled=True,
        allowed_actions=[ActionType.RESTART_VAST_SERVICE],
    )
    preflight = ChangingPreflight()
    remote = RecordingExecutor()
    coordinator = ActionCoordinator(
        database,
        HostRegistry(hosts={host.name: host}),
        settings,
        preflight,
        TypedActionExecutor(remote, settings),
        SuccessfulVerifier(),
        owner_id=7,
        channel_id=9,
    )

    proposal = coordinator.propose(_request(ActionType.RESTART_VAST_SERVICE), "7")
    assert proposal.status == "PENDING"
    settings.allowed_actions.clear()

    assert coordinator.approve(proposal.id, user_id=7, channel_id=9) == (
        False,
        "PREFLIGHT_BLOCKED",
    )
    assert preflight.calls == 2
    assert remote.calls == []
    row = database.get_action_proposal(proposal.id)
    assert row["status"] == "INVALIDATED"
    assert row["invalidated_reason"] == "ACTION_NOT_ALLOWED"
