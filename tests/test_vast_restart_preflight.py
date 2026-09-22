from __future__ import annotations

from types import SimpleNamespace

import pytest

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ServiceParameters,
)
from vast_agent.actions.policies import PolicyDecision, PolicyEngine
from vast_agent.actions.preflight import (
    ProductionPreflightProvider,
    docker_workload_summary,
)
from vast_agent.config import OperationsSettings
from vast_agent.models.host import Capabilities, Host
from vast_agent.models.observation import Observation, SystemSummary
from vast_agent.models.tool_result import ToolResult

EXITED_FIXTURE = """active
id1|C.51978469|Exited (0) 5 hours ago|vastai/kvm:latest
id2|C.51188718|Exited (143) 26 hours ago|vastai/kvm:latest
id3|C.50063228|Exited (0) 7 days ago|vastai/kvm:latest
"""


@pytest.mark.parametrize("status", ["Created", "Dead"])
def test_inactive_docker_states_are_not_workloads(status):
    summary = docker_workload_summary(f"active\nid|C.1|{status}|image\n")
    assert summary.total_containers == 1
    assert summary.running_containers == 0
    assert summary.unknown_containers == 0
    assert not summary.has_active_workload


def test_observed_exited_containers_are_not_workloads():
    summary = docker_workload_summary(EXITED_FIXTURE)
    assert summary.total_containers == 3
    assert summary.running_containers == 0
    assert summary.unknown_containers == 0
    assert not summary.has_active_workload


@pytest.mark.parametrize("status", [
    "Up 40 seconds",
    "Up 40 seconds (Paused)",
    "Restarting (1) 5 seconds ago",
])
def test_running_docker_states_are_active(status):
    summary = docker_workload_summary(
        f"active\nid1|C.old|Exited (0) 5 hours ago|old\n"
        f"id2|C.51946285|{status}|vastai/linux-desktop\n"
    )
    assert summary.total_containers == 2
    assert summary.running_containers == 1
    assert summary.unknown_containers == 0
    assert summary.has_active_workload


@pytest.mark.parametrize("row", [
    "id|C.1|Mystery|image",
    "id|C.1",
    "id|C.1||image",
])
def test_unknown_or_malformed_docker_row_fails_closed(row):
    summary = docker_workload_summary(f"active\n{row}\n")
    assert summary.unknown_containers == 1
    assert summary.has_active_workload


def test_missing_docker_service_line_fails_closed():
    summary = docker_workload_summary("")
    assert summary.total_containers == 0
    assert summary.unknown_containers == 1
    assert summary.has_active_workload


def _request():
    return ActionRequest(
        host="h",
        action_type=ActionType.RESTART_VAST_SERVICE,
        parameters=ServiceParameters(service="vastai.service"),
    )


def _provider(*, docker=EXITED_FIXTURE, vm=" Id Name State\n", docker_success=True,
              vm_success=True):
    def result(stdout="", success=True):
        return ToolResult(success=success, stdout=stdout, duration_ms=1)

    results = {
        "host_ping": result(),
        "get_vast_status": result("ActiveState=active"),
        "get_service_status": result("Id=vastai.service\nActiveState=active"),
        "get_system_health": result("/ 10%"),
        "get_docker_status": result(docker, docker_success),
        "get_vm_status": result(vm, vm_success),
    }
    observation = Observation(
        host="h", ssh_ok=True,
        system=SystemSummary(filesystem_max_percent=10),
        services={"vastai": "active", "docker": "active"},
    )

    class Inspection:
        def inspect_and_record(self, host, executor):
            return SimpleNamespace(observation=observation, tool_results=results)

    class Remote:
        def execute(self, host, command, timeout, cancellation=None):
            return result()

    return ProductionPreflightProvider(
        Inspection(), Remote(),
        OperationsSettings(enabled=True, allowed_actions=["RESTART_VAST_SERVICE"]),
    )


def _host(*, docker=True, libvirt=True):
    return Host(
        name="h", address="192.0.2.1", ssh_user="agent",
        capabilities=Capabilities(vast=True, docker=docker, libvirt=libvirt),
    )


def test_vast_restart_collects_distinct_docker_counts_and_clean_host_is_allowed():
    state = _provider().collect(_host(), _request())
    assert state.evidence_complete
    assert not state.active_workload
    assert not state.running_vm
    assert state.details["docker_total_containers"] == "3"
    assert state.details["docker_running_containers"] == "0"
    assert state.details["docker_unknown_containers"] == "0"
    decision = PolicyEngine().evaluate(
        _request(), state,
        OperationsSettings(enabled=True, allowed_actions=["RESTART_VAST_SERVICE"]),
    )
    assert decision.decision == PolicyDecision.ALLOW


@pytest.mark.parametrize(("failure", "capability"), [
    ("docker", "get_docker_status"),
    ("vm", "get_vm_status"),
])
def test_declared_capability_failure_makes_vast_evidence_incomplete(failure, capability):
    provider = _provider(
        docker_success=failure != "docker", vm_success=failure != "vm",
    )
    state = provider.collect(_host(), _request())
    assert not state.evidence_complete
    assert capability in state.details["required_tools"]
    decision = PolicyEngine().evaluate(
        _request(), state,
        OperationsSettings(enabled=True, allowed_actions=["RESTART_VAST_SERVICE"]),
    )
    assert decision.decision == PolicyDecision.BLOCK
    assert "PREFLIGHT_INCOMPLETE" in decision.reasons


@pytest.mark.parametrize(("docker", "vm", "reason"), [
    ("active\nid|C.1|Up 40 seconds|image\n", "", "VAST_RESTART_ACTIVE_WORKLOAD"),
    (EXITED_FIXTURE, " 1 guest running\n", "VAST_RESTART_RUNNING_VM"),
    ("active\nid|C.1|unknown|image\n", "", "VAST_RESTART_ACTIVE_WORKLOAD"),
])
def test_vast_restart_blocks_unsafe_workloads(docker, vm, reason):
    state = _provider(docker=docker, vm=vm).collect(_host(), _request())
    decision = PolicyEngine().evaluate(
        _request(), state,
        OperationsSettings(enabled=True, allowed_actions=["RESTART_VAST_SERVICE"]),
    )
    assert decision.decision == PolicyDecision.BLOCK
    assert reason in decision.reasons


def test_vast_required_evidence_respects_host_capabilities():
    required = ProductionPreflightProvider._required_tools(
        _host(docker=False, libvirt=False), _request(),
    )
    assert "get_docker_status" not in required
    assert "get_vm_status" not in required
