"""Code-owned preflight, execution, and verification for approved OperationPlans."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vast_agent.actions.operation_plan import OperationPlan
from vast_agent.execution.base import Executor
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult


class GenericPreflightSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host_enabled: bool
    ssh_reachable: bool
    sudo_available: bool
    active_workload: bool
    running_vm: bool


class GenericOperationRuntime:
    """Own every command used after planning; model-provided verification argv is forbidden."""

    def __init__(self, remote: Executor, timeout: int) -> None:
        self.remote = remote
        self.timeout = timeout

    def preflight(self, host: Host, plan: OperationPlan) -> GenericPreflightSnapshot:
        reachable = self.remote.execute(host, ("true",), min(self.timeout, 15))
        sudo = (self.remote.execute(host, ("sudo", "-n", "true"), min(self.timeout, 15))
                if plan.requires_sudo else reachable)
        workload = self.remote.execute(
            host, ("pgrep", "-f", "SRBMiner-MULTI|vastai|docker"), min(self.timeout, 15),
        )
        vms = self.remote.execute(
            host, ("virsh", "list", "--state-running", "--name"), min(self.timeout, 15),
        )
        return GenericPreflightSnapshot(
            host_enabled=host.enabled, ssh_reachable=reachable.success,
            sudo_available=sudo.success, active_workload=workload.success and bool(workload.stdout.strip()),
            running_vm=vms.success and bool(vms.stdout.strip()),
        )

    def execute(self, host: Host, plan: OperationPlan) -> ToolResult:
        return self.remote.execute(host, plan.execution_argv(), self.timeout)

    def verify(self, host: Host, plan: OperationPlan) -> tuple[bool | None, str]:
        kind, target = plan.verification_kind, plan.verification_target
        if kind == "unavailable":
            return None, "verification unavailable"
        if kind == "service_state":
            result = self.remote.execute(host, ("systemctl", "is-active", "--", str(target)), 15)
            expected = "inactive" if plan.argv[0] == "stop" else "active"
            actual = result.stdout.strip()
            return result.success and actual == expected, f"service state: {actual or 'unavailable'}"
        if kind == "gpu_state":
            result = self.remote.execute(
                host,
                ("nvidia-smi", "-i", str(target),
                 "--query-gpu=index,clocks.gr,power.draw,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"),
                15,
            )
            return result.success and bool(result.stdout.strip()), (
                "GPU state re-read" if result.success else "GPU verification unavailable"
            )
        # Executable verification uses a code-owned resolver, not a model argv.
        result = self.remote.execute(host, ("which", "--", str(target)), 15)
        return result.success and bool(result.stdout.strip()), (
            "executable resolved" if result.success else "executable verification unavailable"
        )
