from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    ServiceParameters,
    VMParameters,
)
from vast_agent.actions.registry import ActionRegistry
from vast_agent.config import OperationsSettings
from vast_agent.execution.base import Executor
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult


class TypedActionExecutor:
    """Maps validated models to one code-owned argv; has no raw command API."""

    def __init__(self, remote: Executor, settings: OperationsSettings) -> None:
        self._remote = remote
        self._settings = settings

    def execute(self, host: Host, request: ActionRequest) -> ToolResult:
        ActionRegistry().validate_consistency(request)
        if request.action_type == ActionType.HOST_REBOOT:
            raise ValueError("HOST_REBOOT must use RebootCoordinator")
        argv = self.argv(request)
        return self._remote.execute(host, argv, self._settings.action_timeout_seconds)

    def argv(self, request: ActionRequest) -> Sequence[str]:
        ActionRegistry().validate_consistency(request)
        params = request.parameters
        if isinstance(params, ServiceParameters):
            return ("sudo", "-n", "systemctl", "restart", params.service)
        if isinstance(params, ContainerParameters):
            return ("sudo", "-n", "docker", "container", "restart", params.container)
        if isinstance(params, GPUParameters):
            return ("sudo", "-n", "nvidia-smi", "--gpu-reset", "-i", str(params.gpu_index))
        if isinstance(params, VMParameters):
            script = self._settings.enable_vms_script
            if not script: raise ValueError("VM_ACTION_NOT_CONFIGURED")
            tail = (params.mode, "-f") if params.mode == "on" else (params.mode,)
            return ("sudo", "-n", "python3", script, *tail)
        raise ValueError("unsupported typed action")


class ActionVerifier(Protocol):
    def verify(self, host: Host, request: ActionRequest): ...
