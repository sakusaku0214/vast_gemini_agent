from __future__ import annotations

from vast_agent.actions.models import ActionRequest, ActionType, RiskClass, ServiceParameters


class ActionRegistry:
    """Closed typed registry. It deliberately has no arbitrary-command entry point."""

    _risks = {
        ActionType.RESTART_VAST_SERVICE: RiskClass.WRITE,
        ActionType.RESTART_DOCKER_SERVICE: RiskClass.DANGEROUS,
        ActionType.RESTART_LIBVIRT_SERVICE: RiskClass.DANGEROUS,
        ActionType.RESTART_VAST_CONTAINER: RiskClass.WRITE,
        ActionType.GPU_RESET: RiskClass.DANGEROUS,
        ActionType.VM_MODE_ENABLE: RiskClass.DANGEROUS,
        ActionType.VM_MODE_DISABLE: RiskClass.DANGEROUS,
        ActionType.HOST_REBOOT: RiskClass.REBOOT,
    }

    def risk(self, request: ActionRequest) -> RiskClass:
        return self._risks[request.action_type]

    def validate_consistency(self, request: ActionRequest) -> None:
        expected = {
            ActionType.RESTART_VAST_SERVICE: "vastai.service",
            ActionType.RESTART_DOCKER_SERVICE: "docker.service",
            ActionType.RESTART_LIBVIRT_SERVICE: "libvirtd.service",
        }
        service = expected.get(request.action_type)
        if service and (not isinstance(request.parameters, ServiceParameters) or request.parameters.service != service):
            raise ValueError("action type and service target do not match")
