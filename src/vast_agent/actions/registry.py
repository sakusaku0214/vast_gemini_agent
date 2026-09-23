from __future__ import annotations

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    PackageInstallParameters,
    RebootParameters,
    RiskClass,
    ServiceParameters,
    VMParameters,
)
from vast_agent.actions.package_catalog import package_definition


class ActionRegistry:
    """Closed typed registry. It deliberately has no arbitrary-command entry point."""

    _risks = {
        ActionType.PACKAGE_INSTALL: RiskClass.DANGEROUS,
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
        valid = False
        if service:
            valid = isinstance(request.parameters, ServiceParameters) and request.parameters.service == service
        elif request.action_type == ActionType.RESTART_VAST_CONTAINER:
            valid = isinstance(request.parameters, ContainerParameters)
        elif request.action_type == ActionType.GPU_RESET:
            valid = isinstance(request.parameters, GPUParameters)
        elif request.action_type == ActionType.VM_MODE_ENABLE:
            valid = isinstance(request.parameters, VMParameters) and request.parameters.mode == "on"
        elif request.action_type == ActionType.VM_MODE_DISABLE:
            valid = isinstance(request.parameters, VMParameters) and request.parameters.mode == "off"
        elif request.action_type == ActionType.HOST_REBOOT:
            valid = isinstance(request.parameters, RebootParameters)
        elif request.action_type == ActionType.PACKAGE_INSTALL:
            valid = isinstance(request.parameters, PackageInstallParameters)
            if valid and request.parameters.expected_capability is not None:
                definition = package_definition(request.parameters.package_name)
                valid = (definition is not None and
                         request.parameters.expected_capability == definition.capability_id)
        if not valid:
            raise ValueError("action type and parameters do not match")
