from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vast_agent.actions.capability_backends import CapabilityBackendResolver
from vast_agent.actions.package_catalog import ServiceRequirement, capability_definition


class BackendStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class SoftwareStatus(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ServiceStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNAVAILABLE = "unavailable"
    NOT_REQUIRED = "not_required"


class OverallStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    MISSING = "missing"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CapabilityState:
    capability_id: str
    backend_status: BackendStatus
    software_status: SoftwareStatus
    service_status: ServiceStatus
    overall_status: OverallStatus
    acquisition_possible: bool
    reason: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class SoftwareAvailabilityChecker:
    """Normalize results from the allowlisted package/executable/service READ tools."""

    resolver: CapabilityBackendResolver = CapabilityBackendResolver()

    def assess(
        self,
        capability_id: str,
        *,
        package_installed: bool | None,
        executable_status: dict[str, bool | None],
        service_active: bool | None = None,
        service_installed: bool | None = None,
        host_capabilities: dict[str, bool] | None = None,
    ) -> CapabilityState:
        definition = capability_definition(capability_id)
        if definition is None:
            return capability_state(capability_id, resolver=self.resolver)
        acquisition = definition.acquisition
        if acquisition is None:
            software = SoftwareStatus.AVAILABLE
        else:
            values = [package_installed]
            values.extend(executable_status.get(name) for name in acquisition.executables)
            software = (SoftwareStatus.UNKNOWN if any(value is None for value in values)
                        else SoftwareStatus.MISSING if not all(values)
                        else SoftwareStatus.AVAILABLE)
        requirement = acquisition.service_requirement if acquisition else ServiceRequirement.NONE
        if requirement is ServiceRequirement.NONE:
            service = ServiceStatus.NOT_REQUIRED
        elif service_installed is not True:
            service = ServiceStatus.UNAVAILABLE
        elif service_active:
            service = ServiceStatus.ACTIVE
        else:
            service = ServiceStatus.INACTIVE
        return capability_state(capability_id, software, service,
                                host_capabilities=host_capabilities, resolver=self.resolver)


def capability_state(
    capability_id: str,
    software_status: SoftwareStatus | str = SoftwareStatus.UNKNOWN,
    service_status: ServiceStatus | str | None = None,
    *,
    host_capabilities: dict[str, bool] | None = None,
    evidence: tuple[str, ...] = (),
    resolver: CapabilityBackendResolver | None = None,
) -> CapabilityState:
    """Derive state solely from catalog metadata and validated READ observations."""
    definition = capability_definition(capability_id)
    software = SoftwareStatus(software_status)
    if definition is None:
        return CapabilityState(capability_id, BackendStatus.UNAVAILABLE, software,
                               ServiceStatus.NOT_REQUIRED, OverallStatus.UNKNOWN, False,
                               "capability is not registered", evidence)
    backend = (BackendStatus.AVAILABLE if (resolver or CapabilityBackendResolver()).available(capability_id)
               else BackendStatus.UNAVAILABLE)
    required_hosts = definition.prerequisites.required_host_capabilities
    if host_capabilities is not None and any(not host_capabilities.get(item, False) for item in required_hosts):
        return CapabilityState(capability_id, backend, software, ServiceStatus.NOT_REQUIRED,
                               OverallStatus.UNSUPPORTED, False,
                               "required host capability is unavailable", evidence)
    acquisition = definition.acquisition
    requirement = acquisition.service_requirement if acquisition else ServiceRequirement.NONE
    service = (ServiceStatus.NOT_REQUIRED if requirement is ServiceRequirement.NONE
               else ServiceStatus(service_status or ServiceStatus.UNAVAILABLE))
    possible = bool(acquisition and acquisition.install_supported and software is SoftwareStatus.MISSING)
    if software is SoftwareStatus.MISSING:
        overall, reason = OverallStatus.MISSING, "software prerequisite is missing"
    elif software is SoftwareStatus.UNKNOWN or backend is BackendStatus.UNAVAILABLE:
        overall, reason = OverallStatus.UNKNOWN, "software or READ backend availability is unknown"
    elif requirement is ServiceRequirement.ACTIVE and service is not ServiceStatus.ACTIVE:
        overall, reason = OverallStatus.DEGRADED, "required history collection service is not active"
    elif requirement is ServiceRequirement.INSTALLED and service is ServiceStatus.UNAVAILABLE:
        overall, reason = OverallStatus.DEGRADED, "required service is unavailable"
    else:
        overall, reason = OverallStatus.AVAILABLE, "all required capability prerequisites are available"
    return CapabilityState(capability_id, backend, software, service, overall, possible, reason, evidence)
