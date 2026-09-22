from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class ServiceRequirement(StrEnum):
    NONE = "none"
    INSTALLED = "installed"
    ACTIVE = "active"


@dataclass(frozen=True)
class ReadBackendDefinition:
    tool_name: str
    required: bool = True


@dataclass(frozen=True)
class AcquisitionDefinition:
    package_name: str
    executables: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    install_supported: bool = True
    service_requirement: ServiceRequirement = ServiceRequirement.NONE


@dataclass(frozen=True)
class CapabilityPrerequisites:
    required_host_capabilities: tuple[str, ...] = ()
    optional_host_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityDefinition:
    """One code-owned high-level capability; never populated from model/user input."""

    capability_id: str
    description: str
    category: str
    read_backend: ReadBackendDefinition | None = None
    acquisition: AcquisitionDefinition | None = None
    prerequisites: CapabilityPrerequisites = field(default_factory=CapabilityPrerequisites)
    platforms: tuple[str, ...] = ("debian", "ubuntu")
    history_requires_prior_collection: bool = False
    safety_notes: str | None = None

    # Compatibility properties for callers of the pre-PR20 package catalog.
    @property
    def package_name(self) -> str | None:
        return self.acquisition.package_name if self.acquisition else None

    @property
    def executables(self) -> tuple[str, ...]:
        return self.acquisition.executables if self.acquisition else ()

    @property
    def services(self) -> tuple[str, ...]:
        return self.acquisition.services if self.acquisition else ()

    @property
    def read_tool_name(self) -> str | None:
        return self.read_backend.tool_name if self.read_backend else None

    @property
    def agent_read_supported(self) -> bool:
        return self.read_backend is not None

    @property
    def supports_history(self) -> bool:
        return self.history_requires_prior_collection

    @property
    def post_install_notes(self) -> str | None:
        return self.safety_notes


# Kept as an alias, not a second metadata model.
CapabilityPackageDefinition = CapabilityDefinition


class CapabilityRegistry:
    _ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
    _PACKAGE = re.compile(r"^[a-z0-9][a-z0-9+.-]*$")
    _EXECUTABLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
    _SERVICE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$")
    _PLATFORMS = frozenset({"debian", "ubuntu"})

    def __init__(self, definitions: Sequence[CapabilityDefinition]) -> None:
        self.definitions = tuple(definitions)
        self.by_id = {item.capability_id: item for item in self.definitions}

    def get(self, capability_id: str) -> CapabilityDefinition | None:
        return self.by_id.get(capability_id)

    def validate(self, read_registry: Mapping[str, object] | None = None) -> None:
        if len(self.by_id) != len(self.definitions):
            raise ValueError("duplicate capability_id")
        package_owners: dict[str, CapabilityDefinition] = {}
        for item in self.definitions:
            if not self._ID.fullmatch(item.capability_id):
                raise ValueError(f"invalid capability_id: {item.capability_id}")
            if not item.platforms or not set(item.platforms) <= self._PLATFORMS:
                raise ValueError(f"unsupported platform metadata: {item.capability_id}")
            acquisition = item.acquisition
            if acquisition:
                if not acquisition.package_name or not self._PACKAGE.fullmatch(acquisition.package_name):
                    raise ValueError(f"invalid package name: {item.capability_id}")
                if any(not self._EXECUTABLE.fullmatch(value) for value in acquisition.executables):
                    raise ValueError(f"invalid executable name: {item.capability_id}")
                if any(not self._SERVICE.fullmatch(value) or ".." in value
                       for value in acquisition.services):
                    raise ValueError(f"invalid service name: {item.capability_id}")
                if not isinstance(acquisition.service_requirement, ServiceRequirement):
                    raise ValueError(f"unknown service requirement: {item.capability_id}")
                if (acquisition.service_requirement is not ServiceRequirement.NONE
                        and not acquisition.services):
                    raise ValueError(f"service requirement has no service: {item.capability_id}")
                previous = package_owners.setdefault(acquisition.package_name, item)
                if previous.capability_id != item.capability_id and previous.acquisition != acquisition:
                    raise ValueError(f"conflicting package mapping: {acquisition.package_name}")
            backend = item.read_backend
            if read_registry is not None and backend:
                tool = read_registry.get(backend.tool_name)
                if tool is None:
                    if backend.required:
                        raise ValueError(f"required read backend missing: {backend.tool_name}")
                elif getattr(tool, "risk_class", None) != "READ_ONLY":
                    raise ValueError(f"read backend is not READ_ONLY: {backend.tool_name}")


CAPABILITY_DEFINITIONS = (
    CapabilityDefinition(
        "gpu_diagnostics", "bounded GPU health and ownership diagnostics", "gpu",
        ReadBackendDefinition("query_gpu_diagnostics"),
        prerequisites=CapabilityPrerequisites(optional_host_capabilities=("nvidia",)),
        safety_notes="READ-only PCI/NVML evidence; no reset or remediation is performed.",
    ),
    CapabilityDefinition(
        "docker_diagnostics", "bounded Docker service and workload diagnostics", "containers",
        ReadBackendDefinition("query_docker_diagnostics"),
        prerequisites=CapabilityPrerequisites(required_host_capabilities=("docker",)),
        safety_notes="READ-only service and container summaries; no container mutation is performed.",
    ),
    CapabilityDefinition(
        "traffic_history", "historical network traffic", "network",
        ReadBackendDefinition("query_traffic_history"),
        AcquisitionDefinition("vnstat", ("vnstat",), ("vnstat",), True, ServiceRequirement.ACTIVE),
        history_requires_prior_collection=True,
        safety_notes="導入前の通信履歴は取得できません。今後の履歴を記録できます。",
    ),
    CapabilityDefinition(
        "network_interface_details", "detailed network interface state", "network",
        ReadBackendDefinition("inspect_interface"), AcquisitionDefinition("ethtool", ("ethtool",)),
    ),
    CapabilityDefinition(
        "nvme_health", "NVMe SMART health", "storage",
        ReadBackendDefinition("query_nvme_health"), AcquisitionDefinition("nvme-cli", ("nvme",)),
    ),
)

CAPABILITY_REGISTRY = CapabilityRegistry(CAPABILITY_DEFINITIONS)
CAPABILITY_REGISTRY.validate()
CAPABILITY_PACKAGES = tuple(item for item in CAPABILITY_DEFINITIONS if item.acquisition)
PACKAGES_BY_NAME = {item.acquisition.package_name: item for item in CAPABILITY_PACKAGES if item.acquisition}
PACKAGES_BY_CAPABILITY = CAPABILITY_REGISTRY.by_id


def package_definition(package_name: str) -> CapabilityDefinition | None:
    return PACKAGES_BY_NAME.get(package_name)


def capability_definition(capability_id: str) -> CapabilityDefinition | None:
    return CAPABILITY_REGISTRY.get(capability_id)
