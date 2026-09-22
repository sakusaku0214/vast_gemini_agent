from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityPackageDefinition:
    """Code-owned package acquisition entry; model output cannot extend this catalog."""

    capability_id: str
    description: str
    package_name: str
    executables: tuple[str, ...]
    services: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ("debian", "ubuntu")


CAPABILITY_PACKAGES = (
    CapabilityPackageDefinition(
        "traffic_history", "historical network traffic", "vnstat", ("vnstat",), ("vnstat",),
    ),
    CapabilityPackageDefinition(
        "network_interface_details", "network interface details", "ethtool", ("ethtool",),
    ),
    CapabilityPackageDefinition(
        "nvme_health", "NVMe device health", "nvme-cli", ("nvme",),
    ),
)

PACKAGES_BY_NAME = {entry.package_name: entry for entry in CAPABILITY_PACKAGES}
PACKAGES_BY_CAPABILITY = {entry.capability_id: entry for entry in CAPABILITY_PACKAGES}


def package_definition(package_name: str) -> CapabilityPackageDefinition | None:
    return PACKAGES_BY_NAME.get(package_name)
