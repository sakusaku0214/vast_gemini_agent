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
    supports_history: bool = False
    history_requires_prior_collection: bool = False
    post_install_notes: str | None = None


CAPABILITY_PACKAGES = (
    CapabilityPackageDefinition(
        "traffic_history", "historical network traffic", "vnstat", ("vnstat",), ("vnstat",),
        supports_history=True,
        history_requires_prior_collection=True,
        post_install_notes="導入前の通信履歴は取得できません。今後の履歴を記録できます。",
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


def capability_definition(capability_id: str) -> CapabilityPackageDefinition | None:
    """Resolve a model-selected capability through the authoritative code-owned catalog."""
    return PACKAGES_BY_CAPABILITY.get(capability_id)
