from __future__ import annotations

from enum import StrEnum

from vast_agent.actions.models import ActionRequest, ActionType, PackageInstallParameters
from vast_agent.actions.package_catalog import capability_definition
from vast_agent.agent.models import CapabilityGap


class AcquisitionIntent(StrEnum):
    READ_ONLY = "read_only"
    PROPOSE_IF_NEEDED = "propose_if_needed"
    EXPLICIT_INSTALL = "explicit_install"


_CONDITIONAL_ACQUISITION = (
    "必要なら入れて", "なければ入れて", "無ければ入れて", "入ってないなら入れて",
    "なければ導入", "無ければ導入", "足りないなら追加", "使えるようにして",
    "必要なtool入れて", "必要なツール入れて",
)


def acquisition_intent(text: str) -> AcquisitionIntent:
    folded = text.casefold()
    if any(phrase in folded for phrase in _CONDITIONAL_ACQUISITION):
        return AcquisitionIntent.PROPOSE_IF_NEEDED
    if any(word in folded for word in ("入れて", "install", "インストール")):
        return AcquisitionIntent.EXPLICIT_INSTALL
    return AcquisitionIntent.READ_ONLY


def request_for_gap(host: str, gap: CapabilityGap) -> ActionRequest | None:
    """Bridge only confirmed, sufficiently confident catalog capabilities to typed writes."""
    definition = capability_definition(gap.capability_id)
    if definition is None:
        return None
    assessed = assess_gap(gap)
    if (assessed.status != "missing" or assessed.software_status != "missing"
            or assessed.confidence not in {"medium", "high"}):
        return None
    return ActionRequest(
        host=host,
        action_type=ActionType.PACKAGE_INSTALL,
        parameters=PackageInstallParameters(
            package_name=definition.package_name,
            expected_capability=definition.capability_id,
            reason=f"{definition.description} capability is unavailable",
        ),
    )


def assess_gap(gap: CapabilityGap) -> CapabilityGap:
    """Apply code-owned execution support to the model's host-software observation.

    ``available`` means the registered agent can achieve the goal, not merely that a package
    exists. Model text and its proposed aggregate status never override catalog metadata.
    """
    definition = capability_definition(gap.capability_id)
    if definition is None or gap.software_status == "unknown":
        return gap.model_copy(update={"status": "unknown"})
    if gap.software_status == "missing":
        return gap.model_copy(update={"status": "missing"})
    if definition.agent_read_supported:
        return gap.model_copy(update={"status": "available"})
    reason = gap.reason.rstrip(". ")
    reason += "; host software exists but agent READ capability is not implemented"
    return gap.model_copy(update={"status": "unknown", "reason": reason})
