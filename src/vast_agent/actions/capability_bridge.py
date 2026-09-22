from __future__ import annotations

from enum import StrEnum

from vast_agent.actions.capability_backends import CapabilityBackendResolver
from vast_agent.actions.capability_state import SoftwareAvailabilityChecker
from vast_agent.actions.models import ActionRequest, ActionType, PackageInstallParameters
from vast_agent.actions.package_catalog import ServiceRequirement, capability_definition
from vast_agent.agent.investigation_session import InvestigationSession
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


def request_for_gap(host: str, gap: CapabilityGap, *, consent: bool = False) -> ActionRequest | None:
    """Bridge only confirmed, sufficiently confident catalog capabilities to typed writes."""
    if not consent or not gap._acquisition_grounded:
        return None
    definition = capability_definition(gap.capability_id)
    if (definition is None or definition.acquisition is None
            or not definition.acquisition.install_supported):
        return None
    assessed = assess_gap(gap)
    if (assessed.status != "missing" or assessed.software_status != "missing"
            or assessed.confidence not in {"medium", "high"}):
        return None
    return ActionRequest(
        host=host,
        action_type=ActionType.PACKAGE_INSTALL,
        parameters=PackageInstallParameters(
            package_name=definition.acquisition.package_name,
            expected_capability=definition.capability_id,
            reason=f"{definition.description} capability is unavailable",
        ),
    )


def ground_capability_gap(gap: CapabilityGap, session: InvestigationSession) -> CapabilityGap:
    """Replace model claims with typed, same-session prerequisite READ evidence.

    Package absence plus absence of every catalog-owned executable is required before acquisition
    can be grounded. Missing, failed, cross-host, or contradictory evidence always becomes unknown.
    Model prose, confidence, and ``candidate_package`` are never used as evidence.
    """
    definition = capability_definition(gap.capability_id)
    acquisition = definition.acquisition if definition else None
    if acquisition is None:
        return gap.model_copy(update={
            "status": "unknown", "software_status": "unknown",
            "reason": "acquisition metadata is unavailable",
        })

    def matching(source: str, field: str, value: str):
        return [record for record in session.evidence
                if record.target_host is not None
                and record.target_host.casefold() == session.target_host.casefold()
                and record.source == source and record.arguments.get("host") == session.target_host
                and record.arguments.get(field) == value]

    package_records = matching("query_package", "package_name", acquisition.package_name)
    executable_records = {
        name: matching("query_executable", "executable_name", name)
        for name in acquisition.executables
    }
    package = package_records[-1] if len(package_records) == 1 else None
    executables = {
        name: records[-1] if len(records) == 1 else None
        for name, records in executable_records.items()
    }
    evidence_complete = (
        package is not None and package.status == "available"
        and isinstance(package.facts.get("installed"), bool)
        and all(record is not None and record.status == "available"
                and isinstance(record.facts.get("exists"), bool)
                for record in executables.values())
    )
    if not evidence_complete:
        return gap.model_copy(update={
            "status": "unknown", "software_status": "unknown", "service_status": "unknown",
            "candidate_package": acquisition.package_name,
            "reason": "software absence is not confirmed by complete prerequisite READ evidence",
        })

    package_installed = package.facts["installed"]
    executable_status = {name: record.facts["exists"] for name, record in executables.items()}
    # Any disagreement is uncertainty, rather than permission to install.
    if any(bool(value) != bool(package_installed) for value in executable_status.values()):
        return gap.model_copy(update={
            "status": "unknown", "software_status": "unknown", "service_status": "unknown",
            "candidate_package": acquisition.package_name,
            "reason": "package and executable READ evidence are contradictory",
        })

    service_installed: bool | None = None
    service_active: bool | None = None
    if acquisition.service_requirement is not ServiceRequirement.NONE:
        service_records = [
            record for service in acquisition.services
            for record in matching("query_service", "service_name", service)
        ]
        if len(service_records) == len(acquisition.services) and all(
            record.status == "available" and isinstance(record.facts.get("exists"), bool)
            for record in service_records
        ):
            service_installed = all(bool(record.facts["exists"]) for record in service_records)
            active_values = [record.facts.get("active_state") == "active" for record in service_records]
            service_active = all(active_values) if service_installed else False

    state = SoftwareAvailabilityChecker().assess(
        gap.capability_id, package_installed=bool(package_installed),
        executable_status={name: bool(value) for name, value in executable_status.items()},
        service_installed=service_installed, service_active=service_active,
    )
    grounded = gap.model_copy(update={
        "status": state.overall_status.value,
        "software_status": state.software_status.value,
        "service_status": state.service_status.value,
        "candidate_package": acquisition.package_name,
        "reason": state.reason,
        "evidence": [
            f"query_package:{acquisition.package_name}:installed={package_installed}",
            *[f"query_executable:{name}:exists={value}"
              for name, value in executable_status.items()],
        ],
    })
    grounded._acquisition_grounded = (
        state.software_status.value == "missing" and not bool(package_installed)
        and all(value is False for value in executable_status.values())
    )
    return grounded


def assess_gap(gap: CapabilityGap, resolver: CapabilityBackendResolver | None = None) -> CapabilityGap:
    """Apply the authoritative capability state machine to model READ observations.

    ``available`` means the registered agent can achieve the goal, not merely that a package
    exists. Service requirements and backend safety come from code-owned metadata; the model's
    proposed aggregate status and candidate package never override them.
    """
    definition = capability_definition(gap.capability_id)
    if definition is None:
        return gap.model_copy(update={"status": "unknown"})
    checker = SoftwareAvailabilityChecker(resolver or CapabilityBackendResolver())
    state = checker.assess_status(
        gap.capability_id, gap.software_status, gap.service_status,
        evidence=tuple(gap.evidence),
    )
    reason = gap.reason
    if state.overall_status.value != "available" and state.reason not in reason:
        reason = f"{reason.rstrip('. ')}; {state.reason}"
    return gap.model_copy(update={"status": state.overall_status.value, "reason": reason})
