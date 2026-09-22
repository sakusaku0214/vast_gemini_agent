from __future__ import annotations

import pytest
from pydantic import ValidationError

from vast_agent.actions.capability_backends import CapabilityBackendResolver
from vast_agent.actions.capability_bridge import (
    AcquisitionIntent,
    acquisition_intent,
    assess_gap,
    request_for_gap,
)
from vast_agent.actions.package_catalog import capability_definition
from vast_agent.agent.host_read import FUNCTION_DECLARATIONS
from vast_agent.agent.models import CapabilityGap, InvestigationResult
from vast_agent.services.agent_service import AgentService


def gap(**updates) -> CapabilityGap:
    values = {
        "capability_id": "traffic_history",
        "status": "missing",
        "software_status": "missing",
        "reason": "validated package and executable are absent",
        "evidence": ["vnstat package: not installed", "vnstat executable: missing"],
        "candidate_package": "curl",
        "confidence": "high",
    }
    values.update(updates)
    return CapabilityGap.model_validate(values)


@pytest.mark.parametrize(("text", "expected"), [
    ("magの昨日の通信量見て", AcquisitionIntent.READ_ONLY),
    ("magの昨日の通信量見たい。必要なら入れて", AcquisitionIntent.PROPOSE_IF_NEEDED),
    ("magにvnstat入れて", AcquisitionIntent.EXPLICIT_INSTALL),
])
def test_acquisition_intent_is_separate_from_goal(text, expected):
    assert acquisition_intent(text) == expected


def test_catalog_is_authoritative_over_model_candidate():
    request = request_for_gap("garage-mag", gap(candidate_package="curl"), consent=True)
    assert request is not None
    assert request.parameters.package_name == "vnstat"
    assert request.parameters.expected_capability == "traffic_history"


@pytest.mark.parametrize("updates", [
    {"status": "unknown", "software_status": "unknown"},
    {"status": "available", "software_status": "available"},
    {"confidence": "low"},
])
def test_unconfirmed_gap_cannot_create_action_request(updates):
    assert request_for_gap("garage-mag", gap(**updates)) is None


def test_unknown_capability_and_status_fail_closed_at_validation():
    with pytest.raises(ValidationError):
        gap(capability_id="arbitrary_shell")
    with pytest.raises(ValidationError):
        gap(status="probably-missing")


@pytest.mark.parametrize(("capability", "package"), [
    ("traffic_history", "vnstat"),
    ("network_interface_details", "ethtool"),
    ("nvme_health", "nvme-cli"),
])
def test_capability_vocabulary_maps_only_through_catalog(capability, package):
    definition = capability_definition(capability)
    assert definition is not None
    assert definition.package_name == package


def test_historical_caveat_is_code_owned_and_rendered():
    definition = capability_definition("traffic_history")
    assert definition is not None
    assert definition.history_requires_prior_collection is True
    result = InvestigationResult(
        summary="履歴能力がありません", confidence="high", recommended_action="NONE",
        capability_gaps=[gap()],
    )
    rendered = AgentService._format_investigation("garage-mag", result)
    assert "導入前の通信履歴は取得できません" in rendered
    assert "Catalog候補: vnstat" in rendered
    assert "curl" not in rendered


def test_installed_software_is_full_capability_with_registered_read_tool():
    assessed = assess_gap(gap(
        status="available", software_status="available", service_status="active",
    ))
    assert assessed.status == "available"
    assert request_for_gap("garage-mag", assessed) is None


def test_network_details_is_available_when_software_and_read_tool_exist():
    assessed = assess_gap(gap(
        capability_id="network_interface_details", status="unknown",
        software_status="available", candidate_package="something-else",
    ))
    assert assessed.status == "available"
    definition = capability_definition("network_interface_details")
    assert definition is not None
    assert definition.read_tool_name == "inspect_interface"
    assert definition.agent_read_supported is True


def test_installed_nvme_software_has_registered_backend_and_needs_no_install():
    assessed = assess_gap(gap(
        capability_id="nvme_health", status="available", software_status="available",
    ))
    assert assessed.status == "available"
    assert request_for_gap("garage-mag", assessed) is None


def test_software_missing_is_distinct_from_agent_read_support_missing():
    software_missing = assess_gap(gap(software_status="missing"))
    agent_read_missing = assess_gap(
        gap(status="available", software_status="available"),
        CapabilityBackendResolver(registry={}),
    )
    assert software_missing.status == "missing"
    assert request_for_gap("garage-mag", software_missing, consent=True) is not None
    assert agent_read_missing.status == "unknown"
    assert request_for_gap("garage-mag", agent_read_missing) is None


def test_gemini_declarations_remain_read_only():
    declarations = str(FUNCTION_DECLARATIONS).casefold()
    assert "package_install" not in declarations
    assert "apt-get" not in declarations
