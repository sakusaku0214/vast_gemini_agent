import pytest

from vast_agent.actions.capability_backends import CapabilityBackendResolver
from vast_agent.actions.capability_bridge import assess_gap, request_for_gap
from vast_agent.actions.capability_state import (
    OverallStatus,
    ServiceStatus,
    SoftwareAvailabilityChecker,
    capability_state,
)
from vast_agent.actions.package_catalog import (
    CAPABILITY_REGISTRY,
    AcquisitionDefinition,
    CapabilityDefinition,
    CapabilityRegistry,
    ReadBackendDefinition,
)
from vast_agent.agent.host_read import HOST_READ_CAPABILITIES
from vast_agent.agent.models import CapabilityGap, InvestigationResult
from vast_agent.agent.prompts import SYSTEM_PROMPT, capability_prompt_block
from vast_agent.services.agent_service import AgentService


def test_existing_capabilities_are_declarative_and_resolve_backends():
    expected = {
        "traffic_history": "query_traffic_history",
        "network_interface_details": "inspect_interface",
        "nvme_health": "query_nvme_health",
    }
    assert {item.capability_id: item.read_backend.tool_name
            for item in CAPABILITY_REGISTRY.definitions} == expected
    assert all(CapabilityBackendResolver().available(item) for item in expected)


def test_registry_rejects_duplicates_unknown_and_unsafe_backends():
    item = CapabilityDefinition("sample", "sample", "system")
    with pytest.raises(ValueError, match="duplicate"):
        CapabilityRegistry((item, item)).validate()
    required = CapabilityDefinition(
        "sample", "sample", "system", ReadBackendDefinition("missing"),
    )
    with pytest.raises(ValueError, match="missing"):
        CapabilityRegistry((required,)).validate({})

    class Unsafe:
        risk_class = "WRITE"

    with pytest.raises(ValueError, match="READ_ONLY"):
        CapabilityRegistry((required,)).validate({"missing": Unsafe()})


def test_optional_unimplemented_backend_is_registered_but_unavailable():
    item = CapabilityDefinition(
        "future_health", "future", "system", ReadBackendDefinition("future_read", False),
    )
    CapabilityRegistry((item,)).validate(HOST_READ_CAPABILITIES)
    assert not CapabilityBackendResolver(registry={}).available("future_health")


def test_state_distinguishes_inactive_service_and_no_service_requirement():
    checker = SoftwareAvailabilityChecker()
    traffic = checker.assess(
        "traffic_history", package_installed=True, executable_status={"vnstat": True},
        service_installed=True, service_active=False,
    )
    assert traffic.overall_status is OverallStatus.DEGRADED
    assert traffic.service_status is ServiceStatus.INACTIVE
    network = checker.assess(
        "network_interface_details", package_installed=True,
        executable_status={"ethtool": True},
    )
    assert network.overall_status is OverallStatus.AVAILABLE
    assert network.service_status is ServiceStatus.NOT_REQUIRED


def traffic_gap(service_status: str, software_status: str = "available") -> CapabilityGap:
    return CapabilityGap(
        capability_id="traffic_history", status="available",
        software_status=software_status, service_status=service_status,
        reason=f"vnstat service is {service_status}", confidence="high",
    )


def test_inactive_service_flows_through_gap_without_install_proposal():
    gap = assess_gap(traffic_gap("inactive"))
    assert gap.status == "degraded"
    assert request_for_gap("host", gap) is None


def test_active_missing_and_unknown_service_production_semantics():
    assert assess_gap(traffic_gap("active")).status == "available"
    missing = assess_gap(traffic_gap("unknown", "missing"))
    assert missing.status == "missing"
    assert request_for_gap("host", missing) is None
    # Model status alone is not authoritative acquisition evidence.
    assert request_for_gap("host", missing, consent=True) is None
    unknown = assess_gap(traffic_gap("unknown"))
    assert unknown.status == "unknown"
    assert request_for_gap("host", unknown) is None


def test_renderer_identifies_degraded_as_no_automatic_acquisition():
    result = InvestigationResult(
        summary="history unavailable", confidence="high", recommended_action="NONE",
        capability_gaps=[traffic_gap("inactive")],
    )
    rendered = AgentService._format_investigation("host", result)
    assert "一部利用不可" in rendered
    assert "自動導入対象: なし (package already installed)" in rendered
    assert "Catalog候補" not in rendered


def test_backend_unavailable_never_becomes_available():
    state = capability_state(
        "nvme_health", "available", resolver=CapabilityBackendResolver(registry={}),
    )
    assert state.overall_status is OverallStatus.UNKNOWN


def test_prompt_vocabulary_is_registry_generated_and_bounded():
    assert all(item.capability_id in SYSTEM_PROMPT for item in CAPABILITY_REGISTRY.definitions)
    assert len(capability_prompt_block(max_chars=90)) <= 90
    assert "metadata only, not a knowledge boundary" in capability_prompt_block()


def test_acquisition_metadata_is_catalog_owned():
    definition = CapabilityDefinition(
        "sample", "sample", "system", acquisition=AcquisitionDefinition("safe-package"),
    )
    registry = CapabilityRegistry((definition,))
    registry.validate()
    assert registry.get("sample").acquisition.package_name == "safe-package"


def test_unknown_gap_is_free_form_missing_evidence_not_fake_acquisition():
    from vast_agent.actions.package_catalog import capability_definition

    assert capability_definition("unknown_host_tool") is None
    result = InvestigationResult(
        summary="tool syntax unavailable", missing_evidence=["unknown_host_tool safe READ syntax"],
        confidence="low", recommended_action="NONE",
    )
    assert result.capability_gaps == []
    assert "unknown_host_tool" in result.missing_evidence[0]
