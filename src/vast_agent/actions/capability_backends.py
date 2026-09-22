from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vast_agent.actions.package_catalog import capability_definition
from vast_agent.agent.host_read import HOST_READ_CAPABILITIES


@dataclass(frozen=True)
class CapabilityBackendResolver:
    """Cross-check catalog metadata against the executable READ registry."""

    registry: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.registry is None:
            object.__setattr__(self, "registry", HOST_READ_CAPABILITIES)

    def available(self, capability_id: str) -> bool:
        definition = capability_definition(capability_id)
        if definition is None or not definition.agent_read_supported or not definition.read_tool_name:
            return False
        assert self.registry is not None
        backend = self.registry.get(definition.read_tool_name)
        return (backend is not None and getattr(backend, "available", False)
                and getattr(backend, "risk_class", None) == "READ_ONLY")
