from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    """Compact, explicitly untrusted view of one READ result."""

    source: str
    target_host: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)
    status: Literal["available", "evidence_missing", "tool_unavailable"]
    facts: dict[str, object] = Field(default_factory=dict)
    signatures: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    confidence_hint: Literal["low", "medium", "high"] = "medium"
    summary: str = ""


_UNAVAILABLE = {
    "FUNCTION_NOT_ALLOWED", "HOST_NOT_FOUND", "HOST_DISABLED", "TOOL_UNSUPPORTED",
    "COMMAND_NOT_AVAILABLE",
}


def compact_evidence(source: str, output: dict[str, object], max_chars: int) -> EvidenceRecord:
    """Normalize a tool result without ever treating its content as instructions."""
    error = output.get("error")
    payload = output.get("untrusted_evidence", output)
    nested_error = payload.get("error") if isinstance(payload, dict) else None
    failed_read = isinstance(payload, dict) and payload.get("status") == "failed"
    if error in _UNAVAILABLE or nested_error in _UNAVAILABLE:
        status = "tool_unavailable"
    elif error or (isinstance(payload, dict) and payload.get("error")) or failed_read:
        status = "evidence_missing"
    else:
        status = "available"
    encoded = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    summary = encoded[:max_chars]
    if len(encoded) > max_chars:
        summary += "…[truncated]"
    facts = payload if isinstance(payload, dict) and len(encoded) <= max_chars else {}
    signatures = facts.get("signatures", [])
    if not isinstance(signatures, list):
        signatures = []
    failure_kind = payload.get("failure_kind") if isinstance(payload, dict) else None
    missing = [] if status == "available" else [
        str(error or nested_error or failure_kind or "no usable evidence")
    ]
    return EvidenceRecord(
        source=source, status=status, facts=facts,
        signatures=[str(value) for value in signatures[:12]],
        missing_evidence=missing, confidence_hint="medium" if status == "available" else "low",
        summary=summary,
    )
