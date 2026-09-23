from __future__ import annotations

import json
import re
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
    # Kept for application rendering/cache; summary is the only copy sent back to the model.
    relevant_excerpt: str = Field(default="", max_length=4000, exclude=True)


_UNAVAILABLE = {
    "FUNCTION_NOT_ALLOWED", "HOST_NOT_FOUND", "HOST_DISABLED", "TOOL_UNSUPPORTED",
    "COMMAND_NOT_AVAILABLE",
}

_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x1b\x07]*(?:\x07|\x1b\\))")
# Output captured after an upstream control-character scrub can contain a CSI
# sequence without its ESC byte.  Keep this deliberately limited to SGR colors;
# matching arbitrary bracketed text would corrupt normal command output.
_DAMAGED_SGR = re.compile(r"\[(?:\d{1,3}(?:;\d{1,3})*)?m")


def strip_ansi(value: str) -> str:
    """Remove terminal control sequences without collapsing table whitespace."""
    return _DAMAGED_SGR.sub("", _ANSI_ESCAPE.sub("", value))


def compact_evidence(source: str, output: dict[str, object], max_chars: int) -> EvidenceRecord:
    """Normalize a tool result without ever treating its content as instructions."""
    error = output.get("error")
    payload = output.get("untrusted_evidence", output)
    if isinstance(payload, dict):
        payload = {
            key: strip_ansi(value) if isinstance(value, str) else value
            for key, value in payload.items()
        }
    nested_error = payload.get("error") if isinstance(payload, dict) else None
    failed_read = isinstance(payload, dict) and payload.get("status") == "failed"
    if error in _UNAVAILABLE or nested_error in _UNAVAILABLE:
        status = "tool_unavailable"
    elif error or (isinstance(payload, dict) and payload.get("error")) or failed_read:
        status = "evidence_missing"
    else:
        status = "available"
    excerpt = _relevant_excerpt(payload, max_chars=max_chars)
    encoded = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    summary = encoded[:max_chars]
    if len(encoded) > max_chars:
        summary += "…[truncated]"
    if excerpt and len(encoded) > max_chars:
        summary = excerpt
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
        summary=summary, relevant_excerpt=excerpt,
    )


def _relevant_excerpt(payload: object, *, max_chars: int) -> str:
    """Keep useful command evidence even when the surrounding payload is very large."""
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        return ""
    stdout = payload.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return ""
    executable = str(payload.get("executable", "")).rsplit("/", 1)[-1].casefold()
    argv = [str(value).casefold() for value in payload.get("argv", [])]
    lines = stdout.splitlines()
    if executable == "crontab" and argv == ["-l"]:
        # Comments remain relevant context, but omit empty padding.
        selected = [line for line in lines if line.strip()]
    elif executable == "nvidia-smi" and any("clock" in value for value in argv):
        # nvidia-smi -q is section-oriented. Retain GPU boundaries and the clock
        # sections rather than whichever bytes happen to occur first.
        selected = []
        keep_indent = None
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped.startswith(("GPU ", "Product Name")):
                selected.append(line)
            if any(label in stripped for label in (
                "Locked Clocks", "Max Clocks", "Applications Clocks", "Default Applications Clocks",
            )):
                selected.append(line)
                keep_indent = indent
                continue
            if keep_indent is not None:
                if stripped and indent > keep_indent:
                    selected.append(line)
                elif stripped:
                    keep_indent = None
    else:
        selected = lines
    excerpt = "\n".join(selected).strip()
    return excerpt[:max_chars] + ("…[truncated]" if len(excerpt) > max_chars else "")
