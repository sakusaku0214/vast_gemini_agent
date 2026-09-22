from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, Field

from vast_agent.agent.evidence import EvidenceRecord


class StopReason(StrEnum):
    ANSWERABLE = "ANSWERABLE"
    BOUND_REACHED = "BOUND_REACHED"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    NO_NEW_EVIDENCE = "NO_NEW_EVIDENCE"
    CAPABILITY_GAP = "CAPABILITY_GAP"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class InvestigationPlan(BaseModel):
    goal: str
    questions: list[str] = Field(default_factory=list, max_length=8)
    initial_tools: list[str] = Field(default_factory=list, max_length=4)
    stop_when: list[str] = Field(default_factory=list, max_length=6)
    constraints: list[str] = Field(default_factory=list, max_length=8)


class InvestigationSession(BaseModel):
    """Request-local bounds, trace, and exact-call evidence cache."""

    target_host: str
    goal: str
    round: int = 0
    tool_calls: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    answered_questions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    capability_gaps: list[str] = Field(default_factory=list, max_length=3)
    stop_reason: StopReason | None = None
    max_tool_calls: int = 6
    max_rounds: int = 4
    max_total_evidence_chars: int = 8000

    def call_key(self, tool_name: str, arguments: dict[str, object]) -> str:
        normalized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{self.target_host.casefold()}:{tool_name}:{normalized}"

    def has_call(self, tool_name: str, arguments: dict[str, object]) -> bool:
        return self.call_key(tool_name, arguments) in self.tool_calls

    @property
    def evidence_chars(self) -> int:
        return sum(len(item.model_dump_json()) for item in self.evidence)

    def can_call(self) -> bool:
        # The last LLM round is reserved for synthesis of evidence gathered previously.
        return len(self.tool_calls) < self.max_tool_calls and self.round < self.max_rounds

    def add(self, tool_name: str, arguments: dict[str, object], evidence: EvidenceRecord) -> bool:
        key = self.call_key(tool_name, arguments)
        if key in self.tool_calls:
            self.stop_reason = StopReason.NO_NEW_EVIDENCE
            return False
        self.tool_calls.append(key)
        evidence = evidence.model_copy(update={
            "target_host": self.target_host,
            "arguments": arguments,
        })
        remaining = max(0, self.max_total_evidence_chars - self.evidence_chars)
        if remaining == 0:
            self.stop_reason = StopReason.BOUND_REACHED
            return False
        if len(evidence.model_dump_json()) > remaining:
            evidence = evidence.model_copy(update={"facts": {}, "summary": "[evidence truncated]"})
        if len(evidence.model_dump_json()) > remaining:
            self.stop_reason = StopReason.BOUND_REACHED
            return False
        self.evidence.append(evidence)
        return True

    def trace(self) -> dict[str, object]:
        return {
            "selected_tools": [key.split(":", 2)[1] for key in self.tool_calls],
            "rounds": self.round,
            "stop_reason": self.stop_reason,
            "evidence_sources": [item.source for item in self.evidence],
        }
