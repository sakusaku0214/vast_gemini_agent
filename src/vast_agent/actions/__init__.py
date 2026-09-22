"""Typed, approval-gated state-changing operations (never exposed to Gemini)."""

from vast_agent.actions.models import ActionRequest, ActionType, RiskClass

__all__ = ["ActionRequest", "ActionType", "RiskClass"]
