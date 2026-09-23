from __future__ import annotations

import json

import pytest

from vast_agent.actions.models import ActionType
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.models import AgentResponse
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.config import GeminiSettings
from vast_agent.storage.database import Database


class NoFunctions:
    def execute(self, *args):
        raise AssertionError("action interpretation must not expose or execute tools")


def agent(tmp_path, payload):
    database = Database(tmp_path / "agent.db")
    database.migrate()
    client = ScriptedGeminiClient([AgentResponse(output_text=json.dumps(payload))])
    return InvestigationAgent(client, NoFunctions(), database, GeminiSettings()), client


def test_flexible_action_interpretation_produces_only_grounded_typed_request(tmp_path):
    interpreter, client = agent(tmp_path, {
        "action_type": "PACKAGE_INSTALL",
        "parameters": {
            "kind": "package_install", "package_name": "nvitop",
            "expected_capability": None, "reason": "explicit user request",
        },
    })

    request = interpreter.interpret_action(
        "garage-x570", "X570へnvitopを導入しておいて",
    )

    assert request is not None
    assert request.host == "garage-x570"
    assert request.action_type == ActionType.PACKAGE_INSTALL
    assert request.parameters.package_name == "nvitop"
    assert request.parameters.expected_capability is None
    assert client.requests[0]["tools"] == []


@pytest.mark.parametrize("message", [
    "X570へGPU監視ツールを導入して",  # model-only recommendation
    "X570へnvtopを導入して",  # silent spelling substitution
])
def test_model_only_or_rewritten_package_target_is_not_promoted(tmp_path, message):
    interpreter, _ = agent(tmp_path, {
        "action_type": "PACKAGE_INSTALL",
        "parameters": {
            "kind": "package_install", "package_name": "nvitop",
            "expected_capability": None, "reason": "explicit user request",
        },
    })

    assert interpreter.interpret_action("garage-x570", message) is None


@pytest.mark.parametrize("payload", [
    {"action_type": "RUN_ARBITRARY_SHELL", "parameters": {"command": "id"}},
    {"action_type": "PACKAGE_INSTALL", "parameters": {
        "kind": "package_install", "package_name": "nvitop", "unexpected": "argv",
    }},
    {"action_type": None, "parameters": {}},
])
def test_interpreter_rejects_unknown_actions_untyped_parameters_and_non_writes(tmp_path, payload):
    interpreter, client = agent(tmp_path, payload)

    assert interpreter.interpret_action("garage-x570", "X570をいい感じにして") is None
    assert client.requests[0]["tools"] == []
