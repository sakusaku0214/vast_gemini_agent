
from vast_agent.actions.operation_plan import OperationPlan
from vast_agent.agent.gemini import ScriptedGeminiClient
from vast_agent.agent.models import AgentResponse, InvestigationResult
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.config import GeminiSettings
from vast_agent.storage.database import Database


class NoFunctions:
    pass


def agent(tmp_path, outputs):
    db = Database(tmp_path / "db.sqlite")
    db.migrate()
    return InvestigationAgent(
        ScriptedGeminiClient([AgentResponse(output_text=value) for value in outputs]),
        NoFunctions(), db, GeminiSettings(),
    )


def operation(**changes):
    values = dict(
        host="garage-h12ssl-nt", host_source="conversation context", executable="nvidia-smi",
        argv=["-i", "0", "--lock-gpu-clocks=1500,1500"], requires_sudo=True,
        target="GPU 0", reason="bounded adjustment", expected_effect="lower GPU clock",
        verification_plan="read GPU state", verification_kind="gpu_state",
        verification_target="0", command_source="bounded value derived from request + evidence",
    )
    values.update(changes)
    return OperationPlan(**values).model_dump_json()


def evidence():
    return InvestigationResult(
        summary="GPU0 is loaded at 340 W", findings=["GPU0 clock is 1800 MHz"],
        confidence="high", recommended_action="NONE",
    )


def test_semantic_classifier_gate_was_removed(tmp_path):
    planner = agent(tmp_path, [])
    assert not hasattr(planner, "classify_host_intent")


def test_operation_planner_rejects_changed_host_and_invented_target(tmp_path):
    changed = agent(tmp_path / "host", [operation(host="other-host")])
    assert changed.plan_operation(
        "garage-h12ssl-nt", "conversation context", "GPU0を抑えて", evidence(),
    ) is None

    invented = agent(tmp_path / "target", [operation(verification_target="1", target="GPU 1",
                                                       argv=["-i", "1", "--lock-gpu-clocks=1500,1500"])])
    assert invented.plan_operation(
        "garage-h12ssl-nt", "conversation context", "GPU0を抑えて", evidence(),
    ) is None


def test_operation_planner_rejects_invented_target_and_value_without_verification(tmp_path):
    invented = agent(tmp_path, [operation(
        target="fan-zone-9", executable="vendorctl", argv=["set", "fan-zone-9", "87"],
        verification_kind="unavailable", verification_target=None,
        command_source="model suggestion",
    )])
    assert invented.plan_operation(
        "garage-h12ssl-nt", "conversation context", "冷却を少し調整して", evidence(),
    ) is None
