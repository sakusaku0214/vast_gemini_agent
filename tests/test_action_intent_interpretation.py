from vast_agent.agent.orchestrator import InvestigationAgent


def test_pre_investigation_action_interpreter_was_removed():
    assert not hasattr(InvestigationAgent, "interpret_action")
