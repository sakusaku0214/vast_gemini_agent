import pytest
from pydantic import ValidationError

from vast_agent.actions.operation_plan import OperationPlan, render_operation_proposal


def plan(**overrides):
    values = dict(host="garage-x570", host_source="conversation context", executable="systemctl",
                  argv=["restart", "vastai.service"], requires_sudo=True, target="vastai.service",
                  reason="daemon unresponsive", expected_effect="restart one service",
                  verification_plan="query service state", command_source="CLI help + Agent planning",
                  verification_kind="service_state", verification_target="vastai.service",
                  current_relevant_state="failed", active_workload="none", running_vm="none")
    values.update(overrides)
    return OperationPlan(**values)


def test_plan_fingerprint_covers_exact_operation_and_preflight():
    original = plan()
    changed = plan(argv=["restart", "docker.service"], target="docker.service",
                   verification_target="docker.service")
    assert original.fingerprint({"reachable": True}) != changed.fingerprint({"reachable": True})
    assert original.execution_argv() == ("sudo", "-n", "systemctl", "restart", "vastai.service")


def test_hard_floor_and_shell_strings_are_rejected():
    with pytest.raises(ValidationError):
        plan(executable="bash", argv=["-c", "systemctl restart vastai"])
    with pytest.raises(ValidationError):
        plan(executable="mkfs.ext4", argv=["/dev/sda"])
    with pytest.raises(ValidationError):
        plan(argv=["restart", "vastai.service; reboot"])


@pytest.mark.parametrize(("overrides"), [
    {"executable": "python3", "argv": ["-c", "print('x')"]},
    {"executable": "rm", "argv": ["--force", "--recursive", "/"]},
    {"executable": "dd", "argv": ["if=/dev/zero", "of=/dev/sda"]},
    {"executable": "wipefs", "argv": ["--all", "/dev/sda"]},
    {"host": "fleet"},
    {"argv": ["disable", "vast-agent.service"], "verification_kind": "unavailable",
     "verification_target": None},
    {"executable": "/usr/bin/systemctl"},
])
def test_adversarial_hard_floor(overrides):
    with pytest.raises(ValidationError):
        plan(**overrides)


def test_proposal_is_self_contained():
    rendered = render_operation_proposal(plan(), 12, "2030-01-01T00:00:00Z")
    for label in ("Host:", "Host source:", "Risk: DANGEROUS", "Sudo: required", "Executable:",
                  "Arguments:", "Target:", "Current relevant state:", "Active workload:",
                  "Running VM:", "Reason:", "Expected effect:", "Known side effects:",
                  "Verification plan:", "Rollback available:", "Expiry:"):
        assert label in rendered
