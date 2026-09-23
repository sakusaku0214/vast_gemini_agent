from vast_agent.agent.generic_read import (
    ReadClassification,
    run_validated_read,
    sanitize_output,
    validate_read_argv,
)
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ToolResult


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        self.calls.append((host.name, tuple(command), timeout))
        return ToolResult(success=True, exit_code=0, stdout="ok\x00\n", duration_ms=1)


def host():
    return Host(name="x570", address="192.0.2.1", ssh_user="agent")


def test_help_and_vast_show_are_read():
    assert validate_read_argv("vastai", ["--help"]).classification == ReadClassification.READ
    assert validate_read_argv("vastai", ["show", "machines"]).classification == ReadClassification.READ
    assert validate_read_argv("unknown-cli", ["--help"], help_only=True).classification == ReadClassification.READ


def test_shell_and_mutations_never_auto_execute():
    assert validate_read_argv("vastai", ["show", "machines", "|", "cat"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("bash", ["-c", "id"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("sh", ["-c", "id"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("vastai", ["destroy", "instance", "1"]).classification == ReadClassification.MUTATION
    assert validate_read_argv("systemctl", ["restart", "vastai.service"]).classification == ReadClassification.MUTATION
    assert validate_read_argv("vastai", ["show", "$HOME"]).classification == ReadClassification.BLOCKED


def test_unknown_and_secret_reads_fail_closed():
    assert validate_read_argv("some-cli", ["frobnicate"]).classification == ReadClassification.UNCERTAIN
    assert validate_read_argv("cat", ["~/.ssh/id_rsa"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("printenv", []).classification == ReadClassification.BLOCKED


def test_output_is_bounded_clean_and_redacted():
    value = "token=abc\x00\n" + "x\n" * 300
    result = sanitize_output(value, ["abc"], max_bytes=100, max_lines=5)
    assert "abc" not in result
    assert "\x00" not in result
    assert len(result.encode()) <= 115


def test_validator_is_an_execution_boundary():
    remote = RecordingExecutor()
    blocked = run_validated_read(remote, host(), "systemctl", ["restart", "vastai.service"])
    assert blocked["status"] == "not_executed"
    assert blocked["proposal_required"] is True
    assert remote.calls == []

    result = run_validated_read(remote, host(), "vastai", ["show", "machines"])
    assert result["status"] == "completed"
    assert remote.calls == [("x570", ("vastai", "show", "machines"), 15)]
