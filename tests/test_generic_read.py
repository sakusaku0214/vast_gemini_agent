import pytest

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
    assert validate_read_argv("vastai", ["show", "--help"], help_only=True).classification == ReadClassification.READ


def test_shell_and_mutations_never_auto_execute():
    assert validate_read_argv("vastai", ["show", "machines", "|", "cat"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("bash", ["-c", "id"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("sh", ["-c", "id"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("vastai", ["destroy", "instance", "1"]).classification == ReadClassification.MUTATION
    assert validate_read_argv("systemctl", ["restart", "vastai.service"]).classification == ReadClassification.MUTATION
    assert validate_read_argv("vastai", ["show", "$HOME"]).classification == ReadClassification.BLOCKED


@pytest.mark.parametrize(("executable", "argv"), [
    ("shutdown", ["-h", "now"]),
    ("nvidia-smi", ["--persistence-mode=1"]),
    ("nvidia-smi", ["-rgc"]),
    ("systemctl", ["restart", "vastai.service", "--help"]),
    ("vastai", ["destroy", "instance", "123", "--help"]),
])
def test_help_and_flag_forms_cannot_disguise_mutations(executable, argv):
    remote = RecordingExecutor()
    result = run_validated_read(remote, host(), executable, argv)
    assert result["classification"] != ReadClassification.READ
    assert remote.calls == []


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


@pytest.mark.parametrize(("executable", "argv", "expected"), [
    ("last", ["-x", "reboot", "shutdown"], ReadClassification.READ),
    ("journalctl", ["-b", "-1", "-n", "30", "--no-pager"], ReadClassification.READ),
    ("journalctl", ["--vacuum-time=1d"], ReadClassification.MUTATION),
    ("uptime", [], ReadClassification.READ),
    ("ps", ["-ef"], ReadClassification.READ),
    ("pgrep", ["-a", "vastai"], ReadClassification.READ),
    ("lspci", ["-nnk"], ReadClassification.READ),
    ("lsblk", ["-J"], ReadClassification.READ),
    ("uname", ["-r"], ReadClassification.READ),
    ("docker", ["ps", "-a"], ReadClassification.READ),
    ("virsh", ["list", "--all"], ReadClassification.READ),
    ("nvidia-smi", ["-q"], ReadClassification.READ),
    ("systemctl", ["restart", "vastai"], ReadClassification.MUTATION),
    ("nvidia-smi", ["--gpu-reset"], ReadClassification.MUTATION),
])
def test_executable_aware_diagnostic_classification(executable, argv, expected):
    assert validate_read_argv(executable, argv).classification == expected
