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


class FailingExecutor(RecordingExecutor):
    def __init__(self, result):
        super().__init__()
        self.result = result

    def execute(self, host, command, timeout, cancellation=None):
        self.calls.append((host.name, tuple(command), timeout))
        return self.result


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


def test_unknown_safe_cli_is_not_catalog_gated_but_secrets_are_blocked():
    assert validate_read_argv("some-cli", ["frobnicate"]).classification == ReadClassification.READ
    assert validate_read_argv("cat", ["~/.ssh/id_rsa"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("printenv", []).classification == ReadClassification.BLOCKED
    assert validate_read_argv("ssh", ["other-host", "status"]).classification == ReadClassification.BLOCKED
    assert validate_read_argv("curl", ["https://example.invalid"]).classification == ReadClassification.BLOCKED


def test_unknown_harmless_read_executes_but_flag_driven_mutation_does_not():
    remote = RecordingExecutor()
    read = run_validated_read(remote, host(), "novel-cli", ["inspect", "--format=json"])
    mutation = run_validated_read(
        remote, host(), "novel-cli", ["inspect", "--output=/tmp/state.json"],
    )

    assert read["classification"] == ReadClassification.READ
    assert mutation["classification"] == ReadClassification.MUTATION
    assert remote.calls == [
        ("x570", ("novel-cli", "inspect", "--format=json"), 15),
    ]


@pytest.mark.parametrize(("executable", "argv"), [
    ("curl", ["-XPOST", "localhost:8080/state"]),
    ("curl", ["--data", "enabled=true", "localhost:8080/state"]),
    ("curl", ["-o", "/tmp/result", "localhost:8080/state"]),
    ("find", ["/tmp", "-exec", "touch", "/tmp/marker", "+"]),
    ("sed", ["-i", "s/a/b/", "/tmp/config"]),
    ("tar", ["-xf", "/tmp/archive.tar"]),
    ("python3", ["-c", "print('bypass')"]),
])
def test_flag_network_and_indirect_mutations_are_not_reads(executable, argv):
    remote = RecordingExecutor()
    result = run_validated_read(remote, host(), executable, argv)
    assert result["classification"] != ReadClassification.READ
    assert remote.calls == []


def test_output_is_bounded_clean_and_redacted():
    value = "token=abc\x00\n" + "x\n" * 300
    result = sanitize_output(value, ["abc"], max_bytes=100, max_lines=5)
    assert "abc" not in result
    assert "\x00" not in result
    assert len(result.encode()) <= 115


def test_output_strips_ansi_before_control_characters():
    assert sanitize_output("\x1b[40m\x1b[97mtext\x1b[0m") == "text"
    assert sanitize_output("[48;5;240m[97mtext[0m") == "text"


def test_validator_is_an_execution_boundary():
    remote = RecordingExecutor()
    blocked = run_validated_read(remote, host(), "systemctl", ["restart", "vastai.service"])
    assert blocked["status"] == "not_executed"
    assert blocked["proposal_required"] is True
    assert remote.calls == []

    result = run_validated_read(remote, host(), "vastai", ["show", "machines"])
    assert result["status"] == "completed"
    assert remote.calls == [("x570", ("vastai", "show", "machines"), 15)]


def test_sudo_read_is_typed_preflighted_and_not_a_write():
    remote = RecordingExecutor()
    result = run_validated_read(
        remote, host(), "crontab", ["-l"], requires_sudo=True,
    )
    assert result["classification"] == ReadClassification.READ
    assert result["requires_sudo"] is True
    assert remote.calls == [
        ("x570", ("sudo", "-n", "true"), 15),
        ("x570", ("sudo", "-n", "crontab", "-l"), 15),
    ]


@pytest.mark.parametrize(("result", "kind", "next_fragment"), [
    (ToolResult(success=False, exit_code=127, stderr="command not found", duration_ms=1),
     "command_not_found", "discover executable"),
    (ToolResult(success=False, exit_code=126, stderr="Permission denied", duration_ms=1),
     "permission_denied", "requires_sudo=true"),
    (ToolResult(success=False, exit_code=2, stderr="unknown option\nUsage: novel", duration_ms=1),
     "syntax_error", "--help"),
    (ToolResult(success=False, exit_code=1, stderr="No such file", duration_ms=1),
     "missing_file", "location"),
])
def test_read_failure_is_adaptive_evidence(result, kind, next_fragment):
    output = run_validated_read(FailingExecutor(result), host(), "novel-cli", ["observe"])
    assert output["classification"] == ReadClassification.READ
    assert output["failure_kind"] == kind
    assert next_fragment in output["suggested_next_read"]


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
    ("docker", ["container", "restart", "worker"], ReadClassification.MUTATION),
    ("virsh", ["list", "--all"], ReadClassification.READ),
    ("nvidia-smi", ["-q"], ReadClassification.READ),
    ("nvidia-smi", ["-q", "-d", "CLOCK"], ReadClassification.READ),
    ("nvidia-smi", ["-i", "0", "-q", "-d", "CLOCK"], ReadClassification.READ),
    ("crontab", ["-l"], ReadClassification.READ),
    ("crontab", ["-r"], ReadClassification.MUTATION),
    ("systemctl", ["restart", "vastai"], ReadClassification.MUTATION),
    ("nvidia-smi", ["--gpu-reset"], ReadClassification.MUTATION),
])
def test_executable_aware_diagnostic_classification(executable, argv, expected):
    assert validate_read_argv(executable, argv).classification == expected
