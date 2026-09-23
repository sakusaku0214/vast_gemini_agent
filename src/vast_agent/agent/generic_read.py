"""Risk validation and bounded execution for adaptive CLI READs.

Concrete argv is executable without catalog membership or prior discovery.  The
validator blocks known mutation, shell, and sensitive-data risks; command failure
is returned as evidence so the Agent can adapt its next READ.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from vast_agent.execution.base import Executor, redact
from vast_agent.models.host import Host


class ReadClassification(StrEnum):
    READ = "READ"
    MUTATION = "MUTATION"
    BLOCKED = "BLOCKED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class ReadValidation:
    classification: ReadClassification
    reason: str


_SHELL_EXECUTORS = frozenset({
    "sh", "bash", "dash", "zsh", "fish", "eval", "env", "printenv", "sudo",
    "python", "python3", "perl", "ruby", "node", "php", "awk", "xargs",
    "timeout", "nice", "nohup", "setsid", "stdbuf",
})
_REMOTE_EXECUTORS = frozenset({"ssh", "scp", "sftp", "nc", "netcat", "socat", "telnet"})
_MUTATION_EXECUTABLES = frozenset({
    "shutdown", "reboot", "poweroff", "halt", "rm", "dd", "mount", "umount", "chmod",
    "chown", "kill", "pkill", "killall", "apt", "apt-get", "dpkg", "cp", "mv",
    "mkdir", "touch", "truncate", "tee", "install", "wget",
})
_MUTATION_VERBS = frozenset({
    "create", "destroy", "delete", "remove", "set", "update", "edit", "write",
    "install", "uninstall", "restart", "stop", "start", "enable", "disable",
    "reset", "reboot", "shutdown", "kill", "attach", "detach", "mount", "umount",
    "chmod", "chown", "move", "mv", "rm", "dd", "mkfs", "publish", "send", "upload",
})
_SECRET_MARKERS = (
    ".ssh", "id_rsa", "id_ed25519", "authorized_keys", "credentials", "credential",
    "token", "secret", "shadow", ".aws", ".config/gcloud", "keyring", "session",
)
_FILE_WRITE_OPTIONS = frozenset({
    "append", "in-place", "output", "output-file", "save", "save-to", "tee",
    "write", "write-file",
})
_HTTP_MUTATION_OPTIONS = frozenset({
    "data", "data-ascii", "data-binary", "data-raw", "data-urlencode", "form",
    "form-string", "json", "upload-file",
})
_HTTP_METHODS = frozenset({"post", "put", "patch", "delete", "connect"})
_METACHARACTERS = re.compile(r"(?:\|\||&&|>>|[|<>;`$*?\[\]{}~]|[\r\n\x00])")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _executable_name(executable: str) -> str | None:
    """Return a safe basename for a command name or a validated absolute path."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}", executable):
        return executable.casefold()
    path = PurePosixPath(executable)
    if (path.is_absolute() and ".." not in path.parts and len(executable) <= 256
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}", path.name)
            and not _CONTROL.search(executable) and not _METACHARACTERS.search(executable)):
        return path.name.casefold()
    return None


def validate_read_argv(executable: str, argv: Sequence[str], *, help_only: bool = False) -> ReadValidation:
    """Classify an argv without ever invoking a shell.

    This is risk classification, not a command permission catalog. Unknown CLI
    vocabulary is allowed when no mutation, shell, or sensitive-data risk is found.
    """
    exe = _executable_name(executable)
    if exe is None:
        return ReadValidation(ReadClassification.BLOCKED, "INVALID_EXECUTABLE")
    if exe in _SHELL_EXECUTORS:
        return ReadValidation(ReadClassification.BLOCKED, "SHELL_OR_ENV_EXECUTOR")
    if exe in _REMOTE_EXECUTORS:
        return ReadValidation(ReadClassification.BLOCKED, "NON_LOCAL_EXECUTOR")
    if exe in _MUTATION_EXECUTABLES or exe.startswith("mkfs"):
        return ReadValidation(ReadClassification.MUTATION, "MUTATION_EXECUTABLE")
    if len(argv) > 24 or any(not item or len(item) > 256 for item in argv):
        return ReadValidation(ReadClassification.BLOCKED, "ARGV_BOUNDS")
    if any(_METACHARACTERS.search(item) for item in argv):
        return ReadValidation(ReadClassification.BLOCKED, "SHELL_METACHARACTER")
    if any(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", item) for item in argv):
        return ReadValidation(ReadClassification.BLOCKED, "NON_LOCAL_TARGET")
    folded = tuple(item.casefold() for item in argv)
    if any(marker in item for item in folded for marker in _SECRET_MARKERS):
        return ReadValidation(ReadClassification.BLOCKED, "SENSITIVE_DATA_TARGET")
    if exe == "find" and any(item in {"-exec", "-execdir", "-ok", "-okdir"} for item in folded):
        return ReadValidation(ReadClassification.BLOCKED, "INDIRECT_EXECUTION")
    words = tuple(item.lstrip("-").split("=", 1)[0] for item in folded if not item.startswith("/"))
    option_names = {
        item[2:].split("=", 1)[0] for item in folded if item.startswith("--")
    }
    if option_names & (_MUTATION_VERBS | _FILE_WRITE_OPTIONS):
        return ReadValidation(ReadClassification.MUTATION, "MUTATION_OPTION")
    if exe == "sed" and any(item == "-i" or item.startswith("-i") for item in folded):
        return ReadValidation(ReadClassification.MUTATION, "IN_PLACE_WRITE")
    if exe == "tar" and any(item.startswith("-") and "x" in item.lstrip("-") for item in folded):
        return ReadValidation(ReadClassification.MUTATION, "ARCHIVE_EXTRACTION")
    if exe in {"curl", "http", "https", "httpie"}:
        if folded and folded[0] in _HTTP_METHODS:
            return ReadValidation(ReadClassification.MUTATION, "NETWORK_MUTATION_METHOD")
        if exe in {"http", "https", "httpie"} and any(
            "=" in item and "==" not in item and not item.startswith("http") for item in argv
        ):
            return ReadValidation(ReadClassification.MUTATION, "NETWORK_MUTATION_FIELD")
        if option_names & _HTTP_MUTATION_OPTIONS or any(
            item in {"-d", "-F", "-T", "-o", "-O"} for item in argv
        ):
            return ReadValidation(ReadClassification.MUTATION, "NETWORK_MUTATION_OPTION")
        for index, item in enumerate(argv):
            lowered = item.casefold()
            if item.startswith("-X") and item[2:].casefold() in _HTTP_METHODS:
                return ReadValidation(ReadClassification.MUTATION, "NETWORK_MUTATION_METHOD")
            if item in {"-X", "--request"} and index + 1 < len(folded) \
                    and folded[index + 1] in _HTTP_METHODS:
                return ReadValidation(ReadClassification.MUTATION, "NETWORK_MUTATION_METHOD")
            if lowered.startswith("--request=") and lowered.split("=", 1)[1] in _HTTP_METHODS:
                return ReadValidation(ReadClassification.MUTATION, "NETWORK_MUTATION_METHOD")
    # Known mutation vocabulary wins even when followed by --help. This prevents
    # help-shaped requests from laundering a mutation through the READ boundary.
    # Only the command position is generic: later argv values can legitimately be
    # filters such as ``last -x reboot shutdown``. Executable-specific mutation
    # flags below are compact risk metadata, never a READ permission catalog.
    command_word = folded[0].lstrip("-").split("=", 1)[0] if folded else ""
    if command_word in _MUTATION_VERBS or command_word.startswith("mkfs."):
        return ReadValidation(ReadClassification.MUTATION, "MUTATION_VERB")
    if exe == "docker" and len(words) > 1 and words[0] in {"container", "image", "network", "volume"} \
            and words[1] in _MUTATION_VERBS:
        return ReadValidation(ReadClassification.MUTATION, "NESTED_MUTATION_VERB")
    if exe == "nvidia-smi" and any(
        item in {"-r", "-rgc", "-rmc", "--gpu-reset", "-pm", "--persistence-mode"}
        or item.startswith(("--lock-", "--reset-", "--power-limit", "--persistence-mode="))
        for item in folded
    ):
        return ReadValidation(ReadClassification.MUTATION, "NVIDIA_SMI_MUTATION")
    if exe == "journalctl" and any(item.startswith((
        "--vacuum-", "--rotate", "--flush", "--sync", "--relinquish-var",
        "--smart-relinquish-var",
    )) for item in folded):
        return ReadValidation(ReadClassification.MUTATION, "JOURNAL_MUTATION")
    if exe == "crontab" and folded != ("-l",):
        return ReadValidation(ReadClassification.MUTATION, "CRONTAB_MUTATION")
    # ``-h`` is deliberately not universal help (for example shutdown -h now).
    is_help = folded in (("--help",), ("help",)) or (
        len(folded) > 1 and folded[-1] == "--help"
    )
    if help_only and not is_help:
        return ReadValidation(ReadClassification.BLOCKED, "HELP_FLAG_REQUIRED")
    if is_help:
        return ReadValidation(ReadClassification.READ, "CLI_HELP")
    return ReadValidation(ReadClassification.READ, "NO_MUTATION_RISK_DETECTED")


def _failure_evidence(exit_code: int | None, stderr: str, timed_out: bool) -> tuple[str, str]:
    """Classify a failed READ into evidence and a useful next investigation step."""
    folded = stderr.casefold()
    if timed_out:
        return "timeout", "retry a narrower bounded READ"
    if exit_code == 127 or "command not found" in folded:
        return "command_not_found", "discover executable path or inspect installation evidence"
    if exit_code in {126} or "permission denied" in folded or "operation not permitted" in folded:
        return "permission_denied", "retry the same READ with requires_sudo=true when appropriate"
    if any(marker in folded for marker in ("usage:", "unknown option", "unrecognized", "invalid option")):
        return "syntax_error", "inspect bounded --help and retry a validated READ argv"
    if "no such file" in folded:
        return "missing_file", "inspect a bounded parent location or configuration reference"
    return "command_failed", "inspect the failure and choose the smallest useful follow-up READ"


def sanitize_output(value: str, secrets: Sequence[str] = (), *, max_bytes: int = 16_384,
                    max_lines: int = 200) -> str:
    cleaned = _CONTROL.sub("", redact(value, secrets))
    cleaned = "\n".join(cleaned.splitlines()[:max_lines])
    data = cleaned.encode("utf-8", errors="replace")
    if len(data) > max_bytes:
        cleaned = data[:max_bytes].decode("utf-8", errors="replace") + "…[truncated]"
    return cleaned


def run_validated_read(remote: Executor, host: Host, executable: str, argv: Sequence[str],
                       *, timeout: int = 15, secrets: Sequence[str] = (),
                       help_only: bool = False, requires_sudo: bool = False) -> dict[str, object]:
    validation = validate_read_argv(executable, argv, help_only=help_only)
    if validation.classification != ReadClassification.READ:
        return {"status": "not_executed", "classification": validation.classification,
                "reason": validation.reason, "proposal_required": validation.classification in {
                    ReadClassification.MUTATION, ReadClassification.UNCERTAIN,
                }}
    if requires_sudo:
        preflight = remote.execute(host, ("sudo", "-n", "true"), min(timeout, 15))
        if not preflight.success:
            return {"status": "failed", "classification": "READ", "validation": validation.reason,
                    "requires_sudo": True, "reason": "SUDO_NOT_AVAILABLE", "exit_code": preflight.exit_code,
                    "stdout": "", "stderr": sanitize_output(preflight.stderr, secrets),
                    "timed_out": preflight.timed_out}
    command = (("sudo", "-n", executable, *argv) if requires_sudo else (executable, *argv))
    result = remote.execute(host, command, timeout)
    stdout = sanitize_output(result.stdout, secrets)
    stderr = sanitize_output(result.stderr, secrets)
    failure_kind, next_read = (None, None)
    if not result.success:
        failure_kind, next_read = _failure_evidence(result.exit_code, stderr, result.timed_out)
    return {
        "status": "completed" if result.success else "failed",
        "classification": "READ", "validation": validation.reason,
        "executable": executable, "argv": list(argv), "requires_sudo": requires_sudo,
        "exit_code": result.exit_code,
        "stdout": stdout, "stderr": stderr, "timed_out": result.timed_out,
        "failure_kind": failure_kind, "suggested_next_read": next_read,
    }
