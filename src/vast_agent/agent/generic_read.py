"""Fail-closed validation and execution for discovery-driven CLI READs.

The vocabulary below is a safety boundary for automatic execution, not a catalog
of what the Agent can understand or propose.  Unknown and mutating operations are
sent back to the operation-planning path rather than executed as READs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

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


_SHELL_EXECUTORS = frozenset({"sh", "bash", "dash", "zsh", "fish", "eval", "env", "printenv"})
_MUTATION_VERBS = frozenset({
    "create", "destroy", "delete", "remove", "set", "update", "edit", "write",
    "install", "uninstall", "restart", "stop", "start", "enable", "disable",
    "reset", "reboot", "shutdown", "kill", "attach", "detach", "mount", "umount",
    "chmod", "chown", "move", "mv", "rm", "dd", "mkfs", "publish", "send", "upload",
})
_READ_VERBS = frozenset({
    "help", "show", "list", "status", "get", "describe", "inspect", "query", "search",
    "info", "version", "check", "ps", "logs", "history", "devices", "machines", "offers",
})
_SECRET_MARKERS = (
    ".ssh", "id_rsa", "id_ed25519", "authorized_keys", "credentials", "credential",
    "token", "secret", "shadow", ".aws", ".config/gcloud", "keyring", "session",
)
_METACHARACTERS = re.compile(r"(?:\|\||&&|>>|[|<>;`$*?\[\]{}~]|[\r\n\x00])")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_read_argv(executable: str, argv: Sequence[str], *, help_only: bool = False) -> ReadValidation:
    """Classify an argv without ever invoking a shell.

    Help is universally discoverable for ordinary executables.  Other commands
    require a recognizable READ verb (or options-only invocation), so an unknown
    network CLI operation fails toward Proposal rather than automatic execution.
    """
    exe = executable.casefold()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}", executable):
        return ReadValidation(ReadClassification.BLOCKED, "INVALID_EXECUTABLE")
    if exe in _SHELL_EXECUTORS:
        return ReadValidation(ReadClassification.BLOCKED, "SHELL_OR_ENV_EXECUTOR")
    if len(argv) > 24 or any(not item or len(item) > 256 for item in argv):
        return ReadValidation(ReadClassification.BLOCKED, "ARGV_BOUNDS")
    if any(_METACHARACTERS.search(item) for item in argv):
        return ReadValidation(ReadClassification.BLOCKED, "SHELL_METACHARACTER")
    folded = tuple(item.casefold() for item in argv)
    if any(marker in item for item in folded for marker in _SECRET_MARKERS):
        return ReadValidation(ReadClassification.BLOCKED, "SENSITIVE_DATA_TARGET")
    if any(item in {"-c", "--command"} for item in folded):
        return ReadValidation(ReadClassification.BLOCKED, "COMMAND_INTERPRETER_OPTION")
    is_help = bool(folded) and any(item in {"-h", "--help", "help"} for item in folded)
    if help_only and not is_help:
        return ReadValidation(ReadClassification.BLOCKED, "HELP_FLAG_REQUIRED")
    if is_help:
        return ReadValidation(ReadClassification.READ, "CLI_HELP")
    words = tuple(item.lstrip("-").split("=", 1)[0] for item in folded if not item.startswith("/"))
    if any(word in _MUTATION_VERBS or word.startswith("mkfs.") for word in words):
        return ReadValidation(ReadClassification.MUTATION, "MUTATION_VERB")
    if exe == "cat":
        return ReadValidation(ReadClassification.UNCERTAIN, "GENERIC_FILE_READ_NOT_AUTO_APPROVED")
    if not folded or all(item.startswith("-") for item in folded):
        return ReadValidation(ReadClassification.READ, "OPTIONS_ONLY_READ")
    if any(word in _READ_VERBS for word in words):
        return ReadValidation(ReadClassification.READ, "RECOGNIZED_READ_VERB")
    return ReadValidation(ReadClassification.UNCERTAIN, "UNKNOWN_OPERATION")


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
                       help_only: bool = False) -> dict[str, object]:
    validation = validate_read_argv(executable, argv, help_only=help_only)
    if validation.classification != ReadClassification.READ:
        return {"status": "not_executed", "classification": validation.classification,
                "reason": validation.reason, "proposal_required": validation.classification in {
                    ReadClassification.MUTATION, ReadClassification.UNCERTAIN,
                }}
    result = remote.execute(host, (executable, *argv), timeout)
    return {
        "status": "completed" if result.success else "failed",
        "classification": "READ", "validation": validation.reason,
        "executable": executable, "argv": list(argv), "exit_code": result.exit_code,
        "stdout": sanitize_output(result.stdout, secrets),
        "stderr": sanitize_output(result.stderr, secrets), "timed_out": result.timed_out,
    }
