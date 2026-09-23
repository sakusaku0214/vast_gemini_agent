from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath


class ArgvRejected(ValueError):
    pass


SHELL_EXECUTABLES = {
    "sh", "bash", "dash", "zsh", "ksh", "fish",
    "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
}

SHELL_OPERATOR_TOKENS = {
    "|", "||", "&&", ";", ">", ">>", "<", "<<", "2>", "2>>", "&>",
}


@dataclass(frozen=True)
class ValidatedArgv:
    argv: tuple[str, ...]


def validate_argv(argv: Sequence[str]) -> ValidatedArgv:
    """Validate transport shape only; command meaning belongs to Gemini."""
    if not argv or len(argv) > 64:
        raise ArgvRejected("argv must contain 1..64 tokens")

    clean: list[str] = []
    for token in argv:
        if not isinstance(token, str) or token == "":
            raise ArgvRejected("argv tokens must be non-empty strings")
        if "\x00" in token or any(ord(ch) < 32 or ord(ch) == 127 for ch in token):
            raise ArgvRejected("argv contains NUL/control characters")
        if token in SHELL_OPERATOR_TOKENS:
            raise ArgvRejected("shell operators are not allowed")
        clean.append(token)

    executable = PurePosixPath(clean[0]).name.casefold()
    if executable == "sudo":
        if len(clean) < 3 or clean[1] != "-n":
            raise ArgvRejected("sudo requires exactly the non-interactive -n prefix")
        executable = PurePosixPath(clean[2]).name.casefold()

    if executable in SHELL_EXECUTABLES or executable == "eval":
        raise ArgvRejected("shell/eval execution is not allowed")

    return ValidatedArgv(tuple(clean))


def validate_read_argv(argv: Sequence[str]) -> ValidatedArgv:
    return validate_argv(argv)


def validate_write_argv(argv: Sequence[str]) -> ValidatedArgv:
    return validate_argv(argv)
