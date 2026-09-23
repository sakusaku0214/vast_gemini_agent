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

MUTATING_EXECUTABLES = {
    "rm", "mv", "cp", "install", "tee", "truncate", "touch", "mkdir", "rmdir",
    "chmod", "chown", "chgrp", "ln", "dd", "mkfs", "fdisk", "parted",
    "mount", "umount", "reboot", "shutdown", "poweroff", "halt",
    "kill", "pkill", "killall",
}

SYSTEMCTL_WRITE = {
    "start", "stop", "restart", "reload", "enable", "disable", "mask", "unmask",
    "reboot", "poweroff", "halt", "kill", "set-property", "edit",
}

DOCKER_WRITE = {
    "run", "create", "start", "stop", "restart", "kill", "rm", "rmi", "pull",
    "push", "build", "commit", "exec", "cp", "rename", "update", "pause", "unpause",
}

NVIDIA_WRITE_PREFIXES = (
    "--gpu-reset", "--reset-gpu-clocks", "--reset-memory-clocks",
    "--power-limit", "--lock-gpu-clocks", "--lock-memory-clocks",
)
NVIDIA_WRITE_FLAGS = {"-r", "-rac", "-pl", "-lgc", "-lmc", "-rgc", "-rmc", "-pm"}

SHELL_OPERATOR_TOKENS = {
    "|", "||", "&&", ";", ">", ">>", "<", "<<", "2>", "2>>", "&>",
}

CREDENTIAL_MARKERS = (
    "/.ssh/", "/etc/shadow", "/etc/gshadow", "secrets.env", ".env",
    "id_rsa", "id_ed25519", "authorized_keys", "private_key",
)


@dataclass(frozen=True)
class ValidatedArgv:
    argv: tuple[str, ...]


def validate_read_argv(argv: Sequence[str]) -> ValidatedArgv:
    """Enforce only the hard READ boundary; do not interpret user intent."""
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

    executable_index = 0
    executable = PurePosixPath(clean[0]).name.casefold()
    if executable == "sudo":
        if len(clean) < 3 or clean[1] != "-n":
            raise ArgvRejected("sudo READ requires exactly the non-interactive -n prefix")
        executable_index = 2
        executable = PurePosixPath(clean[2]).name.casefold()

    if executable in SHELL_EXECUTABLES or executable == "eval":
        raise ArgvRejected("shell/eval execution is not allowed")
    if executable in MUTATING_EXECUTABLES:
        raise ArgvRejected("mutating executable is not allowed in READ")

    tail = [item.casefold() for item in clean[executable_index + 1:]]
    if executable == "systemctl" and any(item in SYSTEMCTL_WRITE for item in tail):
        raise ArgvRejected("mutating systemctl operation is not allowed in READ")
    if executable == "docker" and tail and tail[0] in DOCKER_WRITE:
        raise ArgvRejected("mutating docker operation is not allowed in READ")
    if executable == "nvidia-smi":
        lowered = tuple(tail)
        if any(item in NVIDIA_WRITE_FLAGS for item in lowered):
            raise ArgvRejected("mutating nvidia-smi flag is not allowed in READ")
        if any(any(item.startswith(prefix + "=") or item == prefix for prefix in NVIDIA_WRITE_PREFIXES)
               for item in lowered):
            raise ArgvRejected("mutating nvidia-smi flag is not allowed in READ")

    joined = "\n".join(item.casefold() for item in clean)
    if any(marker in joined for marker in CREDENTIAL_MARKERS):
        raise ArgvRejected("credential-bearing paths are not available to READ")

    return ValidatedArgv(tuple(clean))



def validate_write_argv(argv: Sequence[str]) -> ValidatedArgv:
    """Validate transport shape only. Human approval, not code, authorizes the WRITE."""
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

    executable_index = 0
    executable = PurePosixPath(clean[0]).name.casefold()
    if executable == "sudo":
        if len(clean) < 3 or clean[1] != "-n":
            raise ArgvRejected("sudo WRITE requires exactly the non-interactive -n prefix")
        executable_index = 2
        executable = PurePosixPath(clean[2]).name.casefold()

    if executable in SHELL_EXECUTABLES or executable == "eval":
        raise ArgvRejected("shell/eval execution is not allowed")

    return ValidatedArgv(tuple(clean))
