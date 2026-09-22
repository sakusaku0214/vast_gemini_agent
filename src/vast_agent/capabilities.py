from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

from vast_agent.config import ConfigError, load_hosts
from vast_agent.execution.base import Executor
from vast_agent.models.host import Capabilities, Host

# Every probe is a code-owned, read-only argv. No caller-controlled command is accepted.
COMMAND_PROBES: dict[str, tuple[str, ...]] = {
    "nvidia": ("command", "-v", "nvidia-smi"),
    "docker": ("command", "-v", "docker"),
    "libvirt": ("command", "-v", "virsh"),
}
SERVICE_PROBES: dict[str, tuple[str, ...]] = {
    name: ("systemctl", "show", f"{name}.service", "--property=LoadState", "--value")
    for name in ("docker", "libvirtd", "vastai")
}
VAST_INSTALL_PROBE = ("test", "-e", "/var/lib/vastai_kaalia")


def detect_capabilities(host: Host, executor: Executor) -> Capabilities:
    command_exists = {
        name: executor.execute(host, command, 15).success
        for name, command in COMMAND_PROBES.items()
    }
    return Capabilities(
        nvidia=command_exists["nvidia"],
        docker=command_exists["docker"] or _service_exists(host, executor, "docker"),
        libvirt=command_exists["libvirt"] or _service_exists(host, executor, "libvirtd"),
        vast=_service_exists(host, executor, "vastai")
        or executor.execute(host, VAST_INSTALL_PROBE, 15).success,
    )


def _service_exists(host: Host, executor: Executor, service: str) -> bool:
    result = executor.execute(host, SERVICE_PROBES[service], 15)
    load_state = result.stdout.strip().casefold().removeprefix("loadstate=").strip()
    return result.success and load_state not in {"", "not-found", "empty", "error"}


def apply_capabilities(path: Path, host_name: str, capabilities: Capabilities) -> None:
    """Patch just one YAML mapping, validate it, then atomically replace the file."""
    original = path.read_text(encoding="utf-8")
    try:
        root = yaml.compose(original)
        host_node = _mapping_value(_mapping_value(root, "hosts"), host_name)
    except (yaml.YAMLError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Cannot safely update hosts configuration: {exc}") from exc

    values = capabilities.model_dump()
    rendered = "capabilities:\n" + "".join(
        f"  {name}: {'true' if value else 'false'}\n" for name, value in values.items()
    )
    capability_node = _optional_mapping_value(host_node, "capabilities")
    lines = original.splitlines(keepends=True)
    if capability_node is not None:
        start = capability_node.start_mark.line
        end = capability_node.end_mark.line
        indent = capability_node.start_mark.column
        key_line = start
        # The value mark begins after ``capabilities:`` for flow/empty mappings,
        # but on the first child for the usual block mapping.
        while key_line >= 0 and "capabilities:" not in lines[key_line]:
            key_line -= 1
        if key_line < 0:
            raise ConfigError("Cannot locate capabilities block")
        indent = len(lines[key_line]) - len(lines[key_line].lstrip())
        replacement = "".join(" " * indent + line for line in rendered.splitlines(keepends=True))
        candidate = "".join(lines[:key_line]) + replacement + "".join(lines[end:])
    else:
        insert_at = host_node.end_mark.line
        indent = host_node.start_mark.column
        replacement = "".join(" " * indent + line for line in rendered.splitlines(keepends=True))
        candidate = "".join(lines[:insert_at]) + replacement + "".join(lines[insert_at:])

    _atomic_validated_write(path, candidate, host_name, capabilities)


def _mapping_value(node, key: str):
    value = _optional_mapping_value(node, key)
    if value is None:
        raise KeyError(key)
    return value


def _optional_mapping_value(node, key: str):
    for key_node, value_node in node.value:
        if key_node.value == key:
            return value_node
    return None


def _atomic_validated_write(path: Path, candidate: str, host_name: str,
                            capabilities: Capabilities) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(candidate)
            handle.flush()
            os.fsync(handle.fileno())
        registry = load_hosts(temporary)
        if registry.hosts[host_name].capabilities != capabilities:
            raise ConfigError("Capability update validation failed")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
