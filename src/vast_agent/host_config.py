from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml
from yaml.nodes import MappingNode

from vast_agent.config import ConfigError, load_hosts


def set_expected_gpu_count(path: Path, host_name: str, count: int) -> None:
    """Update one host's expected GPU count without re-rendering the YAML file."""
    if count < 1:
        raise ConfigError("expected_gpu_count must be at least 1")

    original = path.read_text(encoding="utf-8")
    try:
        root = yaml.compose(original)
        hosts_node = _mapping_value(root, "hosts")
        host_node = _mapping_value(hosts_node, host_name)
        if not isinstance(host_node, MappingNode):
            raise TypeError(f"host '{host_name}' is not a mapping")
    except (yaml.YAMLError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Cannot safely update hosts configuration: {exc}") from exc

    count_pair = _optional_mapping_pair(host_node, "expected_gpu_count")
    if count_pair is not None:
        key_node, count_node = count_pair
        if count_node.start_mark.line != key_node.start_mark.line:
            raise ConfigError("Cannot safely replace an aliased or multiline expected_gpu_count")
        # Replacing only the scalar span retains indentation and any end-of-line comment.
        candidate = original[:count_node.start_mark.index] + str(count) + original[count_node.end_mark.index:]
    else:
        if host_node.flow_style:
            raise ConfigError("Cannot safely add expected_gpu_count to a flow-style host mapping")
        lines = original.splitlines(keepends=True)
        insert_at = host_node.end_mark.line
        indent = host_node.start_mark.column
        before = "".join(lines[:insert_at])
        separator = "" if not before or before.endswith(("\n", "\r")) else "\n"
        candidate = (
            before
            + separator
            + " " * indent
            + f"expected_gpu_count: {count}\n"
            + "".join(lines[insert_at:])
        )

    _atomic_validated_write(path, candidate, host_name, count)


def _mapping_value(node, key: str):
    value = _optional_mapping_value(node, key)
    if value is None:
        raise KeyError(key)
    return value


def _optional_mapping_value(node, key: str):
    pair = _optional_mapping_pair(node, key)
    return pair[1] if pair is not None else None


def _optional_mapping_pair(node, key: str):
    if not isinstance(node, MappingNode):
        raise TypeError(f"'{key}' parent is not a mapping")
    for key_node, value_node in node.value:
        if key_node.value == key:
            return key_node, value_node
    return None


def _atomic_validated_write(path: Path, candidate: str, host_name: str, count: int) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(candidate)
            handle.flush()
            os.fsync(handle.fileno())
        registry = load_hosts(temporary)
        if registry.hosts[host_name].expected_gpu_count != count:
            raise ConfigError("expected_gpu_count update validation failed")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
