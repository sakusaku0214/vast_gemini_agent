from pathlib import Path

import pytest

from vast_agent.capabilities import (
    COMMAND_PROBES,
    SERVICE_PROBES,
    VAST_INSTALL_PROBE,
    apply_capabilities,
    detect_capabilities,
)
from vast_agent.config import ConfigError, load_hosts
from vast_agent.models.host import Capabilities, Host
from vast_agent.models.tool_result import ToolResult


def _host() -> Host:
    return Host(name="target", address="192.0.2.10", ssh_user="tester")


def _executor(results: dict[tuple[str, ...], ToolResult]):
    class ProbeExecutor:
        def execute(self, host, command, timeout, cancellation=None):
            return results.get(command, ToolResult(success=False, duration_ms=1))

    return ProbeExecutor()


@pytest.mark.parametrize(("results", "expected"), [
    (
        {
            **{probe: ToolResult(success=True, duration_ms=1)
               for probe in COMMAND_PROBES.values()},
            SERVICE_PROBES["vastai"]: ToolResult(
                success=True, stdout="loaded\n", duration_ms=1,
            ),
        },
        {"nvidia": True, "docker": True, "libvirt": True, "vast": True},
    ),
    (
        {
            COMMAND_PROBES["nvidia"]: ToolResult(success=True, duration_ms=1),
            COMMAND_PROBES["docker"]: ToolResult(success=True, duration_ms=1),
            COMMAND_PROBES["libvirt"]: ToolResult(success=False, duration_ms=1),
            SERVICE_PROBES["libvirtd"]: ToolResult(
                success=True, stdout="not-found\n", duration_ms=1,
            ),
            SERVICE_PROBES["vastai"]: ToolResult(
                success=True, stdout="loaded\n", duration_ms=1,
            ),
        },
        {"nvidia": True, "docker": True, "libvirt": False, "vast": True},
    ),
])
def test_detect_capabilities(results, expected):
    assert detect_capabilities(_host(), _executor(results)).model_dump() == expected


@pytest.mark.parametrize("load_state", ["not-found", "LoadState=not-found", "empty", "error", ""])
def test_detect_capabilities_rejects_nonexistent_service(load_state):
    results = {
        probe: ToolResult(success=False, duration_ms=1) for probe in COMMAND_PROBES.values()
    }
    results.update({
        probe: ToolResult(success=True, stdout=f"{load_state}\n", duration_ms=1)
        for probe in SERVICE_PROBES.values()
    })
    results[VAST_INSTALL_PROBE] = ToolResult(success=False, duration_ms=1)
    assert detect_capabilities(_host(), _executor(results)) == Capabilities()


def test_apply_updates_only_target_and_preserves_other_fields(tmp_path: Path):
    path = tmp_path / "hosts.yaml"
    path.write_text(
        "# keep this comment\nhosts:\n  target:\n    address: 192.0.2.10\n"
        "    ssh_user: tester\n    aliases: [box]\n    vast_id: 42\n    enabled: true\n"
        "    capabilities:\n      nvidia: false\n      docker: false\n"
        "  other:\n    address: 192.0.2.11\n    ssh_user: tester\n"
        "    capabilities:\n      nvidia: false\n",
        encoding="utf-8",
    )
    other_before = load_hosts(path).hosts["other"]
    apply_capabilities(path, "target", Capabilities(nvidia=True, docker=True, vast=True))
    text = path.read_text(encoding="utf-8")
    registry = load_hosts(path)
    target = registry.hosts["target"]
    assert "# keep this comment" in text
    assert target.address == "192.0.2.10"
    assert target.aliases == ["box"]
    assert target.vast_id == 42
    assert target.enabled is True
    assert target.capabilities == Capabilities(nvidia=True, docker=True, vast=True)
    assert registry.hosts["other"] == other_before


def test_apply_failure_leaves_original_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "hosts.yaml"
    original = "hosts:\n  target:\n    address: 192.0.2.10\n    ssh_user: tester\n"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr("vast_agent.capabilities.load_hosts", lambda _: (_ for _ in ()).throw(
        ConfigError("validation failed")
    ))
    with pytest.raises(ConfigError):
        apply_capabilities(path, "target", Capabilities(nvidia=True))
    assert path.read_text(encoding="utf-8") == original
