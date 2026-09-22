from pathlib import Path

import pytest

from vast_agent.app import main, parser
from vast_agent.config import ConfigError, load_hosts
from vast_agent.host_config import set_expected_gpu_count


def _valid_yaml(count_line: str = "") -> str:
    return (
        "# registry comment\nhosts:\n  target:\n"
        "    address: 192.0.2.10 # address comment\n"
        "    ssh_user: tester\n    aliases: [garage]\n    enabled: false\n"
        "    capabilities:\n      nvidia: true\n      docker: false\n"
        f"{count_line}"
        "  other:\n    address: 192.0.2.11\n    ssh_user: tester\n"
        "    expected_gpu_count: 4\n"
    )


def test_adds_count_and_preserves_other_hosts_and_fields(tmp_path: Path):
    path = tmp_path / "hosts.yaml"
    original = _valid_yaml()
    path.write_text(original, encoding="utf-8")
    other_before = original[original.index("  other:"):]

    set_expected_gpu_count(path, "target", 2)

    updated = path.read_text(encoding="utf-8")
    target = load_hosts(path).hosts["target"]
    assert target.expected_gpu_count == 2
    assert target.address == "192.0.2.10"
    assert target.aliases == ["garage"]
    assert target.enabled is False
    assert target.capabilities.nvidia is True
    assert updated[updated.index("  other:"):] == other_before
    assert "# registry comment" in updated
    assert "# address comment" in updated


def test_replaces_existing_count_while_preserving_inline_comment(tmp_path: Path):
    path = tmp_path / "hosts.yaml"
    path.write_text(_valid_yaml("    expected_gpu_count: 1 # keep count comment\n"), encoding="utf-8")

    set_expected_gpu_count(path, "target", 2)

    assert "expected_gpu_count: 2 # keep count comment" in path.read_text(encoding="utf-8")


def test_validation_failure_rolls_back(tmp_path: Path, monkeypatch):
    path = tmp_path / "hosts.yaml"
    original = _valid_yaml()
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "vast_agent.host_config.load_hosts",
        lambda _: (_ for _ in ()).throw(ConfigError("validation failed")),
    )

    with pytest.raises(ConfigError, match="validation failed"):
        set_expected_gpu_count(path, "target", 2)

    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_resolves_alias_without_ssh_and_leaves_agent_config_unchanged(
    tmp_path: Path, capsys, monkeypatch,
):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    hosts = tmp_path / "config" / "hosts.yaml"
    hosts.write_text(_valid_yaml(), encoding="utf-8")
    agent = tmp_path / "config" / "agent.yaml"
    agent_before = agent.read_bytes()
    monkeypatch.setattr(
        "vast_agent.app.SSHExecutor",
        lambda *_: pytest.fail("set-gpu-count must not construct an SSH executor"),
    )

    assert main(["--runtime", str(tmp_path), "set-gpu-count", "garage", "2"]) == 0

    assert load_hosts(hosts).hosts["target"].expected_gpu_count == 2
    assert agent.read_bytes() == agent_before
    assert capsys.readouterr().out.endswith(
        "target\nexpected_gpu_count: 2\nhosts.yaml updated\n"
    )


@pytest.mark.parametrize("value", ["0", "not-an-integer"])
def test_cli_rejects_invalid_count(value: str):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(["set-gpu-count", "target", value])
    assert exc.value.code == 2


def test_cli_rejects_unknown_host(tmp_path: Path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "set-gpu-count", "missing", "2"]) == 2
    assert "ERROR HOST_NOT_FOUND: missing" in capsys.readouterr().err
