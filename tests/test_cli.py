from vast_agent.app import main
from vast_agent.execution.base import FakeExecutor
from vast_agent.models.tool_result import ToolResult


def test_cli_install_doctor_hosts(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "doctor"]) in {0, 1}
    assert main(["--runtime", str(tmp_path), "hosts"]) == 0
    assert "Runtime initialized" in capsys.readouterr().out


def test_remaining_future_command_is_explicitly_unimplemented(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "backup"]) == 3
    assert "not implemented" in capsys.readouterr().err


def test_ask_deterministic_does_not_require_gemini(tmp_path, capsys, monkeypatch):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    (tmp_path / "config" / "hosts.yaml").write_text(
        "hosts:\n  test-host:\n    address: 192.0.2.20\n    ssh_user: tester\n"
        "    capabilities:\n      nvidia: true\n",
        encoding="utf-8",
    )
    fake = FakeExecutor({
        "host_ping": ToolResult(success=True, duration_ms=1),
        "get_gpu_status": ToolResult(success=True, stdout="0, GPU-x, Test, 42, 0, 0, 100, P8, 0000:01:00.0", duration_ms=1),
    })
    monkeypatch.setattr("vast_agent.app.SSHExecutor", lambda _: fake)
    assert main(["--runtime", str(tmp_path), "ask", "test-hostのGPU温度"]) == 0
    assert "42" in capsys.readouterr().out
    fake.calls.clear()
    assert main(["--runtime", str(tmp_path), "ask", "test-hostのディスク"]) == 0
    assert fake.calls == [
        "host_ping", "get_system_health", "get_d_state_processes", "get_service_status",
    ]


def test_disabled_host_is_rejected_by_ask_and_investigate(tmp_path, capsys, monkeypatch):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    (tmp_path / "config" / "hosts.yaml").write_text(
        "hosts:\n  disabled-host:\n    address: 192.0.2.30\n    ssh_user: tester\n"
        "    enabled: false\n    capabilities:\n      nvidia: true\n",
        encoding="utf-8",
    )
    fake = FakeExecutor({})
    monkeypatch.setattr("vast_agent.app.SSHExecutor", lambda _: fake)
    assert main(["--runtime", str(tmp_path), "ask", "disabled-hostのGPU温度"]) == 2
    assert main(["--runtime", str(tmp_path), "investigate", "disabled-host", "原因を調べて"]) == 2
    assert fake.calls == []
    assert capsys.readouterr().err.count("HOST_DISABLED") == 2


def test_gemini_check_uses_stateless_text_input(tmp_path, capsys, monkeypatch):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    captured = {}

    class FakeGemini:
        def __init__(self, api_key, api_version):
            captured["config"] = (api_key, api_version)

        def interact(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setattr("vast_agent.app.GoogleInteractionsClient", FakeGemini)
    assert main(["--runtime", str(tmp_path), "gemini-check"]) == 0
    assert captured["inputs"] == [{
        "type": "user_input", "content": [{"type": "text", "text": "Reply OK."}],
    }]
    assert captured["store"] is False
    assert "Interactions API OK" in capsys.readouterr().out


def test_start_respects_discord_disabled(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    (tmp_path / "config" / "agent.yaml").write_text(
        "discord:\n  enabled: false\n", encoding="utf-8",
    )
    assert main(["--runtime", str(tmp_path), "start"]) == 2
    assert "disabled" in capsys.readouterr().err


def test_restart_starts_after_successful_fallback_stop(tmp_path, monkeypatch):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    for key in ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID"):
        monkeypatch.setenv(key, "configured")
    calls = []

    class Runtime:
        def stop(self):
            calls.append("stop")
            return True, "stopped"

        def start(self):
            calls.append("start")
            return True, "started"

    monkeypatch.setattr("vast_agent.app.RuntimeManager", lambda _: Runtime())
    assert main(["--runtime", str(tmp_path), "restart"]) == 0
    assert calls == ["stop", "start"]


def test_restart_does_not_start_after_unsafe_identity_mismatch(tmp_path, monkeypatch):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    calls = []

    class Runtime:
        def stop(self):
            calls.append("stop")
            return False, "stale PID; refusing to terminate"

        def start(self):
            raise AssertionError("unsafe restart must not start a second process")

    monkeypatch.setattr("vast_agent.app.RuntimeManager", lambda _: Runtime())
    assert main(["--runtime", str(tmp_path), "restart"]) == 2
    assert calls == ["stop"]


def test_cli_gpu_includes_pci_while_pci_only_says_nvml_not_observed(
    tmp_path, capsys, monkeypatch,
):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    (tmp_path / "config" / "hosts.yaml").write_text(
        "hosts:\n  test-host:\n    address: 192.0.2.20\n    ssh_user: tester\n"
        "    capabilities:\n      nvidia: true\n",
        encoding="utf-8",
    )
    gpu = ToolResult(
        success=True,
        stdout="0, GPU-x, Test, 42, 0, 0, 100, P8, 0000:01:00.0",
        duration_ms=1,
    )
    pci = ToolResult(
        success=True,
        stdout=("0000:01:00.0 VGA compatible controller: NVIDIA Corporation Test "
                "[10de:0001]\n\tKernel driver in use: nvidia"),
        duration_ms=1,
    )
    fake = FakeExecutor({
        "host_ping": ToolResult(success=True, duration_ms=1),
        "get_gpu_status": gpu,
        "get_pci_status": pci,
    })
    monkeypatch.setattr("vast_agent.app.SSHExecutor", lambda _: fake)

    assert main(["--runtime", str(tmp_path), "inspect", "test-host", "--gpu"]) == 0
    assert "PCI=1, NVML=OK" in capsys.readouterr().out
    assert "get_pci_status" in fake.calls
    assert main(["--runtime", str(tmp_path), "inspect", "test-host", "--pci"]) == 0
    assert "PCI=1, NVML=not observed" in capsys.readouterr().out


def test_detect_capabilities_cli_applies_without_changing_operations(tmp_path, capsys, monkeypatch):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    hosts = tmp_path / "config" / "hosts.yaml"
    hosts.write_text(
        "hosts:\n  test-host:\n    address: 192.0.2.20\n    ssh_user: tester\n",
        encoding="utf-8",
    )
    fake = FakeExecutor({"host_ping": ToolResult(success=True, duration_ms=1)})
    monkeypatch.setattr("vast_agent.app.SSHExecutor", lambda _: fake)
    monkeypatch.setattr(
        "vast_agent.app.detect_capabilities",
        lambda host, executor: __import__("vast_agent.models.host", fromlist=["Capabilities"])
        .Capabilities(nvidia=True, docker=True, libvirt=False, vast=True),
    )
    agent_before = (tmp_path / "config" / "agent.yaml").read_text(encoding="utf-8")
    assert main([
        "--runtime", str(tmp_path), "detect-capabilities", "test-host", "--apply",
    ]) == 0
    output = capsys.readouterr().out
    assert "SSH: OK" in output
    assert "libvirt: false" in output
    assert "hosts.yaml updated" in output
    assert (tmp_path / "config" / "agent.yaml").read_text(encoding="utf-8") == agent_before
