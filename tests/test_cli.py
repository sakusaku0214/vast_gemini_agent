from vast_agent.app import main
from vast_agent.execution.base import FakeExecutor
from vast_agent.models.tool_result import ToolResult


def test_cli_install_doctor_hosts(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "doctor"]) in {0, 1}
    assert main(["--runtime", str(tmp_path), "hosts"]) == 0
    assert "Runtime initialized" in capsys.readouterr().out


def test_future_command_is_explicitly_unimplemented(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "restart"]) == 3
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
