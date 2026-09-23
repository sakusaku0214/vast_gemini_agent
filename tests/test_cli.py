from vast_agent.app import main
from vast_agent.services.agent_service import ServiceReply


def test_cli_install_doctor_hosts(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "doctor"]) in {0, 1}
    assert main(["--runtime", str(tmp_path), "hosts"]) == 0
    assert "Runtime initialized" in capsys.readouterr().out


def test_remaining_future_command_is_explicitly_unimplemented(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "backup"]) == 3
    assert "not implemented" in capsys.readouterr().err


def test_ask_delegates_natural_language_without_local_routing(tmp_path, capsys, monkeypatch):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    seen = {}

    class Service:
        async def handle_question(self, question):
            seen["question"] = question
            return ServiceReply("Gemini handled it", 1)

    monkeypatch.setattr("vast_agent.app._service", lambda *args: Service())
    request = "全台の今日の通信量をランキング付けて一覧にして"
    assert main(["--runtime", str(tmp_path), "ask", request]) == 0
    assert seen["question"] == request
    assert "Gemini handled it" in capsys.readouterr().out


def test_explicit_investigate_rejects_disabled_target(tmp_path, capsys, monkeypatch):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    (tmp_path / "config" / "hosts.yaml").write_text(
        "hosts:\n  disabled-host:\n    address: 192.0.2.30\n    ssh_user: tester\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    assert main([
        "--runtime", str(tmp_path), "investigate", "disabled-host", "原因を調べて"
    ]) == 2
    assert "HOST_DISABLED" in capsys.readouterr().err


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
    assert captured["inputs"] == [{"type": "text", "text": "Reply OK."}]
    assert captured["store"] is False
    assert "Interactions API OK" in capsys.readouterr().out


def test_start_respects_discord_disabled(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    (tmp_path / "config" / "agent.yaml").write_text(
        "discord:\n  enabled: false\n", encoding="utf-8",
    )
    assert main(["--runtime", str(tmp_path), "start"]) == 2
    assert "disabled" in capsys.readouterr().err
