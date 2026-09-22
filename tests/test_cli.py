from vast_agent.app import main


def test_cli_install_doctor_hosts(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "doctor"]) in {0, 1}
    assert main(["--runtime", str(tmp_path), "hosts"]) == 0
    assert "Runtime initialized" in capsys.readouterr().out


def test_future_command_is_explicitly_unimplemented(tmp_path, capsys):
    assert main(["--runtime", str(tmp_path), "install"]) == 0
    assert main(["--runtime", str(tmp_path), "restart"]) == 3
    assert "not implemented" in capsys.readouterr().err
