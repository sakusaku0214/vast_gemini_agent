from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from vast_agent.execution.ssh import SSHExecutor, openssh_path
from vast_agent.models.tool_result import ToolResult
from vast_agent.trust import trust_host


def completed(returncode: int = 0, stdout: bytes | str = b"") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=b"")


def test_windows_known_hosts_path_is_forward_slashed_and_quoted(monkeypatch, host):
    popen = Mock()
    popen.return_value.communicate.return_value = (b"ok", b"")
    popen.return_value.returncode = 0
    monkeypatch.setattr("vast_agent.execution.ssh.subprocess.Popen", popen)
    monkeypatch.setattr("vast_agent.execution.ssh.sys.platform", "win32")
    executor = SSHExecutor(Path("C:/Users/GARAGE PC/AppData/Local/known_hosts"))
    executor.known_hosts = Path(r"C:\Users\GARAGE PC\AppData\Local\known_hosts")

    result = executor.execute(host, ["true"], 10)

    assert isinstance(result, ToolResult) and result.success
    argv = popen.call_args.args[0]
    assert 'UserKnownHostsFile="C:/Users/GARAGE PC/AppData/Local/known_hosts"' in argv
    assert popen.call_args.kwargs.get("shell") is not True


@pytest.mark.parametrize("value", ['bad\"path', "bad\n-oProxyCommand=evil", "bad\x00path"])
def test_openssh_path_rejects_option_injection(value):
    with pytest.raises(ValueError):
        openssh_path(Path(value), windows=True)


def test_posix_known_hosts_path_remains_compatible():
    assert openssh_path(Path("/home/agent/.ssh/known_hosts"), windows=False) == (
        "/home/agent/.ssh/known_hosts"
    )
    assert openssh_path(Path("/home/space user/known_hosts"), windows=False) == (
        '"/home/space user/known_hosts"'
    )


def test_trust_host_windows_fallback_requires_confirmation(monkeypatch, tmp_path, host):
    known_hosts = tmp_path / "known_hosts"
    candidate = f"{host.address} ssh-ed25519 AAAATEST\n"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "ssh-keygen" and argv[1] == "-F":
            return completed(1)
        if argv[0] == "ssh-keyscan":
            return completed(1)
        if argv[0] == "ssh-keygen" and argv[1] == "-lf":
            return completed(0, "256 SHA256:test host (ED25519)\n")
        raise AssertionError(argv)

    monkeypatch.setattr("vast_agent.trust.subprocess.run", fake_run)
    monkeypatch.setattr("vast_agent.trust.sys.platform", "win32")
    monkeypatch.setattr("vast_agent.trust._scan_with_windows_ssh", lambda _: candidate)
    monkeypatch.setattr("builtins.input", lambda _: "NO")

    ok, message = trust_host(host, known_hosts)

    assert not ok and "cancelled" in message.lower()
    assert known_hosts.read_text(encoding="utf-8") == ""
    assert any(argv[0] == "ssh-keyscan" for argv in calls)


def test_windows_fallback_disables_auth_and_uses_disposable_file(monkeypatch, host):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        option = next(item for item in argv if item.startswith("UserKnownHostsFile="))
        path = option.split("=", 1)[1].strip('"')
        Path(path).write_text(f"{host.address} ssh-ed25519 AAAATEST\n", encoding="utf-8")
        return completed(255)

    monkeypatch.setattr("vast_agent.trust.subprocess.run", fake_run)
    from vast_agent.trust import _scan_with_windows_ssh

    assert "AAAATEST" in _scan_with_windows_ssh(host)
    argv = captured["argv"]
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "PubkeyAuthentication=no" in argv
    assert "PasswordAuthentication=no" in argv
    assert "KbdInteractiveAuthentication=no" in argv
    assert "PreferredAuthentications=none" in argv
    assert captured["kwargs"].get("shell") is None


def test_confirmed_fallback_candidate_is_written(monkeypatch, tmp_path, host):
    known_hosts = tmp_path / "known_hosts"
    candidate = f"{host.address} ssh-ed25519 AAAATEST\n"

    def fake_run(argv, **kwargs):
        if argv[0] == "ssh-keygen" and argv[1] == "-F":
            return completed(1)
        if argv[0] == "ssh-keyscan":
            return completed(1)
        return completed(0, "256 SHA256:test host (ED25519)\n")

    monkeypatch.setattr("vast_agent.trust.subprocess.run", fake_run)
    monkeypatch.setattr("vast_agent.trust.sys.platform", "win32")
    monkeypatch.setattr("vast_agent.trust._scan_with_windows_ssh", lambda _: candidate)
    monkeypatch.setattr("builtins.input", lambda _: "YES")

    assert trust_host(host, known_hosts) == (True, "Host keys registered")
    assert known_hosts.read_text(encoding="utf-8") == candidate


def test_existing_key_refuses_before_scan_or_confirmation(monkeypatch, tmp_path, host):
    known_hosts = tmp_path / "known_hosts"
    run = Mock(return_value=completed(0, b"existing"))
    prompt = Mock()
    monkeypatch.setattr("vast_agent.trust.subprocess.run", run)
    monkeypatch.setattr("builtins.input", prompt)

    ok, message = trust_host(host, known_hosts)

    assert not ok and "already exists" in message
    assert run.call_count == 1
    prompt.assert_not_called()
