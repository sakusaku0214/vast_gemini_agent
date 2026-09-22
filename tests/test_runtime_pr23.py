from __future__ import annotations

from types import SimpleNamespace

import pytest

from vast_agent.paths import RuntimePaths
from vast_agent.runtime import RuntimeManager


@pytest.mark.parametrize("signal_error", [OSError("bad parameter"), SystemError("kill failed")])
def test_windows_ctrl_break_failure_uses_graceful_taskkill_and_cleans_state(
    tmp_path, monkeypatch, signal_error,
):
    paths = RuntimePaths(tmp_path)
    paths.create()
    paths.process_state.write_text('{"pid": 123}', encoding="utf-8")
    runtime = RuntimeManager(paths)
    monkeypatch.setattr("vast_agent.runtime.os.name", "nt")
    monkeypatch.setattr("vast_agent.runtime.signal.CTRL_BREAK_EVENT", 1, raising=False)
    monkeypatch.setattr(runtime, "status", lambda: ("RUNNING", {"pid": 123}))
    monkeypatch.setattr("vast_agent.runtime.os.kill", lambda *_: (_ for _ in ()).throw(signal_error))
    monkeypatch.setattr(runtime, "_command_line", lambda _: None)
    monkeypatch.setattr(runtime, "_process_exists", lambda _: False)
    commands = []
    monkeypatch.setattr(
        "vast_agent.runtime.subprocess.run",
        lambda command, **_: commands.append(command) or SimpleNamespace(returncode=0),
    )

    assert runtime.stop() == (True, "stopped")
    assert commands == [["taskkill", "/PID", "123"]]
    assert not paths.process_state.exists()


def test_stale_live_pid_is_not_signalled_or_cleaned(tmp_path, monkeypatch):
    paths = RuntimePaths(tmp_path)
    paths.create()
    paths.process_state.write_text('{"pid": 123}', encoding="utf-8")
    runtime = RuntimeManager(paths)
    monkeypatch.setattr(runtime, "status", lambda: ("STOPPED / stale pid", {"pid": 123}))
    monkeypatch.setattr(runtime, "_process_exists", lambda _: True)
    monkeypatch.setattr(
        "vast_agent.runtime.os.kill",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not signal an unrelated process")),
    )

    ok, message = runtime.stop()
    assert not ok
    assert "refusing" in message
    assert paths.process_state.exists()


def test_dead_stale_pid_state_is_safely_removed(tmp_path, monkeypatch):
    paths = RuntimePaths(tmp_path)
    paths.create()
    paths.process_state.write_text('{"pid": 123}', encoding="utf-8")
    runtime = RuntimeManager(paths)
    monkeypatch.setattr(runtime, "status", lambda: ("STOPPED / stale pid", {"pid": 123}))
    monkeypatch.setattr(runtime, "_process_exists", lambda _: False)

    assert runtime.stop() == (True, "not running (stale state removed)")
    assert not paths.process_state.exists()


def test_windows_force_kill_is_only_used_after_marker_reverification(tmp_path, monkeypatch):
    paths = RuntimePaths(tmp_path)
    paths.create()
    runtime = RuntimeManager(paths)
    monkeypatch.setattr("vast_agent.runtime.os.name", "nt")
    monkeypatch.setattr("vast_agent.runtime.signal.CTRL_BREAK_EVENT", 1, raising=False)
    monkeypatch.setattr(runtime, "status", lambda: ("RUNNING", {"pid": 123}))
    monkeypatch.setattr("vast_agent.runtime.os.kill", lambda *_: None)
    monkeypatch.setattr(runtime, "_command_line", lambda _: "python vast-agent-run-discord")
    monkeypatch.setattr(runtime, "_process_exists", lambda _: False)
    commands = []
    monkeypatch.setattr(
        "vast_agent.runtime.subprocess.run",
        lambda command, **_: commands.append(command) or SimpleNamespace(returncode=0),
    )

    assert runtime.stop(grace=0) == (True, "stopped (forced)")
    assert commands == [["taskkill", "/F", "/PID", "123"]]


def test_identity_change_prevents_forced_kill(tmp_path, monkeypatch):
    runtime = RuntimeManager(RuntimePaths(tmp_path))
    monkeypatch.setattr(runtime, "status", lambda: ("RUNNING", {"pid": 123}))
    monkeypatch.setattr("vast_agent.runtime.os.kill", lambda *_: None)
    monkeypatch.setattr(runtime, "_command_line", lambda _: "unrelated-process")

    assert runtime.stop(grace=0) == (
        False, "process identity changed; refusing forced termination",
    )
