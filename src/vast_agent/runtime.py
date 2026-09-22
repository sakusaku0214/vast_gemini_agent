from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

from vast_agent.paths import RuntimePaths


class RuntimeManager:
    MARKER = "vast-agent-run-discord"

    def __init__(self, paths: RuntimePaths) -> None:
        self.paths = paths

    def _state(self) -> dict[str, object] | None:
        try: return json.loads(self.paths.process_state.read_text(encoding="utf-8"))
        except (OSError, ValueError): return None

    @staticmethod
    def _command_line(pid: int) -> str | None:
        if os.name == "nt":
            command = ["powershell", "-NoProfile", "-Command",
                       f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"]
        else:
            command = ["ps", "-p", str(pid), "-o", "args="]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
            return result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            return None

    def status(self) -> tuple[str, dict[str, object] | None]:
        state = self._state()
        if not state: return "STOPPED", None
        pid = int(state.get("pid", 0)); command = self._command_line(pid)
        if command and self.MARKER in command: return "RUNNING", state
        return "STOPPED / stale pid", state

    @staticmethod
    def _process_exists(pid: int) -> bool:
        """Check liveness without signalling or otherwise modifying the process."""
        if os.name == "nt":
            command = [
                "powershell", "-NoProfile", "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
            ]
            try:
                return subprocess.run(
                    command, capture_output=True, timeout=5, check=False,
                ).returncode == 0
            except OSError:
                # Failure to establish absence must fail closed.
                return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except (OSError, PermissionError):
            return True
        return True

    def start(self) -> tuple[bool, str]:
        status, _ = self.status()
        if status == "RUNNING": return False, "already running"
        self.paths.create()
        log = self.paths.agent_logs / "agent.log"
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
        stream = log.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "vast_agent", "--runtime", str(self.paths.root),
             "run-discord", self.MARKER], stdout=stream, stderr=stream,
            stdin=subprocess.DEVNULL, creationflags=flags, start_new_session=os.name != "nt",
        )
        state = {"pid": proc.pid, "started_at": datetime.now(UTC).isoformat(), "version": "2.0.0"}
        self.paths.process_state.write_text(json.dumps(state), encoding="utf-8")
        time.sleep(.2)
        if proc.poll() is not None: return False, "process exited during startup; check agent log"
        return True, f"started PID {proc.pid}"

    def stop(self, grace: float = 8) -> tuple[bool, str]:
        status, state = self.status()
        if status != "RUNNING" or not state:
            if status == "STOPPED":
                return True, "not running"
            pid = int(state.get("pid", 0))
            if pid > 0 and not self._process_exists(pid):
                self.paths.process_state.unlink(missing_ok=True)
                return True, "not running (stale state removed)"
            return False, "stale PID; refusing to terminate"
        pid = int(state["pid"])
        try:
            if os.name == "nt":
                try:
                    os.kill(pid, signal.CTRL_BREAK_EVENT)
                except (OSError, SystemError):
                    # A detached process may not share a console. Identity was already verified;
                    # taskkill without /F is the safe graceful fallback.
                    subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True,
                                   timeout=5, check=False)
            else:
                os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if self._command_line(pid) is None and not self._process_exists(pid):
                    self.paths.process_state.unlink(missing_ok=True); return True, "stopped"
                time.sleep(.1)
            command = self._command_line(pid)
            if not command:
                if not self._process_exists(pid):
                    self.paths.process_state.unlink(missing_ok=True)
                    return True, "stopped"
                return False, "process identity unavailable; refusing forced termination"
            if self.MARKER not in command:
                return False, "process identity changed; refusing forced termination"
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True,
                               timeout=5, check=False)
            else:
                os.kill(pid, signal.SIGKILL)
            if not self._process_exists(pid):
                self.paths.process_state.unlink(missing_ok=True)
                return True, "stopped (forced)"
            return False, "verified process did not stop"
        except OSError:
            if not self._process_exists(pid):
                self.paths.process_state.unlink(missing_ok=True)
                return True, "stopped"
            return False, "verified process could not be stopped"
