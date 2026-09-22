from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @classmethod
    def discover(cls, override: Path | None = None) -> RuntimePaths:
        if override is not None:
            return cls(Path(override))
        env = os.environ.get("VAST_AGENT_RUNTIME")
        if env:
            return cls(Path(env))
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / ".local" / "share"
        return cls(base / "VastGeminiAgent")

    @property
    def config_dir(self) -> Path: return self.root / "config"
    @property
    def hosts_file(self) -> Path: return self.config_dir / "hosts.yaml"
    @property
    def agent_file(self) -> Path: return self.config_dir / "agent.yaml"
    @property
    def secrets_file(self) -> Path: return self.root / "secrets" / "secrets.env"
    @property
    def known_hosts(self) -> Path: return self.root / "ssh" / "known_hosts"
    @property
    def database(self) -> Path: return self.root / "data" / "agent.db"
    @property
    def incident_logs(self) -> Path: return self.root / "logs" / "incidents"

    def create(self) -> None:
        for part in (
            "config", "secrets", "ssh", "data", "logs/agent", "logs/incidents",
            "backups", "state",
        ):
            (self.root / part).mkdir(parents=True, exist_ok=True)
        self.known_hosts.touch(exist_ok=True)
