from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from vast_agent.models.host import Host
from vast_agent.paths import RuntimePaths


class ConfigError(ValueError):
    pass


class HostRegistry(BaseModel):
    hosts: dict[str, Host]

    def resolve(self, value: str) -> Host:
        if value in self.hosts:
            return self.hosts[value]
        matches = [h for h in self.hosts.values() if value in h.aliases]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(value)


def load_hosts(path: Path) -> HostRegistry:
    if not path.exists():
        return HostRegistry(hosts={})
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = raw.get("hosts", {})
        return HostRegistry(hosts={name: Host(name=name, **value) for name, value in values.items()})
    except (OSError, yaml.YAMLError, ValidationError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid hosts configuration: {exc}") from exc


def initialize_config(paths: RuntimePaths, examples: Path | None = None) -> None:
    paths.create()
    defaults = {
        paths.hosts_file: "hosts: {}\n",
        paths.agent_file: "ssh:\n  connect_timeout: 10\n  server_alive_interval: 5\n  server_alive_count_max: 2\n",
    }
    for target, content in defaults.items():
        if not target.exists():
            if examples and (examples / target.name.replace(".yaml", ".example.yaml")).exists():
                source = examples / target.name.replace(".yaml", ".example.yaml")
                content = source.read_text(encoding="utf-8")
            target.write_text(content, encoding="utf-8")
