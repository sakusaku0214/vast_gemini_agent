from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from vast_agent.models.host import Host
from vast_agent.paths import RuntimePaths


class ConfigError(ValueError):
    pass


class GeminiSettings(BaseModel):
    model: str = "gemini-3.8-flash"
    api_version: str = "v1"
    default_thinking_level: str = "low"
    investigate_thinking_level: str = "medium"
    allow_high_thinking: bool = True
    store_interactions: bool = False
    max_agent_steps: int = 4
    max_tool_calls: int = 6
    max_llm_calls: int = 4
    max_evidence_chars_per_tool: int = 2500
    max_history_incidents: int = 3
    agent_wall_time_seconds: int = 120


def load_gemini_settings(path: Path) -> GeminiSettings:
    if not path.exists():
        return GeminiSettings()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return GeminiSettings.model_validate(raw.get("gemini", {}))
    except (OSError, yaml.YAMLError, ValidationError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid agent configuration: {exc}") from exc


class HostRegistry(BaseModel):
    hosts: dict[str, Host]

    def resolve(self, value: str) -> Host:
        needle = value.casefold()
        matches = [h for h in self.hosts.values() if needle in {
            h.name.casefold(), *(a.casefold() for a in h.aliases),
            *((str(h.vast_id),) if h.vast_id is not None else ()),
        }]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(value)

    def resolve_in_text(self, text: str) -> Host:
        folded = text.casefold()
        matches = []
        for host in self.hosts.values():
            names = [host.name, *host.aliases]
            if host.vast_id is not None:
                names.append(str(host.vast_id))
            if any(name.casefold() in folded for name in names):
                matches.append(host)
        if len(matches) == 1:
            return matches[0]
        raise KeyError(text)


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
        paths.agent_file: "ssh:\n  connect_timeout: 10\n  server_alive_interval: 5\n  server_alive_count_max: 2\ngemini:\n  model: gemini-3.8-flash\n  api_version: v1\n  default_thinking_level: low\n  investigate_thinking_level: medium\n  allow_high_thinking: true\n  store_interactions: false\n",
    }
    for target, content in defaults.items():
        if not target.exists():
            if examples and (examples / target.name.replace(".yaml", ".example.yaml")).exists():
                source = examples / target.name.replace(".yaml", ".example.yaml")
                content = source.read_text(encoding="utf-8")
            target.write_text(content, encoding="utf-8")
