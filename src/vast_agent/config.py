from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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


class DiscordSettings(BaseModel):
    enabled: bool = True


class JobSettings(BaseModel):
    max_parallel_hosts: int = 3
    max_recent_jobs: int = 20


class ConversationSettings(BaseModel):
    remember_last_host: bool = True


class OperationsSettings(BaseModel):
    """Fail-closed settings for the state-changing action subsystem."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    approval_ttl_seconds: int = Field(default=600, ge=60, le=3600)
    action_timeout_seconds: int = Field(default=60, ge=5, le=600)
    reboot_recovery_timeout_seconds: int = Field(default=300, ge=10, le=3600)
    reboot_poll_seconds: int = Field(default=10, ge=1, le=60)
    enable_vms_script: str | None = None

    @field_validator("enable_vms_script")
    @classmethod
    def validate_vm_script(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "\x00" in value:
            raise ValueError("enable_vms_script contains NUL")
        path = Path(value)
        if not path.is_absolute() or path.name != "enable_vms.py" or "\\" in value:
            raise ValueError("enable_vms_script must be an absolute POSIX path named enable_vms.py")
        return value


def load_gemini_settings(path: Path) -> GeminiSettings:
    if not path.exists():
        return GeminiSettings()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return GeminiSettings.model_validate(raw.get("gemini", {}))
    except (OSError, yaml.YAMLError, ValidationError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid agent configuration: {exc}") from exc


def load_runtime_settings(path: Path) -> tuple[DiscordSettings, JobSettings, ConversationSettings]:
    if not path.exists():
        return DiscordSettings(), JobSettings(), ConversationSettings()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return (DiscordSettings.model_validate(raw.get("discord", {})),
                JobSettings.model_validate(raw.get("jobs", {})),
                ConversationSettings.model_validate(raw.get("conversation", {})))
    except (OSError, yaml.YAMLError, ValidationError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid agent configuration: {exc}") from exc


def load_operations_settings(path: Path) -> OperationsSettings:
    if not path.exists():
        return OperationsSettings()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return OperationsSettings.model_validate(raw.get("operations", {}))
    except (OSError, yaml.YAMLError, ValidationError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid operations configuration: {exc}") from exc


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
        paths.agent_file: "ssh:\n  connect_timeout: 10\n  server_alive_interval: 5\n  server_alive_count_max: 2\ngemini:\n  model: gemini-3.8-flash\n  api_version: v1\n  default_thinking_level: low\n  investigate_thinking_level: medium\n  allow_high_thinking: true\n  store_interactions: false\ndiscord:\n  enabled: true\njobs:\n  max_parallel_hosts: 3\n  max_recent_jobs: 20\nconversation:\n  remember_last_host: true\noperations:\n  enabled: false\n  approval_ttl_seconds: 600\n  action_timeout_seconds: 60\n  reboot_recovery_timeout_seconds: 300\n  reboot_poll_seconds: 10\n  enable_vms_script: null\n",
    }
    for target, content in defaults.items():
        if not target.exists():
            if examples and (examples / target.name.replace(".yaml", ".example.yaml")).exists():
                source = examples / target.name.replace(".yaml", ".example.yaml")
                content = source.read_text(encoding="utf-8")
            target.write_text(content, encoding="utf-8")
