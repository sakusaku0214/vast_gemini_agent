from __future__ import annotations

import pytest

from vast_agent.app import _secrets
from vast_agent.paths import RuntimePaths

KEYS = (
    "DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID",
    "GEMINI_API_KEY", "SEARCH_API_KEY",
)


def write_secrets(paths: RuntimePaths, values: dict[str, str]) -> None:
    paths.secrets_file.parent.mkdir(parents=True, exist_ok=True)
    paths.secrets_file.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()), encoding="utf-8",
    )


@pytest.mark.parametrize("key", KEYS)
def test_runtime_secret_wins_over_legacy_environment(tmp_path, monkeypatch, key):
    paths = RuntimePaths(tmp_path)
    write_secrets(paths, {key: "runtime-value"})
    monkeypatch.setenv(key, "unrelated-global-value")
    assert _secrets(paths)[key] == "runtime-value"


@pytest.mark.parametrize("key", KEYS)
def test_agent_specific_environment_wins_over_runtime_file(tmp_path, monkeypatch, key):
    paths = RuntimePaths(tmp_path)
    write_secrets(paths, {key: "runtime-value"})
    monkeypatch.setenv(f"VAST_AGENT_{key}", "agent-specific-value")
    assert _secrets(paths)[key] == "agent-specific-value"


@pytest.mark.parametrize("key", KEYS)
def test_legacy_environment_remains_a_fallback(tmp_path, monkeypatch, key):
    paths = RuntimePaths(tmp_path)
    monkeypatch.setenv(key, "legacy-value")
    assert _secrets(paths)[key] == "legacy-value"
