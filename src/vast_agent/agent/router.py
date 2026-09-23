from __future__ import annotations

from vast_agent.agent.models import IntentDecision, Route
from vast_agent.config import HostRegistry

SCOPES = {
    "gpu温度": "gpu", "gpu状態": "gpu", "gpuの状態": "gpu", "gpu見て": "gpu",
    "pci状態": "pci", "pciの状態": "pci", "pci見て": "pci",
    "docker状態": "docker", "dockerの状態": "docker", "docker見て": "docker",
    "vm状態": "vm", "vmの状態": "vm", "vm見て": "vm",
    "ディスク": "system", "system状態": "system", "システム状態": "system",
    "vast状態": "vast", "vastの状態": "vast", "vast見て": "vast",
}
WRITE_WORDS = ("再起動", "restart", "reset", "リセット", "停止", "止めて", "stop", "reboot", "shutdown", "クロック制限", "絞って", "clock lock")
ADVICE_WORDS = ("すべき", "必要", "候補", "した方が", "でしょう", "ですか", "？", "?")
AGENT_WORDS = ("おかしく", "原因", "調べ", "なぜ", "なんで", "前にも", "過去", "消え")


def route_intent(text: str, registry: HostRegistry) -> IntentDecision:
    folded = text.casefold()
    write_mentioned = any(word in folded for word in WRITE_WORDS)
    if write_mentioned and not any(word in folded for word in ADVICE_WORDS):
        return IntentDecision(route=Route.UNSUPPORTED_WRITE, reason="write operation requested")
    if "ホスト一覧" in folded or "host list" in folded:
        return IntentDecision(route=Route.DETERMINISTIC, action="list_hosts")
    try:
        host = registry.resolve_in_text(text)
    except KeyError:
        host = None
    # Investigation language is host-specific only when a configured host is explicit.
    # Generic requests such as "最新情報を調べて" belong to the general READ tool loop.
    if write_mentioned or (host is not None and any(word in folded for word in AGENT_WORDS)):
        return IntentDecision(route=Route.AGENT, host=host.name if host else None, action="investigate")
    for phrase, scope in SCOPES.items():
        if phrase in folded:
            return IntentDecision(route=Route.DETERMINISTIC, host=host.name if host else None,
                                  action="inspect", scope=scope)
    # Any non-write question with an explicit host is a read-only investigation.
    if host is not None:
        return IntentDecision(route=Route.AGENT, host=host.name, action="investigate")
    return IntentDecision(route=Route.UNKNOWN)
