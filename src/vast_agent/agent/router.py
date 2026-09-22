from __future__ import annotations

from vast_agent.agent.models import IntentDecision, Route
from vast_agent.config import HostRegistry

SCOPES = {
    "gpu温度": "gpu", "gpu状態": "gpu", "gpuの状態": "gpu",
    "pci状態": "pci", "pciの状態": "pci", "docker状態": "docker",
    "dockerの状態": "docker", "vm状態": "vm", "vmの状態": "vm",
    "ディスク": "system", "vast状態": "vast", "vastの状態": "vast",
}
WRITE_WORDS = ("再起動", "restart", "reset", "リセット", "停止", "stop", "reboot", "shutdown")
AGENT_WORDS = ("おかしく", "原因", "調べ", "なぜ", "なんで", "前にも", "過去", "消え")


def route_intent(text: str, registry: HostRegistry) -> IntentDecision:
    folded = text.casefold()
    if any(word in folded for word in WRITE_WORDS):
        return IntentDecision(route=Route.UNSUPPORTED_WRITE, reason="write operation requested")
    if "ホスト一覧" in folded or "host list" in folded:
        return IntentDecision(route=Route.DETERMINISTIC, action="list_hosts")
    try:
        host = registry.resolve_in_text(text)
    except KeyError:
        host = None
    if any(word in folded for word in AGENT_WORDS):
        return IntentDecision(route=Route.AGENT, host=host.name if host else None, action="investigate")
    for phrase, scope in SCOPES.items():
        if phrase in folded:
            return IntentDecision(route=Route.DETERMINISTIC, host=host.name if host else None,
                                  action="inspect", scope=scope)
    return IntentDecision(route=Route.UNKNOWN, host=host.name if host else None)
