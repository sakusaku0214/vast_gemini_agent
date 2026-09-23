from __future__ import annotations

import re

from vast_agent.agent.models import IntentDecision, Route
from vast_agent.config import HostRegistry

# Deterministic zero-token fast paths only; never a capability or reasoning boundary.
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


def explicit_write_intent(text: str) -> bool:
    """Detect an explicit mutation request; this is application state, not model policy.

    Questions and recollections about commands remain READs.  A mixed request that asks
    for the command *and* imperatively requests the change is still a WRITE request.
    """
    folded = text.casefold().strip()
    if re.search(r"(?:たっけ|てたっけ|必要なら|なければ|無ければ|入ってないなら)", folded):
        return False
    mutation = any(word in folded for word in WRITE_WORDS) or bool(re.search(
        r"(?:開放|解除|無効|有効|動かし直|適用|変更|削除|入れて|落として|"
        r"\b(?:enable|disable|install|remove|start|kill)\b)", folded,
    ))
    if not mutation:
        return False
    imperative = bool(re.search(
        r"(?:して(?:おいて|ください|くれ|ほしい)?|しといて|して$|"
        r"開放して|解除して|止めて|絞って|入れて|"
        r"\b(?:please\s+)?(?:restart|reset|stop|reboot|shutdown|enable|disable|install|remove|start|kill)\b)",
        folded,
    ))
    advice_only = any(word in folded for word in ADVICE_WORDS)
    return imperative or not advice_only


def route_intent(text: str, registry: HostRegistry) -> IntentDecision:
    folded = text.casefold()
    write_mentioned = any(word in folded for word in WRITE_WORDS)
    if explicit_write_intent(text):
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
        # A fast path must be unambiguously a request to read.  Extra language such as
        # ``docker状態直して`` belongs to the Agent even though it contains a scope phrase.
        suffix = folded.split(phrase, 1)[1] if phrase in folded else None
        read_ending = suffix is not None and any(
            marker in suffix for marker in ("見て", "確認", "状況", "どう", "?", "？")
        )
        if suffix is not None and (
            re.fullmatch(r"\s*[?？。!！\s]*", suffix) or read_ending
        ) and not any(word in suffix for word in WRITE_WORDS):
            return IntentDecision(route=Route.DETERMINISTIC, host=host.name if host else None,
                                  action="inspect", scope=scope)
    # Any non-write question with an explicit host is a read-only investigation.
    if host is not None:
        return IntentDecision(route=Route.AGENT, host=host.name, action="investigate")
    return IntentDecision(route=Route.UNKNOWN)
