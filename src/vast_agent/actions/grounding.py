from __future__ import annotations

import re

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    PackageInstallParameters,
    ServiceParameters,
    VMParameters,
)


def action_target_is_grounded(request: ActionRequest, message: str) -> bool:
    """Require every mutable target value to occur explicitly in this user turn."""
    params = request.parameters
    folded = message.casefold()
    if isinstance(params, PackageInstallParameters):
        escaped = re.escape(params.package_name)
        present = re.search(rf"(?<![a-z0-9+.-]){escaped}(?![a-z0-9+.-])", folded)
        return bool(present and params.expected_capability is None)
    if isinstance(params, ContainerParameters):
        return bool(re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(params.container)}(?![0-9])", message,
        ))
    if isinstance(params, GPUParameters):
        return bool(re.search(rf"GPU\s*{params.gpu_index}(?![0-9])", message, re.I))
    if isinstance(params, ServiceParameters):
        token = {
            ActionType.RESTART_VAST_SERVICE: "vast",
            ActionType.RESTART_DOCKER_SERVICE: "docker",
            ActionType.RESTART_LIBVIRT_SERVICE: "libvirt",
        }.get(request.action_type)
        return token is not None and token in folded
    if isinstance(params, VMParameters):
        return bool(re.search(rf"(?<![a-z0-9]){params.mode}(?![a-z0-9])", folded))
    if request.action_type == ActionType.HOST_REBOOT:
        # The host is grounded separately, but reboot intent must also be authored in this
        # turn.  Advice and questions stay on the existing assessment path and cannot be
        # promoted to a mutation merely because the model selected HOST_REBOOT.
        reboot_intent = bool(re.search(r"再起動|リブート|(?<![a-z0-9])reboot(?![a-z0-9])", folded))
        advice_or_question = any(marker in folded for marker in (
            "すべき", "必要", "した方", "でしょう", "ですか", "?", "？",
        ))
        return reboot_intent and not advice_or_question
    return False
