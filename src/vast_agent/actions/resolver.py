from __future__ import annotations

import re

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    RebootParameters,
    ServiceParameters,
    VMParameters,
)
from vast_agent.config import HostRegistry


class ActionIntentResolver:
    """Deterministic Japanese action parser. It never asks an LLM to select a target."""

    def __init__(self, hosts: HostRegistry) -> None:
        self.hosts = hosts

    def resolve(self, text: str, context_host: str | None = None) -> ActionRequest | None:
        # Advisory questions remain investigation-only.
        if any(word in text.casefold() for word in ("すべき", "必要？", "必要?")):
            return None
        try: host = self.hosts.resolve_in_text(text)
        except KeyError:
            if context_host and re.fullmatch(r"\s*C\.[0-9]+\s*(?:を)?再起動して\s*", text):
                host = self.hosts.resolve(context_host)
            else: return None
        container = re.search(r"(?<![A-Za-z0-9_])(C\.[0-9]+)(?![0-9])", text)
        if container and "再起動" in text:
            return ActionRequest(host=host.name, action_type=ActionType.RESTART_VAST_CONTAINER,
                                 parameters=ContainerParameters(container=container.group(1)))
        gpu = re.search(r"GPU\s*([0-9]+)\s*(?:を)?\s*(?:reset|リセット)", text, re.I)
        if gpu:
            return ActionRequest(host=host.name, action_type=ActionType.GPU_RESET,
                                 parameters=GPUParameters(gpu_index=int(gpu.group(1))))
        vm = re.search(r"VM\s*(?:を)?\s*(on|off)", text, re.I)
        if vm:
            mode = vm.group(1).lower()
            action = ActionType.VM_MODE_ENABLE if mode == "on" else ActionType.VM_MODE_DISABLE
            return ActionRequest(host=host.name, action_type=action, parameters=VMParameters(mode=mode))
        if "再起動" in text and re.search(r"vast(?:ai)?", text, re.I):
            return ActionRequest(host=host.name, action_type=ActionType.RESTART_VAST_SERVICE,
                                 parameters=ServiceParameters(service="vastai.service"))
        if "再起動" in text:
            return ActionRequest(host=host.name, action_type=ActionType.HOST_REBOOT,
                                 parameters=RebootParameters(assessment="HOST_REBOOT_CANDIDATE"))
        return None
