from __future__ import annotations

import re

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    PackageInstallParameters,
    RebootParameters,
    ServiceParameters,
    VMParameters,
)
from vast_agent.config import HostRegistry


class InvalidPackageNameError(ValueError):
    """An explicit install target was present, but failed the package schema."""


class ActionIntentResolver:
    """Deterministic Japanese action parser. It never asks an LLM to select a target."""

    def __init__(self, hosts: HostRegistry) -> None:
        self.hosts = hosts

    def resolve(self, text: str, context_host: str | None = None) -> ActionRequest | None:
        # Advisory questions remain investigation-only.
        if any(word in text.casefold() for word in (
            "すべき", "必要？", "必要?", "した方", "でしょう", "ですか", "？", "?",
        )):
            return None
        try: host = self.hosts.resolve_in_text(text)
        except KeyError:
            if context_host and re.fullmatch(r"\s*C\.[0-9]+\s*(?:を)?再起動して\s*", text):
                host = self.hosts.resolve(context_host)
            else: return None
        if any(word in text.casefold() for word in (
            "入れて", "入れといて", "install", "インストール",
        )):
            package_name = self._explicit_package(text, host)
            if package_name is not None:
                try:
                    parameters = PackageInstallParameters(
                        package_name=package_name,
                        expected_capability=None,
                        reason="explicit user request",
                    )
                except ValueError as exc:
                    raise InvalidPackageNameError(package_name) from exc
                return ActionRequest(
                    host=host.name, action_type=ActionType.PACKAGE_INSTALL,
                    parameters=parameters,
                )
            return None
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

    @staticmethod
    def _explicit_package(text: str, host) -> str | None:
        """Extract the exact user-authored token adjacent to an install expression.

        The host is deliberately required in the current message by ``resolve``.  This
        parser does not search a package catalog, infer a recommendation, or correct a
        spelling.  The typed parameter model remains the authority for valid syntax.
        """
        folded = text.casefold()
        matches = [
            (folded.find(name.casefold()), name)
            for name in (host.name, *host.aliases)
            if folded.find(name.casefold()) >= 0
        ]
        if not matches:
            return None
        host_at, matched_name = min(matches, key=lambda item: item[0])
        tail = text[host_at + len(matched_name):].strip()
        tail = re.sub(r"^(?:に|へ|で)\s*", "", tail)
        japanese = re.search(r"(?:入れて(?:おいて)?|入れといて|インストール(?:して)?)", tail, re.I)
        if japanese:
            value = tail[:japanese.start()].strip()
            value = re.sub(r"(?:が)?(?:なければ|無ければ|入ってないなら|必要なら)$", "", value)
            value = re.sub(r"を$", "", value).strip()
            return value if re.search(r"[a-z0-9]", value) else None
        english = re.search(r"(?<![a-z0-9])install(?![a-z0-9])", tail, re.I)
        if english:
            before = tail[:english.start()].strip().removesuffix("を").strip()
            after = tail[english.end():].strip()
            value = before or after
            return value if value and re.search(r"[a-z0-9]", value, re.I) else None
        return None
