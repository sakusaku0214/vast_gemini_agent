from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ProposalStatus,
    RebootParameters,
)
from vast_agent.actions.resolver import ActionIntentResolver
from vast_agent.agent.models import Route
from vast_agent.agent.router import route_intent
from vast_agent.config import HostRegistry
from vast_agent.conversation.state import ConversationState, ConversationStore
from vast_agent.execution.base import redact
from vast_agent.jobs.cancellation import current_cancellation
from vast_agent.jobs.locks import LockClass
from vast_agent.jobs.manager import JobManager
from vast_agent.services.inspection import InspectionService


@dataclass
class ServiceReply:
    text: str
    job_id: int | None = None
    ignored: bool = False
    proposal: object | None = None
    policy: str | None = None


class AgentService:
    """Shared natural-language application layer for CLI and Discord."""

    def __init__(self, registry: HostRegistry, inspection: InspectionService, executor,
                 jobs: JobManager, conversations: ConversationStore, agent=None,
                 max_parallel_hosts: int = 3, secrets: tuple[str, ...] = (),
                 remember_last_host: bool = True, actions=None) -> None:
        self.registry = registry; self.inspection = inspection; self.executor = executor
        self.jobs = jobs; self.conversations = conversations; self.agent = agent
        self.max_parallel_hosts = max_parallel_hosts
        self.secrets = secrets
        self.remember_last_host = remember_last_host
        self.actions = actions
        self.action_resolver = ActionIntentResolver(registry)

    async def handle_question(self, text: str, owner: int | str = "cli",
                              channel: int | str = "cli") -> ServiceReply:
        state = self.conversations.get(owner, channel)
        folded = text.casefold().strip()
        if folded == "!jobs" or "ジョブ見せ" in folded:
            lines = [f"#{j['id']} {j['status']:<11} {j['host'] or '-'} {j['kind']}"
                     for j in self.jobs.recent()]
            return ServiceReply("Jobs\n" + ("\n".join(lines) or "(none)"))
        if self._is_cancel_request(folded):
            explicit = re.search(r"#(\d+)", folded)
            target = int(explicit.group(1)) if explicit else state.last_job_id
            if target is None:
                running = self.jobs.running()
                if len(running) != 1:
                    candidates = ", ".join(f"#{j['id']}" for j in running) or "なし"
                    return ServiceReply(f"停止対象を指定してください。実行中: {candidates}")
                target = int(running[0]["id"])
            return ServiceReply(f"Job #{target}: {self.jobs.cancel(target)}", target)
        if self._is_fleet(folded) and any(word in folded for word in ("再起動", "restart", "reset", "リセット", "reboot")):
            return ServiceReply("fleet WRITEは禁止されています。対象hostを1台指定してください。")
        if self._is_fleet(folded):
            scope = "gpu" if "gpu" in folded else "vast" if "vast" in folded else "system"
            return await self._fleet(self._clean(text), scope, state, owner, channel)

        from_recommendation = folded in {"それやって", "それをやって"}
        if from_recommendation:
            if state.last_recommended_action == "HOST_REBOOT_CANDIDATE" and state.last_host:
                action = ActionRequest(
                    host=state.last_host, action_type=ActionType.HOST_REBOOT,
                    parameters=RebootParameters(assessment="HOST_REBOOT_CANDIDATE"),
                )
            else:
                return ServiceReply("対象actionが一意ではありません。hostと操作を指定してください。")
        else:
            action = self.action_resolver.resolve(text, state.last_host)

        if action is not None:
            if self.actions is None:
                return ServiceReply("WRITE operations are not configured.")
            if action.action_type == ActionType.HOST_REBOOT and not from_recommendation:
                if self.agent is None:
                    return ServiceReply("REBOOT_ASSESSMENT_UNAVAILABLE")
                assessment = await asyncio.to_thread(self.agent.investigate, action.host,
                                                     self._clean(text))
                if assessment.recommended_action != "HOST_REBOOT_CANDIDATE":
                    return ServiceReply(self._clean(self._format_investigation(action.host, assessment)))
            proposal = await asyncio.to_thread(self.actions.propose, action, str(owner))
            state.last_host = action.host
            self.conversations.save(owner, channel, state)
            disabled = "\n⛔ Operations disabled: approval cannot execute." if not self.actions.settings.enabled else ""
            policy = "BLOCK" if proposal.status == ProposalStatus.BLOCKED else "ALLOW"
            summary = self._proposal_text(proposal, policy) + disabled
            return ServiceReply(self._clean(summary), proposal=proposal, policy=policy)

        if ("rebootすべき" in folded or "再起動すべき" in folded) and self.agent is not None:
            host, choices = self._resolve_host(text, None, state)
            if host is None or choices:
                return ServiceReply("対象hostを1台指定してください。")
            assessment = await asyncio.to_thread(self.agent.investigate, host.name,
                                                 self._clean(text))
            state.last_host = host.name
            state.last_recommended_action = assessment.recommended_action
            self.conversations.save(owner, channel, state)
            return ServiceReply(self._clean(self._format_investigation(host.name, assessment)))

        decision = route_intent(text, self.registry)
        if decision.route == Route.UNSUPPORTED_WRITE:
            return ServiceReply("この操作はWRITE/Approval Phaseで対応予定")
        host, choices = self._resolve_host(text, decision.host, state)
        if choices:
            return ServiceReply("hostが曖昧です: " + ", ".join(choices))
        if host is None:
            return ServiceReply("対象hostを指定してください。")
        if not host.enabled:
            return ServiceReply(f"{host.name} はdisabledです。")
        if decision.route == Route.UNKNOWN:
            # Follow-up words preserve host but never bypass deterministic routing.
            if any(word in folded for word in ("pci", "ついでに", "それも", "さっき")):
                decision.route = Route.DETERMINISTIC; decision.scope = "pci" if "pci" in folded else state.last_scope
            else:
                return ServiceReply("調査内容を指定してください。")
        kind = "investigate" if decision.route == Route.AGENT else decision.scope or "inspect"
        safe_text = self._clean(text)
        job, token = self.jobs.create(kind, host.name, safe_text)
        state.last_host = host.name if self.remember_last_host else None
        state.last_job_id = job.id; state.last_scope = decision.scope
        self.conversations.save(owner, channel, state)
        self.jobs.start(job)
        lock_class = LockClass.HEAVY_READ if decision.route == Route.AGENT else LockClass.READ
        def work():
            context = current_cancellation.set(token)
            try:
                with self.jobs.locks.acquire(host.name, lock_class):
                    if token.cancelled: return "cancelled"
                    if decision.route == Route.AGENT:
                        if self.agent is None: return "Gemini unavailable: GEMINI_API_KEY is not configured."
                        result = self.agent.investigate(host.name, safe_text)
                        return self._clean(self._format_investigation(host.name, result))
                    record = self.inspection.inspect_and_record(host, self.executor, decision.scope, token)
                    return self._format_observation(record.observation)
            finally:
                current_cancellation.reset(context)
        try:
            summary = await asyncio.to_thread(work)
            summary = self._clean(summary)
            self.jobs.finish(job, summary)
        except Exception:
            summary = "調査を完了できませんでした。詳細はagent logを確認してください。"
            self.jobs.finish(job, summary, "SERVICE_ERROR")
        return ServiceReply(self._clean(f"Job #{job.id}\n{summary}"), job.id)

    @staticmethod
    def _is_cancel_request(text: str) -> bool:
        return (text == "止めて"
                or text.startswith("!cancel")
                or bool(re.search(r"#\d+\s*止めて", text))
                or bool(re.search(r"(?:今の調査|この調査|ジョブ)(?:を)?止めて", text)))

    def _clean(self, value: str) -> str:
        return redact(value, self.secrets)

    @staticmethod
    def _proposal_text(proposal, policy: str) -> str:
        state = proposal.preflight_summary
        return (f"⚠️ Proposal #{proposal.id}\nHost: {proposal.host}\n"
                f"Action: {proposal.action_type}\nRisk: {proposal.risk_class}\n"
                f"Expires: {proposal.expires_at.isoformat()}\nPolicy: {policy}\n"
                f"Preflight: SSH={'OK' if state.ssh_reachable else 'BLOCK'}, "
                f"sudo={'OK' if state.sudo_available else 'BLOCK'}, "
                f"target={state.current_state}, VM={'running' if state.running_vm else 'none'}, "
                f"workload={'active' if state.active_workload else 'none'}")

    def _resolve_host(self, text: str, routed: str | None, state: ConversationState):
        folded = text.casefold(); matches = []
        for host in self.registry.hosts.values():
            names = [host.name, *host.aliases]
            if any(name.casefold() in folded for name in names): matches.append(host)
        if len(matches) > 1: return None, [h.name for h in matches]
        if len(matches) == 1: return matches[0], []
        if routed:
            try: return self.registry.resolve(routed), []
            except KeyError: pass
        if (self.remember_last_host
                and any(word in folded for word in ("ついでに", "それも", "さっき", "必要", "すべき"))
                and state.last_host):
            try: return self.registry.resolve(state.last_host), []
            except KeyError: pass
        return None, []

    @staticmethod
    def _is_fleet(text: str) -> bool:
        return any(word in text for word in ("全台", "全マシン", "fleet "))

    async def _fleet(self, request: str, scope: str, state: ConversationState,
                     owner, channel) -> ServiceReply:
        job, token = self.jobs.create(f"fleet-{scope}", None, request)
        state.last_job_id = job.id; state.last_scope = scope
        self.conversations.save(owner, channel, state); self.jobs.start(job)
        semaphore = asyncio.Semaphore(self.max_parallel_hosts)
        async def one(host):
            async with semaphore:
                if token.cancelled: return f"{host.name:<20} CANCELLED"
                def inspect():
                    with self.jobs.locks.acquire(host.name, LockClass.HEAVY_READ):
                        if token.cancelled:
                            return None
                        return self.inspection.inspect_and_record(host, self.executor, scope, token).observation
                try:
                    observation = await asyncio.to_thread(inspect)
                except Exception:
                    return f"{host.name:<20} ERROR inspection failed"
                if observation is None: return f"{host.name:<20} CANCELLED"
                if token.cancelled: return f"{host.name:<20} CANCELLED"
                state_name = "OK" if observation.ssh_ok and not observation.signatures else "WARN"
                detail = self._fleet_detail(observation, scope)
                return f"{host.name:<20} {state_name:<5} {detail}"
        try:
            tasks = [asyncio.create_task(one(h)) for h in self.registry.hosts.values() if h.enabled]
            rows = await asyncio.gather(*tasks)
            summary = f"🖥 Fleet {scope.upper()}\n" + "\n".join(rows)
            error_code = "FLEET_PARTIAL_FAILURE" if any(" ERROR " in row for row in rows) else None
            self.jobs.finish(job, self._clean(summary), error_code)
        except asyncio.CancelledError:
            summary = "Fleet coordination failed. 詳細はagent logを確認してください。"
            self.jobs.finish(job, summary, "FLEET_COORDINATOR_ERROR")
            raise
        except Exception:
            summary = "Fleet coordination failed. 詳細はagent logを確認してください。"
            self.jobs.finish(job, summary, "FLEET_COORDINATOR_ERROR")
        return ServiceReply(self._clean(f"Job #{job.id}\n{summary}"), job.id)

    @staticmethod
    def _fleet_detail(observation, scope):
        if scope == "gpu":
            if not observation.gpu.nvml_ok: return "NVML unavailable"
            devices = observation.gpu.devices
            return ", ".join(f"{d.model or 'GPU'} {d.temperature_c}°C" for d in devices) or "no GPU"
        if scope == "vast":
            return f"vast={observation.services.get('vastai.service', 'unknown')}"
        return f"disk={observation.system.filesystem_max_percent}% failed={observation.system.failed_units}"

    @staticmethod
    def _format_observation(observation) -> str:
        gpu = observation.gpu; first = gpu.devices[0] if gpu.devices else None
        serious = {"NVIDIA_FALLEN_OFF_BUS", "GPU_STUCK_D3", "NVIDIA_GSP_FAILURE"}
        if not observation.ssh_ok or serious.intersection(observation.signatures): icon = "🔴"
        elif observation.signatures or not gpu.nvml_ok or not first: icon = "🟡"
        else: icon = "🟢"
        return (f"{icon} {observation.host}\nGPU: {first.model if first else 'not detected'}\n"
                f"Temp: {first.temperature_c if first else '-'}°C\n"
                f"Util: {first.utilization_percent if first else '-'}%\n"
                f"NVML: {'OK' if gpu.nvml_ok else 'unavailable'}")

    @staticmethod
    def _format_investigation(host, result) -> str:
        signatures = "\n".join(f"• {item}" for item in result.signatures) or "• none"
        return (f"🔎 {host} — 調査結果\n\n要約:\n{result.summary}\n\n検出:\n{signatures}\n\n"
                f"提案:\n{result.recommended_action}\n\n確度:\n{result.confidence}")
