from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass

from vast_agent.actions.capability_bridge import (
    AcquisitionIntent,
    acquisition_intent,
    assess_gap,
    request_for_gap,
)
from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    PackageInstallParameters,
    ProposalStatus,
    RebootParameters,
)
from vast_agent.actions.operation_plan import render_operation_proposal
from vast_agent.actions.package_catalog import capability_definition
from vast_agent.actions.preflight import package_manager_state
from vast_agent.actions.resolver import ActionIntentResolver, InvalidPackageNameError
from vast_agent.agent.models import Route
from vast_agent.agent.router import explicit_write_intent, route_intent
from vast_agent.config import HostRegistry
from vast_agent.conversation.state import ConversationState, ConversationStore
from vast_agent.diagnostics.parser import summarize_docker_status, summarize_vm_status
from vast_agent.execution.base import redact
from vast_agent.jobs.cancellation import current_cancellation
from vast_agent.jobs.locks import LockAcquisitionCancelled, LockClass
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
        pending_goal = state.investigation_context.get("pending_write_goal")
        command_hint = self._is_command_hint(folded) and isinstance(pending_goal, str)
        write_requested = explicit_write_intent(text) or command_hint
        operation_text = (
            f"Pending requested operation: {pending_goal}\nUser command hint: {text}"
            if command_hint else text
        )
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
        if self._requests_previous_evidence(folded):
            retained = state.investigation_context.get("relevant_excerpt")
            if isinstance(retained, str) and retained.strip():
                return ServiceReply(self._render_evidence(retained))
        fleet_hosts = self._resolve_host_set(text, state)
        fleet_scope = self._is_explicit_fleet(folded) or (
            "全部" in folded and state.context_kind == "fleet"
        ) or len(fleet_hosts) > 1
        if fleet_hosts and fleet_scope:
            if "通信量" in folded or "トラフィック" in folded or "traffic" in folded:
                return await self._fleet_traffic(
                    self._clean(text), state, owner, channel, fleet_hosts,
                )
            if any(phrase in folded for phrase in ("gpu状態", "gpu温度", "vast状態")):
                scope = "gpu" if "gpu" in folded else "vast"
                return await self._fleet(self._clean(text), scope, state, owner, channel, fleet_hosts)
            return await self._fleet_adaptive(
                self._clean(text), state, owner, channel, fleet_hosts,
            )

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
            try:
                action = self.action_resolver.resolve(text, state.last_host)
            except InvalidPackageNameError:
                return ServiceReply("INVALID_PACKAGE_NAME: package名の形式が不正です。")

        if action is not None:
            if self.actions is None:
                return ServiceReply("WRITE operations are not configured.")
            if isinstance(action.parameters, PackageInstallParameters) and any(
                word in folded for word in ("なければ", "入ってないなら", "必要なら", "無ければ")
            ):
                state_check = await asyncio.to_thread(
                    self.actions.preflight.collect, self.registry.resolve(action.host), action,
                )
                if state_check.package_installed:
                    version = f" ({state_check.package_version})" if state_check.package_version else ""
                    return ServiceReply(
                        f"{action.parameters.package_name}{version} は既に導入済みです。Proposalは不要です。"
                    )
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
            decision = self.actions.policy.evaluate(
                action, proposal.preflight_summary, self.actions.settings,
            )
            summary = self._proposal_text(proposal, policy, decision.reasons) + disabled
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
        if decision.route == Route.UNSUPPORTED_WRITE or write_requested:
            return await self._propose_generic_operation(
                operation_text, owner, channel, state, user_text=text,
            )

        host, choices = self._resolve_host(text, decision.host, state)
        if choices:
            return ServiceReply("hostが曖昧です: " + ", ".join(choices))
        if decision.route == Route.UNKNOWN and host is None:
            if acquisition_intent(text) != AcquisitionIntent.READ_ONLY:
                return ServiceReply("WRITE対象のhostを明示してください。")
            if self._is_read_follow_up(folded):
                return ServiceReply("対象hostを指定してください。")
            if self.agent is None:
                return ServiceReply("Gemini unavailable. 一般質問に回答できません。")
            answer = await asyncio.to_thread(self.agent.answer_general, self._clean(text))
            # A general turn deliberately leaves last_host and last_scope intact.
            # It does, however, end the immediate high-confidence WRITE context.
            state.write_context_host = None
            self.conversations.save(owner, channel, state)
            return ServiceReply(self._clean(answer))
        if host is None:
            return ServiceReply("対象hostを指定してください。")
        if not host.enabled:
            return ServiceReply(f"{host.name} はdisabledです。")
        if decision.route == Route.UNKNOWN:
            # A fixed host is sufficient to investigate. Natural language does not need to
            # satisfy another intent gate; only the router's already-selected trivial reads
            # bypass the Agent.
            decision.route = Route.AGENT
        kind = "investigate" if decision.route == Route.AGENT else decision.scope or "inspect"
        safe_text = self._clean(text)
        job, token = self.jobs.create(kind, host.name, safe_text)
        state.last_host = host.name if self.remember_last_host else None
        state.last_job_id = job.id
        state.last_scope = decision.scope or ("investigation" if decision.route == Route.AGENT else None)
        state.write_context_host = host.name
        state.current_hosts = (host.name,)
        state.context_kind = "single"
        try:
            explicit_context = self.registry.resolve_in_text(text) == host
        except KeyError:
            explicit_context = False
        state.context_source = "explicit" if explicit_context else "conversation"
        self.conversations.save(owner, channel, state)
        self.jobs.start(job)
        lock_class = LockClass.HEAVY_READ if decision.route == Route.AGENT else LockClass.READ
        investigation_result = None
        def work():
            context = current_cancellation.set(token)
            try:
                with self.jobs.locks.acquire(host.name, lock_class, token):
                    if token.cancelled: return "cancelled"
                    if decision.route == Route.AGENT:
                        if self.agent is None: return "Gemini unavailable: GEMINI_API_KEY is not configured."
                        result = self._call_investigate(host.name, safe_text, state.investigation_context)
                        return result
                    record = self.inspection.inspect_and_record(host, self.executor, decision.scope, token)
                    return self._format_observation(record.observation, decision.scope)
            except LockAcquisitionCancelled:
                return "cancelled"
            finally:
                current_cancellation.reset(context)
        try:
            work_result = await asyncio.to_thread(work)
            if decision.route == Route.AGENT and not isinstance(work_result, str):
                investigation_result = work_result
                summary = self._format_investigation(
                    host.name, work_result,
                    include_evidence=self._requests_evidence_render(folded),
                )
            else:
                summary = work_result
            summary = self._clean(summary)
            self.jobs.finish(job, summary)
        except Exception:
            summary = "調査を完了できませんでした。詳細はagent logを確認してください。"
            self.jobs.finish(job, summary, "SERVICE_ERROR")
        if investigation_result is not None:
            state.investigation_context = self._compact_context(
                safe_text, (host.name,), investigation_result,
            )
            self.conversations.save(owner, channel, state)
        if (investigation_result is not None
                and acquisition_intent(text) == AcquisitionIntent.PROPOSE_IF_NEEDED):
            requests = [
                request_for_gap(host.name, gap, consent=True)
                for gap in getattr(investigation_result, "capability_gaps", [])
            ]
            requests = [request for request in requests if request is not None]
            # Minimum-capability and no-chaining rule: one unambiguous acquisition at most.
            if len(requests) == 1:
                if self.actions is None:
                    summary += "\n\nWRITE operations are not configured; Proposalは作成されません。"
                else:
                    proposal = await asyncio.to_thread(self.actions.propose, requests[0], str(owner))
                    policy = "BLOCK" if proposal.status == ProposalStatus.BLOCKED else "ALLOW"
                    rendered = self._proposal_text(proposal, policy)
                    return ServiceReply(
                        self._clean(f"Job #{job.id}\n{summary}\n\n{rendered}"),
                        job.id, proposal=proposal, policy=policy,
                    )
        if investigation_result is not None and (
            investigation_result.mutation_requested or write_requested
        ):
            return await self._proposal_from_investigation(
                text, owner, channel, state, host, investigation_result,
            )
        return ServiceReply(self._clean(f"Job #{job.id}\n{summary}"), job.id)

    def _call_investigate(self, host: str, goal: str, context: dict[str, object]):
        method = self.agent.investigate
        if "context" in inspect.signature(method).parameters:
            return method(host, goal, context=context)
        return method(host, goal)

    @staticmethod
    def _compact_context(goal: str, hosts: tuple[str, ...], result) -> dict[str, object]:
        """Persist conclusions and a small sanitized excerpt; never raw unbounded logs."""
        context = {
            "previous_goal": goal[:500],
            "hosts": list(hosts)[:16],
            "conclusion": result.summary[:1200],
            "key_findings": [str(item)[:300] for item in result.findings[:8]],
            "missing_evidence": [str(item)[:300] for item in result.missing_evidence[:6]],
            "confidence": result.confidence,
        }
        cached = getattr(result, "_context_evidence", {})
        if isinstance(cached, dict):
            excerpt = cached.get("relevant_excerpt")
            paths = cached.get("executable_paths")
            if isinstance(excerpt, str) and excerpt:
                context["relevant_excerpt"] = excerpt[:4000]
            if isinstance(paths, list):
                context["executable_paths"] = [str(path)[:256] for path in paths[:4]]
        return context

    async def _proposal_from_investigation(
        self, text, owner, channel, state, host, investigation,
    ) -> ServiceReply:
        """Turn the Agent's post-READ conclusion into one inert, approval-gated plan."""
        if self._is_explicit_fleet(text.casefold()):
            return ServiceReply("fleet WRITEは禁止されています。対象hostを1台指定してください。")
        planner = getattr(self.agent, "plan_operation", None) if self.agent else None
        if planner is None or self.actions is None:
            return ServiceReply("WRITE/Approval operation planning capability unavailable")
        try:
            explicit = self.registry.resolve_in_text(text)
        except KeyError:
            explicit = None
        host_source = "current message" if explicit is not None else "conversation context"
        plan = await asyncio.to_thread(
            planner, host.name, host_source, self._clean(text), investigation,
        )
        if plan is None:
            return ServiceReply("変更対象または値が曖昧です。対象hostと具体的targetを確認してください。")
        proposal = await asyncio.to_thread(self.actions.propose_operation, plan, str(owner))
        state.last_host = host.name
        state.write_context_host = None
        self.conversations.save(owner, channel, state)
        rendered = render_operation_proposal(plan, proposal.id, proposal.expires_at.isoformat())
        return ServiceReply(self._clean(rendered), proposal=proposal, policy="ALLOW")

    async def _propose_generic_operation(
        self, text, owner, channel, state, *, user_text: str | None = None,
    ) -> ServiceReply:
        routing_text = user_text or text
        write_host, host_source, choices = self._resolve_write_host(routing_text, state)
        if choices:
            return ServiceReply("hostが曖昧です: " + ", ".join(choices))
        if write_host is None:
            return ServiceReply("WRITE/Approval: 対象hostまたは具体的targetを確認してください。")
        planner = getattr(self.agent, "plan_operation", None) if self.agent else None
        if planner is None or self.actions is None:
            return ServiceReply("WRITE/Approval operation planning capability unavailable")
        investigation = await asyncio.to_thread(
            self._call_investigate, write_host.name, self._clean(text),
            state.investigation_context,
        )
        plan = await asyncio.to_thread(
            planner, write_host.name, host_source, self._clean(text), investigation,
        )
        if plan is None:
            state.investigation_context = self._compact_context(
                text, (write_host.name,), investigation,
            )
            state.investigation_context["pending_write_goal"] = text[:500]
            self.conversations.save(owner, channel, state)
            conclusion = str(investigation.summary).strip()
            detail = f"\n調査結果: {conclusion}" if conclusion else ""
            return ServiceReply(
                "operation classification or target grounding is uncertain; "
                f"変更対象または値が曖昧なため確認が必要です。{detail}",
            )
        proposal = await asyncio.to_thread(self.actions.propose_operation, plan, str(owner))
        state.last_host = write_host.name
        state.write_context_host = None
        state.investigation_context.pop("pending_write_goal", None)
        self.conversations.save(owner, channel, state)
        rendered = render_operation_proposal(plan, proposal.id, proposal.expires_at.isoformat())
        return ServiceReply(self._clean(rendered), proposal=proposal, policy="ALLOW")

    @staticmethod
    def _is_command_hint(text: str) -> bool:
        """A command-shaped follow-up refines, but never executes, a pending operation."""
        return bool(re.search(
            r"(?:^|\s)(?:sudo\s+)?[a-z0-9_.+/-]+(?:\s+-{1,2}[a-z0-9][a-z0-9-]*)+",
            text, re.I,
        )) and bool(re.search(r"だった|かな|かも|コマンド|\?|？", text, re.I))

    @staticmethod
    def _is_cancel_request(text: str) -> bool:
        return (text in {"中止", "キャンセル", "やめて", "止めて", "今のやつ止めて", "今の調査止めて"}
                or text.startswith("!cancel")
                or bool(re.search(r"#\d+\s*止めて", text))
                or bool(re.search(r"(?:今の調査|この調査|ジョブ)(?:を)?止めて", text)))

    @staticmethod
    def _requests_evidence_render(text: str) -> bool:
        """Recognize a request to render acquired evidence in this reply/channel."""
        return bool(re.search(
            r"一覧|内容|見せて|表示|そのまま|コマンド結果|貼って|流して|"
            r"(?:show|list|display|paste)(?:\s|$)", text, re.I,
        ))

    @classmethod
    def _requests_previous_evidence(cls, text: str) -> bool:
        """Identify a high-confidence request to re-render the immediately prior result.

        Display vocabulary alone is deliberately insufficient: it commonly modifies a
        new investigation goal (for example ``docker一覧見せて``). Reuse requires an
        anaphoric reference, or an otherwise subject-free request to deliver the result
        into the current conversation surface.
        """
        if not cls._requests_evidence_render(text):
            return False
        previous_reference = re.search(
            r"(?:それ|その(?:一覧|内容|結果)|さっきの|前の|直前の|もう一回|もう一度)", text,
        )
        if previous_reference:
            return True
        return bool(
            re.fullmatch(
                r"\s*(?:一覧|内容|結果)(?:を)?(?:discord|ここ|このチャンネル)(?:へ|に)?"
                r"(?:流して|流せない[?？]?|貼って|出して|表示して)\s*",
                text,
                re.I,
            )
        )

    @staticmethod
    def _render_evidence(evidence: str) -> str:
        bounded = evidence[:4000]
        suffix = "\n…[truncated]" if len(evidence) > len(bounded) else ""
        return f"取得済みのREAD結果です。\n```text\n{bounded}{suffix}\n```"

    def _clean(self, value: str) -> str:
        return redact(value, self.secrets)

    @staticmethod
    def _proposal_text(proposal, policy: str, reasons: list[str] | None = None) -> str:
        state = proposal.preflight_summary
        package = (f"Package: {proposal.parameters.package_name}\n"
                   if isinstance(proposal.parameters, PackageInstallParameters) else "")
        reason = (f"Reason: {proposal.parameters.reason}\n"
                  if isinstance(proposal.parameters, PackageInstallParameters) else "")
        package_preflight = ""
        if isinstance(proposal.parameters, PackageInstallParameters):
            package_preflight = (
                f", package={state.current_state}, candidate={state.package_candidate or 'none'}, "
                f"package-manager={package_manager_state(state.package_manager_busy)}"
            )
        target = getattr(proposal.parameters, "service", None) or getattr(
            proposal.parameters, "container", None,
        ) or (f"GPU {proposal.parameters.gpu_index}" if hasattr(proposal.parameters, "gpu_index")
              else getattr(proposal.parameters, "package_name", proposal.action_type))
        return (f"⚠️ Operation Proposal #{proposal.id}\nHost: {proposal.host}\n"
                f"Host source: current message\n"
                f"Action: {proposal.action_type}\nRisk: {proposal.risk_class}\n"
                f"Sudo: required\nTarget: {target}\n"
                f"{package}{reason}"
                f"Expires: {proposal.expires_at.isoformat()}\nPolicy: {policy}"
                f"{(' (' + ', '.join(reasons) + ')') if reasons else ''}\n"
                f"Preflight: SSH={'OK' if state.ssh_reachable else 'BLOCK'}, "
                f"sudo={'OK' if state.sudo_available else 'BLOCK'}, "
                f"target={state.current_state}, VM={'running' if state.running_vm else 'none'}, "
                f"workload={'active' if state.active_workload else 'none'}{package_preflight}\n"
                "Expected effect: perform only the displayed typed action\n"
                "Verification plan: re-read the action-specific target state\n"
                "Rollback available: no (a rollback requires a new Proposal)")

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
        # Scope resolution is independent of READ/WRITE interpretation. A preceding single-host
        # investigation establishes context for natural follow-ups; fleet never does.
        if (self.remember_last_host and state.last_host and not self._is_explicit_fleet(folded)
                and (self._has_host_relevance(folded) or self._is_read_follow_up(folded)
                     or (state.investigation_context
                         and re.fullmatch(r"\s*(?:つまり|要するに)[^\n]{0,20}", folded)))):
            try: return self.registry.resolve(state.last_host), []
            except KeyError: pass
        return None, []

    @staticmethod
    def _has_host_relevance(text: str) -> bool:
        """Recognize structural operational references, not READ/WRITE intent."""
        if re.search(r"gpu\s*[0-9]*|nvidia-smi|nvtop|vastai?|docker|pci|vm|vnstat", text, re.I):
            return True
        return any(term in text for term in (
            "状態", "二枚", "片方", "通信量", "温度", "ディスク", "サービス",
            "パッケージ", "実行ファイル", "ログ", "さっき", "このホスト", "このマシン",
        ))

    def _resolve_write_host(self, text: str, state: ConversationState):
        """Allow only high-confidence, immediately grounded single-host WRITE context."""
        folded = text.casefold()
        matches = [host for host in self.registry.hosts.values()
                   if any(name.casefold() in folded for name in (host.name, *host.aliases))]
        if len(matches) > 1:
            return None, None, [host.name for host in matches]
        if len(matches) == 1:
            return matches[0], "current message", []
        vague_only = folded.strip() in {"そっち", "それ", "あれ", "適当に", "そっち適当に絞って"}
        if (self.remember_last_host and state.last_host and state.last_scope
                and state.write_context_host == state.last_host
                and not vague_only and not self._is_explicit_fleet(folded)):
            try:
                return self.registry.resolve(state.last_host), "conversation context", []
            except KeyError:
                pass
        return None, None, []

    @staticmethod
    def _is_read_follow_up(text: str) -> bool:
        """Recognize anaphoric or abbreviated READ questions without guessing a host.

        The vocabulary is grouped by conversational role rather than being used as one routing
        keyword bag. An explicit host is resolved before this helper, and mutation language always
        disables inheritance even when a sentence also contains a READ-domain word.
        """
        mutation_terms = (
            "再起動", "restart", "reset", "リセット", "止めて", "停止", "stop",
            "入れて", "remove", "削除", "enable", "disable",
            "reboot", "shutdown", "kill",
        )
        if any(term in text for term in mutation_terms):
            return False

        # Definitions are general knowledge, not an implicit request to inspect the last host.
        if re.search(r"(?:って|とは)\s*(?:何|なに)", text):
            return False
        if any(marker in text for marker in ("最新", "ニュース", "公開情報", "リリース情報")):
            return False

        references = (
            "こいつ", "このマシン", "このホスト", "それ", "そいつ", "そっち",
            "ついでに", "続けて", "さっき", "じゃあ", "では",
        )
        if any(reference in text for reference in references):
            return True
        if re.fullmatch(r"\s*(?:ある|どう|生きてる)(?:の)?\s*[?？]?\s*", text):
            return True

        japanese_domains = (
            "状態", "ディスク", "ネットワーク", "ネット周り", "インターフェース",
            "ドライバ", "速度", "サービス", "パッケージ", "プロセス", "カーネル",
            "実行ファイル", "コマンド", "入ってる", "入れてた", "生きてる",
            "通信量", "トラフィック", "温度", "値", "結果", "エラー", "原因", "止まってる",
        )
        english_domains = re.search(
            r"(?<![a-z0-9_])(?:gpu|pci|vast|docker|vm|network|nic|lan|interface|driver|"
            r"speed|service|package|process|os|kernel|executable)(?![a-z0-9_])",
            text,
        )
        read_question = any(marker in text for marker in (
            "?", "？", "は", "見て", "調べ", "確認", "教えて", "どう", "ある", "何", "なに", "値",
        ))
        return read_question and (
            english_domains is not None or any(domain in text for domain in japanese_domains)
        )

    @staticmethod
    def _is_explicit_fleet(text: str) -> bool:
        return any(word in text for word in ("全台", "全マシン", "fleet"))

    def _resolve_host_set(self, text: str, state: ConversationState):
        """Resolve and freeze scope in application code; the model never selects addresses."""
        folded = text.casefold()
        enabled = [host for host in self.registry.hosts.values() if host.enabled]
        if self._is_explicit_fleet(folded):
            return tuple(enabled)
        if "全部" in folded and state.context_kind == "fleet":
            names = set(state.current_hosts)
            return tuple(host for host in enabled if host.name in names)
        if (state.context_kind == "fleet" and state.current_hosts
                and (self._has_host_relevance(folded) or self._is_read_follow_up(folded))):
            names = set(state.current_hosts)
            return tuple(host for host in enabled if host.name in names)
        matches = [
            host for host in enabled
            if any(name.casefold() in folded for name in (host.name, *host.aliases))
        ]
        return tuple(matches)

    async def _fleet_adaptive(self, request, state, owner, channel, hosts) -> ServiceReply:
        """Run the same safe InvestigationAgent independently across a fixed host set."""
        job, token = self.jobs.create("fleet-investigate", None, request)
        prior_context = dict(state.investigation_context)
        state.last_host = None
        state.write_context_host = None
        state.current_hosts = tuple(host.name for host in hosts)
        state.context_kind = "fleet"
        state.context_source = "explicit"
        state.last_job_id = job.id
        state.last_scope = "fleet"
        self.conversations.save(owner, channel, state)
        self.jobs.start(job)
        semaphore = asyncio.Semaphore(self.max_parallel_hosts)

        async def one(host):
            async with semaphore:
                if token.cancelled:
                    return host.name, None, "cancelled"
                def investigate_host():
                    context = current_cancellation.set(token)
                    try:
                        with self.jobs.locks.acquire(
                            host.name, LockClass.HEAVY_READ, token,
                        ):
                            if token.cancelled:
                                return None
                            return self._call_investigate(host.name, request, prior_context)
                    finally:
                        current_cancellation.reset(context)
                try:
                    return host.name, await asyncio.to_thread(investigate_host), None
                except LockAcquisitionCancelled:
                    return host.name, None, "cancelled"
                except Exception:
                    return host.name, None, "investigation failed"

        collected = await asyncio.gather(*(one(host) for host in hosts))
        results = {name: result for name, result, error in collected if result is not None}
        unavailable = {name: error for name, result, error in collected if result is None}
        mutation_requested = (
            any(result.mutation_requested for result in results.values())
            or route_intent(request, self.registry).route == Route.UNSUPPORTED_WRITE
            or acquisition_intent(request) != AcquisitionIntent.READ_ONLY
        )
        synthesizer = getattr(self.agent, "synthesize_fleet", None) if self.agent else None
        if synthesizer and results:
            summary = await asyncio.to_thread(
                synthesizer, request, tuple(host.name for host in hosts), results,
            )
        else:
            summary = "\n".join(
                f"- **{name}**: {result.summary}" for name, result in results.items()
            ) or "利用可能なhost結果がありません。"
        if unavailable:
            summary += "\n\nUnavailable\n" + "\n".join(
                f"- {name}: {error}" for name, error in unavailable.items()
            )
        if mutation_requested:
            summary += (
                "\n\nfleet WRITEは実行できません。必要なら対象hostを1台ずつ明示して、"
                "個別のProposalを作成してください。"
            )
        state.investigation_context = {
            "previous_goal": request[:500],
            "hosts": list(state.current_hosts),
            "conclusion": summary[:1200],
            "key_findings": [result.summary[:300] for result in results.values()][:8],
            "missing_evidence": list(unavailable)[:6],
        }
        self.conversations.save(owner, channel, state)
        self.jobs.finish(job, self._clean(summary), "FLEET_PARTIAL_FAILURE" if unavailable else None)
        return ServiceReply(self._clean(f"Job #{job.id}\n{summary}"), job.id)

    async def _fleet(self, request: str, scope: str, state: ConversationState,
                     owner, channel, hosts=None) -> ServiceReply:
        job, token = self.jobs.create(f"fleet-{scope}", None, request)
        # Fleet output is never high-confidence single-host context for a later WRITE.
        state.last_host = None; state.write_context_host = None
        state.current_hosts = tuple(host.name for host in (hosts or ()))
        state.context_kind = "fleet"
        state.context_source = "explicit"
        state.last_job_id = job.id; state.last_scope = scope
        self.conversations.save(owner, channel, state); self.jobs.start(job)
        semaphore = asyncio.Semaphore(self.max_parallel_hosts)
        async def one(host):
            async with semaphore:
                if token.cancelled: return f"{host.name:<20} CANCELLED"
                def inspect():
                    with self.jobs.locks.acquire(host.name, LockClass.HEAVY_READ, token):
                        if token.cancelled:
                            return None
                        return self.inspection.inspect_and_record(host, self.executor, scope, token).observation
                try:
                    observation = await asyncio.to_thread(inspect)
                except LockAcquisitionCancelled:
                    return f"{host.name:<20} CANCELLED"
                except Exception:
                    return f"{host.name:<20} ERROR inspection failed"
                if observation is None: return f"{host.name:<20} CANCELLED"
                if token.cancelled: return f"{host.name:<20} CANCELLED"
                state_name = "OK" if observation.ssh_ok and not observation.signatures else "WARN"
                detail = self._fleet_detail(observation, scope)
                return f"{host.name:<20} {state_name:<5} {detail}"
        try:
            tasks = [asyncio.create_task(one(h)) for h in (
                hosts or tuple(h for h in self.registry.hosts.values() if h.enabled)
            )]
            rows = await asyncio.gather(*tasks)
            if scope == "gpu" and any(word in request for word in ("高い順", "ランキング")):
                def hottest(row: str) -> int:
                    values = [int(value) for value in re.findall(r"(\d+)°C", row)]
                    return max(values, default=-1)
                rows.sort(key=hottest, reverse=True)
            if scope == "vast" and "止ま" in request:
                rows = [
                    row for row in rows
                    if " ERROR " in row or not re.search(r"vast=active(?:\s|$)", row)
                ]
                if not rows:
                    rows = ["停止中のhostはありません"]
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

    async def _fleet_traffic(self, request, state, owner, channel, hosts) -> ServiceReply:
        """Collect one common READ primitive concurrently and aggregate numeric evidence."""
        job, token = self.jobs.create("fleet-traffic", None, request)
        state.last_host = None
        state.write_context_host = None
        state.current_hosts = tuple(host.name for host in hosts)
        state.context_kind = "fleet"
        state.context_source = "explicit"
        state.last_job_id = job.id
        state.last_scope = "fleet"
        self.conversations.save(owner, channel, state)
        self.jobs.start(job)
        semaphore = asyncio.Semaphore(self.max_parallel_hosts)
        functions = getattr(self.agent, "functions", None)
        period, heading = self._traffic_period(request)

        async def one(host):
            async with semaphore:
                if token.cancelled:
                    return host.name, None, "cancelled"
                if functions is None:
                    return host.name, None, "traffic READ unavailable"
                try:
                    value = await asyncio.to_thread(
                        functions.execute, "query_traffic_history",
                        {"host": host.name, "period": period, "interface": None},
                        host.name,
                    )
                    evidence = value.get("untrusted_evidence", {})
                    if isinstance(evidence, dict) and evidence.get("status") == "available":
                        return host.name, evidence, None
                    reason = evidence.get("failure_kind") or evidence.get("error") or evidence.get("status")
                    return host.name, None, str(reason or "unavailable").casefold()
                except Exception:
                    return host.name, None, "read failed"

        results = await asyncio.gather(*(one(host) for host in hosts))
        available = sorted(
            ((name, value) for name, value, error in results if value is not None),
            key=lambda item: int(item[1]["total_bytes"]), reverse=True,
        )
        rows = ["| # | Host | RX | TX | Total |", "|---:|---|---:|---:|---:|"]
        for rank, (name, value) in enumerate(available, 1):
            rows.append(
                f"| {rank} | {name} | {self._bytes(value['rx_bytes'])} | "
                f"{self._bytes(value['tx_bytes'])} | {self._bytes(value['total_bytes'])} |"
            )
        unavailable = [f"- {name}: {error}" for name, _, error in results if error]
        summary = f"{heading}の通信量 (host-local time)\n" + "\n".join(rows)
        if unavailable:
            summary += "\n\nUnavailable\n" + "\n".join(unavailable)
        state.investigation_context = {
            "previous_goal": request[:500], "hosts": list(state.current_hosts),
            "conclusion": summary[:1200], "key_findings": [row[:300] for row in rows[2:10]],
            "missing_evidence": unavailable[:6],
        }
        self.conversations.save(owner, channel, state)
        self.jobs.finish(job, self._clean(summary), "FLEET_PARTIAL_FAILURE" if unavailable else None)
        return ServiceReply(self._clean(f"Job #{job.id}\n{summary}"), job.id)

    @staticmethod
    def _traffic_period(request: str) -> tuple[str, str]:
        """Map an explicit user period to the backend; default to the current host-local day."""
        folded = request.casefold()
        if "過去24時間" in folded or re.search(r"(?<![a-z0-9])24h(?![a-z0-9])", folded):
            return "last_24h", "過去24時間"
        if "直近7日" in folded or "1週間" in folded:
            return "last_7d", "直近7日"
        if "昨日" in folded:
            return "yesterday", "昨日"
        if "今日" in folded:
            return "today", "今日"
        return "today", "今日"

    @staticmethod
    def _bytes(value: int) -> str:
        amount = float(value)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if amount < 1024 or unit == "TiB":
                return f"{amount:.1f} {unit}"
            amount /= 1024
        return f"{amount:.1f} TiB"

    @staticmethod
    def _fleet_detail(observation, scope):
        if scope == "gpu":
            if not observation.gpu.nvml_ok: return "NVML unavailable"
            devices = observation.gpu.devices
            return ", ".join(f"{d.model or 'GPU'} {d.temperature_c}°C" for d in devices) or "no GPU"
        if scope == "vast":
            return f"vast={observation.services.get('vastai', 'unknown')}"
        return f"disk={observation.system.filesystem_max_percent}% failed={observation.system.failed_units}"

    @staticmethod
    def _format_observation(observation, scope: str | None = None) -> str:
        gpu = observation.gpu; first = gpu.devices[0] if gpu.devices else None
        serious = {"NVIDIA_FALLEN_OFF_BUS", "GPU_STUCK_D3", "NVIDIA_GSP_FAILURE"}
        if not observation.ssh_ok or serious.intersection(observation.signatures): icon = "🔴"
        elif observation.signatures or (scope in (None, "gpu") and (not gpu.nvml_ok or not first)):
            icon = "🟡"
        else: icon = "🟢"
        signatures = ", ".join(observation.signatures) or "none"
        header = f"{icon} {observation.host} — {(scope or 'full').upper()}"
        if scope in (None, "gpu"):
            rows = [
                f"GPU{getattr(device, 'index', index)} {device.model or 'unknown'} | "
                f"{device.temperature_c if device.temperature_c is not None else '-'}°C | "
                f"util {device.utilization_percent if device.utilization_percent is not None else '-'}% | "
                f"power {getattr(device, 'power_draw_w', None) if getattr(device, 'power_draw_w', None) is not None else '-'}W | "
                f"VRAM {getattr(device, 'vram_used_mb', None) if getattr(device, 'vram_used_mb', None) is not None else '-'}"
                f"/{getattr(device, 'vram_total_mb', None) if getattr(device, 'vram_total_mb', None) is not None else '-'} MiB"
                for index, device in enumerate(gpu.devices)
            ]
            return f"{header}\n" + ("\n".join(rows) or "GPU: not detected") + (
                f"\nNVML: {'OK' if gpu.nvml_ok else 'unavailable'}"
            )
        if scope == "pci":
            return (f"{header}\nPCI count: {gpu.pci_count}\nnvidia: {gpu.nvidia_bound}\n"
                    f"vfio: {gpu.vfio_bound}\nunbound: {gpu.unbound}\nSignatures: {signatures}")
        if scope == "vast":
            return f"{header}\nvastai: {observation.services.get('vastai', 'unknown')}\nSignatures: {signatures}"
        if scope == "docker":
            containers = summarize_docker_status(str(observation.details.get("docker_status", "")))
            return (f"{header}\ndocker: {observation.services.get('docker', 'unknown')}\n"
                    f"Containers: total={containers['total']}, running={containers['running']}, "
                    f"stopped={containers['stopped']}\nSignatures: {signatures}")
        if scope == "vm":
            vm = summarize_vm_status(str(observation.details.get("vm_status", "")))
            domains = ", ".join(
                f"{domain['name']}={domain['state']}" for domain in vm["domains"][:5]
            ) or "none"
            return (f"{header}\nVMs: total={vm['total']}, running={vm['running']}\n"
                    f"Domains: {domains}\nPCI binding: nvidia={gpu.nvidia_bound}, "
                    f"vfio={gpu.vfio_bound}, unbound={gpu.unbound}\nSignatures: {signatures}")
        system = observation.system
        services = ", ".join(f"{name}={value}" for name, value in observation.services.items()) or "none"
        return (f"{header}\nFilesystem max usage: "
                f"{system.filesystem_max_percent if system.filesystem_max_percent is not None else '-'}%\n"
                f"Failed units: {system.failed_units}\nD-state count: {system.d_state_processes}\n"
                f"Services: {services}\nSignatures: {signatures}")

    @staticmethod
    def _format_investigation(host, result, *, include_evidence: bool = False) -> str:
        signatures = "\n".join(f"• {item}" for item in result.signatures) or "• none"
        gaps = []
        for gap in getattr(result, "capability_gaps", []):
            gap = assess_gap(gap)
            definition = capability_definition(gap.capability_id)
            if definition is None:
                continue
            evidence = "\n".join(f"- {item}" for item in gap.evidence) or "- 確認情報なし"
            state = {
                "available": "利用可能", "degraded": "一部利用不可", "missing": "不足",
                "unknown": "確認不能", "unsupported": "非対応",
            }[gap.status]
            note = ""
            if gap.status == "missing" and definition.acquisition is not None:
                if definition.acquisition.install_supported:
                    note = f"\nCatalog候補: {definition.acquisition.package_name}"
                else:
                    note = "\n自動導入対象なし"
                if definition.post_install_notes:
                    note += f"\n{definition.post_install_notes}"
            elif gap.status == "missing":
                note = "\n自動導入対象なし"
            elif gap.status == "degraded":
                note = "\n自動導入対象: なし (package already installed)"
            gaps.append(
                f"能力: {gap.capability_id} ({state})\n理由: {gap.reason}\n確認:\n{evidence}{note}"
            )
        gap_text = "\n\nCapability gap:\n" + "\n\n".join(gaps) if gaps else ""
        rendered = (f"🔎 {host} — 調査結果\n\n要約:\n{result.summary}\n\n検出:\n{signatures}\n\n"
                    f"提案:\n{result.recommended_action}\n\n確度:\n{result.confidence}{gap_text}")
        evidence = getattr(result, "_display_evidence", "")
        if include_evidence and evidence:
            rendered += "\n\n" + AgentService._render_evidence(evidence)
        return rendered
