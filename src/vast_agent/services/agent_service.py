from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass

from vast_agent.approval import ProposalStore
from vast_agent.conversation.state import ConversationStore
from vast_agent.execution.argv import ArgvRejected, validate_write_argv
from vast_agent.execution.base import redact
from vast_agent.jobs.cancellation import current_cancellation
from vast_agent.jobs.manager import JobManager


@dataclass
class ServiceReply:
    text: str
    job_id: int | None = None
    ignored: bool = False


class AgentService:
    """Thin natural-language entrypoint. Gemini owns meaning; code owns lifecycle."""

    def __init__(
        self,
        registry,
        inspection,
        executor,
        jobs: JobManager,
        conversations: ConversationStore,
        agent=None,
        proposals: ProposalStore | None = None,
        max_parallel_hosts: int = 3,
        secrets: tuple[str, ...] = (),
        remember_last_host: bool = True,
    ) -> None:
        self.registry = registry
        self.inspection = inspection
        self.executor = executor
        self.jobs = jobs
        self.conversations = conversations
        self.agent = agent
        self.proposals = proposals or ProposalStore(jobs.database)
        self.max_parallel_hosts = max_parallel_hosts
        self.secrets = secrets
        self.remember_last_host = remember_last_host

    async def handle_question(
        self,
        text: str,
        owner: int | str = "cli",
        channel: int | str = "cli",
    ) -> ServiceReply:
        state = self.conversations.get(owner, channel)
        folded = text.casefold().strip()

        if folded == "!jobs" or "ジョブ見せ" in folded:
            lines = [
                f"#{j['id']} {j['status']:<11} {j['host'] or '-'} {j['kind']}"
                for j in self.jobs.recent()
            ]
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

        approval = self._approval_id(text, str(owner), str(channel))
        if approval is not None:
            return await self._execute_approved(approval, str(owner), str(channel))

        if self.agent is None:
            return ServiceReply("Gemini unavailable: GEMINI_API_KEY is not configured.")

        safe_text = self._clean(text)
        job, token = self.jobs.create("agent", None, safe_text)
        state.last_job_id = job.id
        self.conversations.save(owner, channel, state)
        self.jobs.start(job)

        def work() -> str:
            context = current_cancellation.set(token)
            try:
                if token.cancelled:
                    return "cancelled"
                return self._clean(
                    self.agent.investigate(
                        None,
                        safe_text,
                        owner_id=str(owner),
                        channel_id=str(channel),
                    )
                )
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

    def _approval_id(self, text: str, owner: str, channel: str) -> int | None:
        match = re.fullmatch(r"\s*承認(?:\s*#?(\d+))?\s*", text)
        if not match:
            return None
        if match.group(1):
            return int(match.group(1))

        pending = self.proposals.pending_for(owner, channel)
        if len(pending) == 1:
            return pending[0].id
        return -1

    async def _execute_approved(
        self,
        proposal_id: int,
        owner: str,
        channel: str,
    ) -> ServiceReply:
        if proposal_id < 0:
            pending = self.proposals.pending_for(owner, channel)
            ids = ", ".join(f"#{item.id}" for item in pending) or "なし"
            return ServiceReply(f"承認対象を指定してください。承認待ち: {ids}")

        proposal = self.proposals.claim(proposal_id, owner, channel)
        if proposal is None:
            existing = self.proposals.get(proposal_id)
            if existing is None:
                return ServiceReply(f"Proposal #{proposal_id} は存在しません。")
            if existing.owner_id != owner or existing.channel_id != channel:
                return ServiceReply(f"Proposal #{proposal_id} はこのOWNER/channelでは承認できません。")
            return ServiceReply(
                f"Proposal #{proposal_id} は承認待ちではありません: {existing.status}"
            )

        try:
            host = self.registry.resolve(proposal.host)
        except KeyError:
            self.proposals.mark_failed(proposal.id)
            return ServiceReply(f"Proposal #{proposal.id}: hostが見つかりません。")
        if not host.enabled:
            self.proposals.mark_failed(proposal.id)
            return ServiceReply(f"Proposal #{proposal.id}: hostはdisabledです。")

        try:
            argv = validate_write_argv(proposal.argv).argv
        except ArgvRejected as exc:
            self.proposals.mark_failed(proposal.id)
            return ServiceReply(f"Proposal #{proposal.id}: WRITE境界で拒否: {exc}")

        result = await asyncio.to_thread(
            self.executor.execute,
            host,
            argv,
            proposal.timeout,
            None,
        )
        if result.success:
            self.proposals.mark_executed(proposal.id)
            status = "EXECUTED"
        else:
            self.proposals.mark_failed(proposal.id)
            status = "FAILED"

        stdout = self._compact(result.stdout)
        stderr = self._compact(result.stderr)
        command = shlex.join(argv)
        text = (
            f"Proposal #{proposal.id} {status}\n"
            f"Host: {proposal.host}\n"
            f"Command: {command}\n"
            f"Exit: {result.exit_code}\n"
            f"stdout:\n{stdout or '(empty)'}\n"
            f"stderr:\n{stderr or '(empty)'}"
        )
        return ServiceReply(self._clean(text))

    @staticmethod
    def _is_cancel_request(text: str) -> bool:
        return (
            text == "止めて"
            or text.startswith("!cancel")
            or bool(re.search(r"#\d+\s*止めて", text))
            or bool(re.search(r"(?:今の調査|この調査|ジョブ)(?:を)?止めて", text))
        )

    def _compact(self, value: str, limit: int = 2500) -> str:
        text = self._clean(value)
        if len(text) <= limit:
            return text
        head = limit // 3
        tail = limit - head
        return text[:head] + "\n…[truncated]…\n" + text[-tail:]

    def _clean(self, value: str) -> str:
        return redact(value, self.secrets)
