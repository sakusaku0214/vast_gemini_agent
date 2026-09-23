from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from vast_agent.conversation.state import ConversationStore
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
        max_parallel_hosts: int = 3,
        secrets: tuple[str, ...] = (),
        remember_last_host: bool = True,
    ) -> None:
        # registry/inspection/executor/max_parallel_hosts stay in the constructor for
        # runtime compatibility with the Phase 7-9 bootstrap. They are deliberately
        # not used to interpret natural-language requests here.
        self.registry = registry
        self.inspection = inspection
        self.executor = executor
        self.jobs = jobs
        self.conversations = conversations
        self.agent = agent
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
                return self._clean(self.agent.investigate(None, safe_text))
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
        return (
            text == "止めて"
            or text.startswith("!cancel")
            or bool(re.search(r"#\d+\s*止めて", text))
            or bool(re.search(r"(?:今の調査|この調査|ジョブ)(?:を)?止めて", text))
        )

    def _clean(self, value: str) -> str:
        return redact(value, self.secrets)
