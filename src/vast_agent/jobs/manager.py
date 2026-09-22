from __future__ import annotations

import threading
from datetime import UTC, datetime

from vast_agent.jobs.cancellation import CancellationToken
from vast_agent.jobs.locks import HostLockManager
from vast_agent.jobs.models import Job, JobStatus
from vast_agent.storage.database import Database


class JobManager:
    def __init__(self, database: Database, max_recent: int = 20) -> None:
        self.database = database
        self.max_recent = max_recent
        self.locks = HostLockManager()
        self._tokens: dict[int, CancellationToken] = {}
        self._guard = threading.Lock()
        database.migrate()

    def reconcile_stale(self) -> int:
        """Mark jobs left active by a previous process; call once during bot bootstrap."""
        return self.database.reconcile_jobs()

    def create(self, kind: str, host: str | None, request: str) -> tuple[Job, CancellationToken]:
        now = datetime.now(UTC)
        job_id = self.database.create_job(kind, host, request[:500], now)
        job = Job(job_id, kind, host, JobStatus.QUEUED, now, request[:500])
        token = CancellationToken()
        with self._guard:
            self._tokens[job_id] = token
        return job, token

    def start(self, job: Job) -> None:
        job.status = JobStatus.RUNNING; job.started_at = datetime.now(UTC)
        self.database.update_job(job)

    def finish(self, job: Job, summary: str, error_code: str | None = None) -> None:
        token = self._tokens.get(job.id)
        job.status = JobStatus.CANCELLED if token and token.cancelled else (
            JobStatus.FAILED if error_code else JobStatus.SUCCEEDED
        )
        job.finished_at = datetime.now(UTC); job.result_summary = summary[:1000]
        job.error_code = error_code
        self.database.update_job(job)
        with self._guard:
            self._tokens.pop(job.id, None)

    def cancel(self, job_id: int) -> str:
        with self._guard:
            token = self._tokens.get(job_id)
        if token is None:
            row = self.database.get_job(job_id)
            return "already finished" if row else "not found"
        self.database.set_job_status(job_id, JobStatus.CANCELLING)
        token.cancel()
        return "cancellation requested"

    def recent(self) -> list[dict[str, object]]:
        return self.database.recent_jobs(self.max_recent)

    def running(self) -> list[dict[str, object]]:
        return [j for j in self.recent() if j["status"] in {"RUNNING", "CANCELLING", "QUEUED"}]
