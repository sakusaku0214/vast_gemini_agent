from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from vast_agent.jobs.models import Job, JobStatus
from vast_agent.models.host import Host
from vast_agent.models.observation import Observation
from vast_agent.models.tool_result import ToolResult
from vast_agent.storage.migrations import MIGRATIONS


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def migrate(self) -> int:
        with self.connect() as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='schema_version'").fetchone()
            current = db.execute("SELECT version FROM schema_version").fetchone()[0] if exists else 0
            for version, sql in enumerate(MIGRATIONS, 1):
                if version > current:
                    db.executescript(sql)
                    db.execute("UPDATE schema_version SET version=?", (version,))
            return len(MIGRATIONS)

    def upsert_host(self, host: Host) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO hosts VALUES(?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                "vast_id=excluded.vast_id,address=excluded.address,ssh_user=excluded.ssh_user,"
                "ssh_port=excluded.ssh_port,enabled=excluded.enabled",
                (host.name, host.vast_id, host.address, host.ssh_user, host.ssh_port, host.enabled),
            )

    def save_observation(self, observation: Observation, raw_log_path: str | None = None) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO observations(host,observed_at,payload_json,raw_log_path) VALUES(?,?,?,?)",
                (observation.host, observation.observed_at.isoformat(),
                 json.dumps(observation.model_dump(mode="json"), ensure_ascii=False), raw_log_path),
            )
            return int(cursor.lastrowid)

    def save_tool_run(
        self, observation_id: int, tool_name: str, result: ToolResult,
        raw_log_path: str | None = None,
    ) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO tool_runs(observation_id,tool_name,success,exit_code,duration_ms,"
                "error_code,raw_log_path) VALUES(?,?,?,?,?,?,?)",
                (
                    observation_id, tool_name, result.success, result.exit_code,
                    result.duration_ms, result.error_code, raw_log_path,
                ),
            )
            return int(cursor.lastrowid)

    def open_incident(self, host: str, signature: str, severity: str, summary: str) -> int:
        now = datetime.now(UTC).isoformat()
        dedup = f"{host}:{signature}"
        with self.connect() as db:
            row = db.execute(
                "SELECT id FROM incidents WHERE dedup_key=? AND status='open'", (dedup,),
            ).fetchone()
            if row:
                db.execute("UPDATE incidents SET last_seen_at=?,summary=? WHERE id=?", (now, summary, row[0]))
                return int(row[0])
            cursor = db.execute(
                "INSERT INTO incidents(host,primary_signature,opened_at,last_seen_at,status,severity,summary,dedup_key) "
                "VALUES(?,?,?,?,?,?,?,?)", (host, signature, now, now, "open", severity, summary, dedup),
            )
            return int(cursor.lastrowid)

    def recent_incidents(self, host: str, signature: str | None, limit: int) -> list[dict[str, object]]:
        sql = ("SELECT primary_signature,opened_at,last_seen_at,summary,status FROM incidents "
               "WHERE host=?")
        params: list[object] = [host]
        if signature:
            sql += " AND primary_signature=?"; params.append(signature)
        sql += " ORDER BY last_seen_at DESC LIMIT ?"; params.append(min(max(limit, 1), 5))
        with self.connect() as db:
            rows = db.execute(sql, params).fetchall()
        keys = ("signature", "opened_at", "last_seen_at", "summary", "status")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def save_token_usage(self, purpose: str, model: str, thinking_level: str,
                         usage: dict[str, int | None]) -> int:
        self.migrate()
        fields = ("input_tokens", "output_tokens", "thought_tokens", "cached_tokens",
                  "tool_use_tokens", "total_tokens")
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO token_usage(created_at,purpose,model,thinking_level,"
                + ",".join(fields) + ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(UTC).isoformat(), purpose, model, thinking_level,
                 *(usage.get(field) for field in fields)),
            )
            return int(cursor.lastrowid)

    def reconcile_jobs(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET status='INTERRUPTED',finished_at=?,error_code='PROCESS_RESTARTED' "
                "WHERE status IN ('QUEUED','RUNNING','CANCELLING')", (now,),
            )
            return cursor.rowcount

    def create_job(self, kind: str, host: str | None, request: str, created: datetime) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO jobs(kind,host,status,created_at,request_summary) VALUES(?,?,?,?,?)",
                (kind, host, "QUEUED", created.isoformat(), request),
            )
            return int(cursor.lastrowid)

    def update_job(self, job: Job) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,started_at=?,finished_at=?,result_summary=?,error_code=? WHERE id=?",
                (job.status, job.started_at.isoformat() if job.started_at else None,
                 job.finished_at.isoformat() if job.finished_at else None,
                 job.result_summary, job.error_code, job.id),
            )

    def set_job_status(self, job_id: int, status: JobStatus) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))

    def get_job(self, job_id: int) -> dict[str, object] | None:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def recent_jobs(self, limit: int = 20) -> list[dict[str, object]]:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def running_job_count(self) -> int:
        with self.connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('QUEUED','RUNNING','CANCELLING')",
            ).fetchone()
        return int(row[0])

    def load_conversation(self, owner: str, channel: str) -> dict[str, object] | None:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM conversation_state WHERE owner_id=? AND channel_id=?",
                (owner, channel),
            ).fetchone()
        return dict(row) if row else None

    def save_conversation(self, owner: str, channel: str, host: str | None,
                          job_id: int | None, scope: str | None) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversation_state VALUES(?,?,?,?,?,?) ON CONFLICT(owner_id,channel_id) "
                "DO UPDATE SET last_host=excluded.last_host,last_job_id=excluded.last_job_id,"
                "last_scope=excluded.last_scope,updated_at=excluded.updated_at",
                (owner, channel, host, job_id, scope, datetime.now(UTC).isoformat()),
            )
