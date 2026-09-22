import sqlite3
import time
from pathlib import Path

from conftest import fixture_results

from vast_agent.execution.base import FakeExecutor
from vast_agent.models.host import Host
from vast_agent.models.observation import Observation
from vast_agent.models.tool_result import ToolResult
from vast_agent.services.inspection import InspectionService
from vast_agent.storage.database import Database


def test_schema_observation_and_incident_dedup(tmp_path):
    database = Database(tmp_path / "agent.db")
    assert database.migrate() == 2
    host = Host(name="node", address="192.0.2.2", ssh_user="user")
    database.upsert_host(host)
    assert database.save_observation(Observation(host="node", ssh_ok=True)) > 0
    first = database.open_incident("node", "NVML_UNAVAILABLE", "high", "NVML unavailable")
    second = database.open_incident("node", "NVML_UNAVAILABLE", "high", "Still unavailable")
    assert first == second
    with sqlite3.connect(database.path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"schema_version", "hosts", "observations", "tool_runs", "incidents"} <= tables


def test_inspection_persists_runs_logs_and_deduplicates_incident(tmp_path, host):
    database = Database(tmp_path / "agent.db")
    results = fixture_results("normal")
    results["get_kernel_gpu_errors"] = results["get_kernel_gpu_errors"].model_copy(
        update={"stdout": "NVRM: GPU has fallen off the bus\napi_token=super-secret"},
    )
    service = InspectionService(database, tmp_path / "logs" / "observations", ["super-secret"])

    first = service.inspect_and_record(host, FakeExecutor(results), "gpu")
    time.sleep(0.001)
    second = service.inspect_and_record(host, FakeExecutor(results), "gpu")

    assert first.observation_id != second.observation_id
    assert first.incident_ids == second.incident_ids
    assert set(first.incident_ids) == {"NVIDIA_FALLEN_OFF_BUS"}
    with sqlite3.connect(database.path) as db:
        assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM tool_runs").fetchone()[0] == 8
        assert db.execute("SELECT count(*) FROM incidents").fetchone()[0] == 1
        opened, last_seen = db.execute(
            "SELECT opened_at,last_seen_at FROM incidents",
        ).fetchone()
        assert last_seen > opened
        payloads = [row[0] for row in db.execute("SELECT payload_json FROM observations")]
        paths = [row[0] for row in db.execute(
            "SELECT raw_log_path FROM tool_runs WHERE raw_log_path IS NOT NULL",
        )]
    assert paths
    logs = [Path(path).read_text(encoding="utf-8") for path in paths]
    assert all("super-secret" not in payload for payload in payloads)
    assert all("[REDACTED]" in payload for payload in payloads)
    assert all("super-secret" not in log for log in logs)
    assert any("[REDACTED]" in log for log in logs)


def test_offline_inspection_only_runs_ping_and_persists_failure(tmp_path, host):
    database = Database(tmp_path / "agent.db")
    executor = FakeExecutor({
        "host_ping": ToolResult(
            success=False, exit_code=255, stderr="connection timed out", duration_ms=10,
        ),
    })
    record = InspectionService(
        database, tmp_path / "logs" / "observations",
    ).inspect_and_record(host, executor)

    assert executor.calls == ["host_ping"]
    assert record.observation.signatures == ["SSH_UNREACHABLE"]
    assert set(record.incident_ids) == {"SSH_UNREACHABLE"}
    with sqlite3.connect(database.path) as db:
        assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM tool_runs").fetchone()[0] == 1
        assert db.execute("SELECT tool_name FROM tool_runs").fetchone()[0] == "host_ping"
        assert db.execute("SELECT count(*) FROM incidents").fetchone()[0] == 1
