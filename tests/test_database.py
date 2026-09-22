import sqlite3

from vast_agent.models.host import Host
from vast_agent.models.observation import Observation
from vast_agent.storage.database import Database


def test_schema_observation_and_incident_dedup(tmp_path):
    database = Database(tmp_path / "agent.db")
    assert database.migrate() == 1
    host = Host(name="node", address="192.0.2.2", ssh_user="user")
    database.upsert_host(host)
    assert database.save_observation(Observation(host="node", ssh_ok=True)) > 0
    first = database.open_incident("node", "NVML_UNAVAILABLE", "high", "NVML unavailable")
    second = database.open_incident("node", "NVML_UNAVAILABLE", "high", "Still unavailable")
    assert first == second
    with sqlite3.connect(database.path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"schema_version", "hosts", "observations", "tool_runs", "incidents"} <= tables
