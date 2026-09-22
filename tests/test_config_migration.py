
import pytest

from vast_agent.config import load_hosts
from vast_agent.migrate_v1 import extract_machines, migrate_v1
from vast_agent.paths import RuntimePaths


def test_runtime_override_and_create(tmp_path):
    paths = RuntimePaths.discover(tmp_path)
    paths.create()
    assert paths.known_hosts.is_file()
    assert paths.database.parent.is_dir()


def test_literal_migration_merges_without_execution(tmp_path):
    marker = tmp_path / "executed"
    source = tmp_path / "old.py"
    source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "MACHINES = {11908: {'name': 'garage', 'ip': '192.0.2.44', 'user': 'alice'}}\n",
        encoding="utf-8",
    )
    destination = tmp_path / "hosts.yaml"
    count, names = migrate_v1(source, destination)
    assert (count, names) == (1, ["garage"])
    assert not marker.exists()
    assert load_hosts(destination).hosts["garage"].vast_id == 11908
    assert migrate_v1(source, destination)[0] == 0


def test_non_literal_machines_rejected(tmp_path):
    source = tmp_path / "bad.py"
    source.write_text("MACHINES = dict(secret='x')", encoding="utf-8")
    with pytest.raises(ValueError): extract_machines(source)
