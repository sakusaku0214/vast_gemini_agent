from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from vast_agent.config import load_hosts
from vast_agent.models.host import Host


def extract_machines(source: Path) -> dict[object, object]:
    tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "MACHINES" for target in targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict): raise ValueError("MACHINES must be a literal dictionary")
                return value
    raise ValueError("Literal MACHINES assignment was not found")


def _pick(data: dict, *keys: str, default=None):
    for key in keys:
        if key in data: return data[key]
    return default


def migrate_v1(source: Path, destination: Path) -> tuple[int, list[str]]:
    machines = extract_machines(source)
    existing = load_hosts(destination).hosts if destination.exists() else {}
    added: list[str] = []
    for key, raw in machines.items():
        if not isinstance(raw, dict): continue
        name = str(_pick(raw, "name", "machine_name", default=key))
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or f"machine-{key}"
        if name in existing: continue
        address = _pick(raw, "ip", "address", "host")
        user = _pick(raw, "ssh_user", "user", "username")
        vast_id = _pick(raw, "vast_id", "id", "machine_id", default=key if str(key).isdigit() else None)
        if address and user:
            existing[name] = Host(name=name, vast_id=int(vast_id) if vast_id is not None else None,
                                  address=str(address), ssh_user=str(user))
            added.append(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"hosts": {name: host.model_dump(exclude={"name"}, mode="json") for name, host in existing.items()}}
    destination.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return len(added), added
