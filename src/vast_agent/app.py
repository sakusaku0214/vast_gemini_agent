from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from vast_agent.config import ConfigError, initialize_config, load_hosts
from vast_agent.execution.ssh import SSHExecutor
from vast_agent.inspector import GROUPS
from vast_agent.migrate_v1 import migrate_v1
from vast_agent.paths import RuntimePaths
from vast_agent.services.inspection import InspectionService, load_redaction_secrets
from vast_agent.storage.database import Database
from vast_agent.tools.registry import run_tool
from vast_agent.trust import trust_host


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="vast-agent", description="Vast Gemini Agent v2 (read-only)")
    root.add_argument("--runtime", type=Path, help="Override runtime root (testing/advanced use)")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("install", "doctor", "config", "hosts"):
        sub.add_parser(name)
    for name in ("trust-host", "test-host"):
        item = sub.add_parser(name); item.add_argument("host")
    inspect = sub.add_parser("inspect"); inspect.add_argument("host")
    flags = inspect.add_mutually_exclusive_group()
    for group in GROUPS: flags.add_argument(f"--{group}", action="store_true")
    inspect.add_argument("--json", action="store_true")
    migrate = sub.add_parser("migrate-v1"); migrate.add_argument("path", type=Path)
    for name in ("start", "stop", "restart", "status", "update", "backup"):
        sub.add_parser(name)
    return root


def _registry(paths: RuntimePaths):
    try: return load_hosts(paths.hosts_file)
    except ConfigError as exc:
        print(f"ERROR CONFIG_INVALID: {exc}", file=sys.stderr); return None


def doctor(paths: RuntimePaths) -> int:
    print("Vast Gemini Agent Doctor\n")
    checks: list[tuple[str, str, str]] = []
    checks.append(("Runtime directory", "OK" if paths.root.is_dir() else "ERROR", str(paths.root)))
    checks.append(("Python", "OK" if sys.version_info >= (3, 12) else "ERROR", sys.version.split()[0]))
    checks.append(("Virtual environment", "OK" if sys.prefix != sys.base_prefix else "WARNING", sys.prefix))
    registry = _registry(paths)
    checks.append(("Configuration", "OK" if registry is not None and paths.hosts_file.exists() else "ERROR", str(paths.hosts_file)))
    try:
        Database(paths.database).migrate(); db_state = "OK"
    except Exception as exc:  # CLI boundary: convert failures to a concise health check
        db_state = "ERROR"; checks.append(("SQLite", db_state, str(exc)))
    else: checks.append(("SQLite", db_state, str(paths.database)))
    ssh = shutil.which("ssh")
    checks.append(("OpenSSH", "OK" if ssh else "ERROR", ssh or "ssh executable not found"))
    checks.append(("known_hosts", "OK" if paths.known_hosts.exists() else "WARNING", str(paths.known_hosts)))
    for label, state, detail in checks: print(f"{label:<22} {state:<7} {detail}")
    print(f"\nConfigured hosts: {len(registry.hosts) if registry else 0}")
    states = {state for _, state, _ in checks}
    result = "UNHEALTHY" if "ERROR" in states else "HEALTHY" if "WARNING" not in states else "HEALTHY WITH WARNINGS"
    print(f"\nResult: {result}")
    return 1 if "ERROR" in states else 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = RuntimePaths.discover(args.runtime)
    if args.command == "install":
        initialize_config(paths); Database(paths.database).migrate()
        print(f"Runtime initialized: {paths.root}"); return 0
    if args.command == "doctor": return doctor(paths)
    if args.command == "config":
        print(f"Runtime: {paths.root}\nHosts: {paths.hosts_file}\nAgent: {paths.agent_file}\nSecrets: configured={paths.secrets_file.exists()}")
        return 0
    if args.command == "migrate-v1":
        try: count, names = migrate_v1(args.path, paths.hosts_file)
        except (OSError, SyntaxError, ValueError, ConfigError) as exc:
            print(f"ERROR CONFIG_INVALID: {exc}", file=sys.stderr); return 2
        print(f"Migrated {count} host(s): {', '.join(names) if names else '(none)'}"); return 0
    registry = _registry(paths)
    if registry is None: return 2
    if args.command == "hosts":
        for host in registry.hosts.values():
            print(f"{host.name:<24} {'enabled' if host.enabled else 'disabled':<8} {host.address}:{host.ssh_port}")
        return 0
    if args.command in {"start", "stop", "restart", "status", "update", "backup"}:
        print(f"ERROR: command '{args.command}' is not implemented in Phase 0-3", file=sys.stderr); return 3
    try: host = registry.resolve(args.host)
    except KeyError:
        print(f"ERROR HOST_NOT_FOUND: {args.host}", file=sys.stderr); return 2
    if not host.enabled:
        print(f"ERROR HOST_NOT_FOUND: host '{host.name}' is disabled", file=sys.stderr); return 2
    if args.command == "trust-host":
        try: ok, message = trust_host(host, paths.known_hosts)
        except (OSError, TimeoutError) as exc: ok, message = False, str(exc)
        print(message); return 0 if ok else 2
    executor = SSHExecutor(paths.known_hosts)
    if args.command == "test-host":
        result = run_tool("host_ping", host, executor)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)); return 0 if result.success else 2
    if args.command == "inspect":
        group = next((name for name in GROUPS if getattr(args, name)), None)
        record = InspectionService(
            Database(paths.database), paths.observation_logs,
            load_redaction_secrets(paths.secrets_file),
        ).inspect_and_record(host, executor, group)
        observation = record.observation
        payload = observation.model_dump(mode="json")
        if args.json: print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Host: {host.name}\nSSH: {'OK' if observation.ssh_ok else 'ERROR'}")
            print(f"GPU: PCI={observation.gpu.pci_count}, NVML={'OK' if observation.gpu.nvml_ok else 'unavailable'}, nvidia={observation.gpu.nvidia_bound}, vfio={observation.gpu.vfio_bound}, unbound={observation.gpu.unbound}")
            print("Services: " + (", ".join(f"{k}={v}" for k, v in observation.services.items()) or "none observed"))
            print("Signatures: " + (", ".join(observation.signatures) or "none"))
        return 0 if observation.ssh_ok else 2
    return 2
