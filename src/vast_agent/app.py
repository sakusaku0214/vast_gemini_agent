from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import GoogleInteractionsClient
from vast_agent.agent.models import Route
from vast_agent.agent.orchestrator import InvestigationAgent
from vast_agent.agent.prompts import SYSTEM_PROMPT
from vast_agent.agent.router import route_intent
from vast_agent.config import (
    ConfigError,
    initialize_config,
    load_gemini_settings,
    load_hosts,
    load_runtime_settings,
)
from vast_agent.conversation.state import ConversationStore
from vast_agent.execution.base import redact
from vast_agent.execution.ssh import SSHExecutor
from vast_agent.inspector import GROUPS
from vast_agent.jobs.manager import JobManager
from vast_agent.logging_utils import configure_agent_logging
from vast_agent.migrate_v1 import migrate_v1
from vast_agent.paths import RuntimePaths
from vast_agent.runtime import RuntimeManager
from vast_agent.services.agent_service import AgentService
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
    ask = sub.add_parser("ask"); ask.add_argument("question")
    investigate = sub.add_parser("investigate"); investigate.add_argument("host"); investigate.add_argument("question")
    sub.add_parser("gemini-check")
    migrate = sub.add_parser("migrate-v1"); migrate.add_argument("path", type=Path)
    for name in ("start", "stop", "restart", "status", "update", "backup", "discord-check"):
        sub.add_parser(name)
    run = sub.add_parser("run-discord"); run.add_argument("marker", nargs="?")
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
    try: settings = load_gemini_settings(paths.agent_file); gemini_state = "OK"
    except ConfigError as exc: settings = None; gemini_state = f"ERROR: {exc}"
    checks.append(("Gemini config", "OK" if settings else "ERROR", gemini_state))
    checks.append(("API key", "OK" if _api_key(paths) else "WARNING",
                   "configured" if _api_key(paths) else "not configured"))
    for label, state, detail in checks: print(f"{label:<22} {state:<7} {detail}")
    print(f"\nConfigured hosts: {len(registry.hosts) if registry else 0}")
    states = {state for _, state, _ in checks}
    result = "UNHEALTHY" if "ERROR" in states else "HEALTHY" if "WARNING" not in states else "HEALTHY WITH WARNINGS"
    print(f"\nResult: {result}")
    return 1 if "ERROR" in states else 0


def _api_key(paths: RuntimePaths) -> str | None:
    return _secrets(paths).get("GEMINI_API_KEY")


def _secrets(paths: RuntimePaths) -> dict[str, str]:
    values: dict[str, str] = {}
    if paths.secrets_file.exists():
        for line in paths.secrets_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1); values[key.strip()] = value.strip().strip("'\"")
    for key in ("GEMINI_API_KEY", "DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID"):
        if os.environ.get(key): values[key] = os.environ[key]
    return values


def _redaction_secrets(paths: RuntimePaths) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*load_redaction_secrets(paths.secrets_file), *_secrets(paths).values())))


def _agent(paths: RuntimePaths, registry, executor):
    settings = load_gemini_settings(paths.agent_file); key = _api_key(paths)
    if not key: return None
    database = Database(paths.database)
    secrets = _redaction_secrets(paths)
    functions = FunctionExecutor(
        registry,
        executor,
        settings.max_evidence_chars_per_tool,
        secrets,
    )
    return InvestigationAgent(
        GoogleInteractionsClient(key, settings.api_version),
        functions,
        database,
        settings,
        registry,
    )


def _service(paths: RuntimePaths, registry, executor) -> AgentService:
    database = Database(paths.database); database.migrate()
    _, job_settings, conversation_settings = load_runtime_settings(paths.agent_file)
    secrets = _redaction_secrets(paths)
    inspection = InspectionService(database, paths.observation_logs,
                                   secrets)
    return AgentService(registry, inspection, executor,
                        JobManager(database, job_settings.max_recent_jobs),
                        ConversationStore(database), _agent(paths, registry, executor),
                        job_settings.max_parallel_hosts, secrets,
                        conversation_settings.remember_last_host)


def _discord_enabled(paths: RuntimePaths) -> bool:
    discord_settings, _, _ = load_runtime_settings(paths.agent_file)
    return discord_settings.enabled


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
    if args.command == "gemini-check":
        key = _api_key(paths)
        if not key:
            print("Gemini API       NOT CONFIGURED\nAPI key           missing"); return 2
        settings = load_gemini_settings(paths.agent_file)
        try:
            client = GoogleInteractionsClient(key, settings.api_version)
            client.interact(model=settings.model, inputs=[{"type": "text", "text": "Reply OK."}],
                            system_instruction=SYSTEM_PROMPT, tools=[], thinking_level="low", store=False)
        except Exception:
            print(f"Gemini API       ERROR\nModel            {settings.model}\nInteractions API ERROR"); return 2
        print(f"Gemini API       OK\nModel            {settings.model}\nInteractions API OK"); return 0
    if args.command == "migrate-v1":
        try: count, names = migrate_v1(args.path, paths.hosts_file)
        except (OSError, SyntaxError, ValueError, ConfigError) as exc:
            print(f"ERROR CONFIG_INVALID: {exc}", file=sys.stderr); return 2
        print(f"Migrated {count} host(s): {', '.join(names) if names else '(none)'}"); return 0
    if args.command == "discord-check":
        secrets = _secrets(paths)
        for label, key in (("Bot token", "DISCORD_BOT_TOKEN"), ("Channel ID", "DISCORD_CHANNEL_ID"),
                           ("Owner ID", "DISCORD_OWNER_USER_ID")):
            print(f"{label:<15} {'configured' if secrets.get(key) else 'missing'}")
        return 0 if all(secrets.get(k) for k in ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID")) else 2
    registry = _registry(paths)
    if registry is None: return 2
    if args.command == "hosts":
        for host in registry.hosts.values():
            print(f"{host.name:<24} {'enabled' if host.enabled else 'disabled':<8} {host.address}:{host.ssh_port}")
        return 0
    if args.command in {"start", "stop", "restart", "status"}:
        runtime = RuntimeManager(paths)
        if args.command == "start":
            if not _discord_enabled(paths):
                print("Discord is disabled in agent.yaml.", file=sys.stderr); return 2
            required = ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID")
            missing = [key for key in required if not _secrets(paths).get(key)]
            if missing:
                print("Discord configuration missing: " + ", ".join(missing), file=sys.stderr)
                return 2
            ok, message = runtime.start()
        elif args.command == "stop": ok, message = runtime.stop()
        elif args.command == "restart":
            ok, message = runtime.stop()
            if ok:
                if not _discord_enabled(paths):
                    print("Discord is disabled in agent.yaml.", file=sys.stderr); return 2
                required = ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID")
                missing = [key for key in required if not _secrets(paths).get(key)]
                if missing:
                    print("Discord configuration missing: " + ", ".join(missing), file=sys.stderr)
                    return 2
                ok, message = runtime.start()
        else:
            state, detail = runtime.status(); secrets = _secrets(paths)
            database = Database(paths.database); database.migrate()
            print("Vast Gemini Agent\n")
            print(f"Status      {state}")
            if detail: print(f"PID         {detail.get('pid')}\nStarted     {detail.get('started_at')}")
            print(f"Discord     {'configured' if all(secrets.get(k) for k in ('DISCORD_BOT_TOKEN','DISCORD_CHANNEL_ID','DISCORD_OWNER_USER_ID')) else 'not configured'}")
            print(f"Gemini      {'configured' if secrets.get('GEMINI_API_KEY') else 'not configured'}")
            print(f"Hosts       {len(registry.hosts)}\nJobs        {database.running_job_count()} running")
            return 0
        print(message); return 0 if ok else 2
    if args.command in {"update", "backup"}:
        print(f"ERROR: command '{args.command}' is not implemented", file=sys.stderr); return 3
    executor = SSHExecutor(paths.known_hosts)
    if args.command == "run-discord":
        from vast_agent.discord_app.bot import create_bot

        if not _discord_enabled(paths):
            print("Discord is disabled in agent.yaml.", file=sys.stderr); return 2
        secrets = _secrets(paths); required = ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_OWNER_USER_ID")
        missing = [key for key in required if not secrets.get(key)]
        if missing:
            print("Discord configuration missing: " + ", ".join(missing), file=sys.stderr); return 2
        try: owner = int(secrets["DISCORD_OWNER_USER_ID"]); channel = int(secrets["DISCORD_CHANNEL_ID"])
        except ValueError:
            print("Discord channel and owner IDs must be integers.", file=sys.stderr); return 2
        paths.create()
        redaction_secrets = _redaction_secrets(paths)
        configure_agent_logging(paths.agent_logs / "agent.log", redaction_secrets)
        service = _service(paths, registry, executor)
        service.jobs.reconcile_stale()
        client = create_bot(owner, channel, service)
        with contextlib.suppress(KeyboardInterrupt):
            client.run(secrets["DISCORD_BOT_TOKEN"], reconnect=True, log_handler=None)
        return 0
    if args.command == "ask":
        reply = asyncio.run(_service(paths, registry, executor).handle_question(args.question))
        print(reply.text)
        return 0 if reply.job_id is not None else 2
    if args.command == "investigate":
        try: target = registry.resolve(args.host)
        except KeyError:
            print(f"ERROR HOST_NOT_FOUND: {args.host}", file=sys.stderr); return 2
        if not target.enabled:
            print(f"ERROR HOST_DISABLED: {target.name}", file=sys.stderr); return 2
        agent = _agent(paths, registry, executor)
        if agent is None: print("Gemini unavailable: GEMINI_API_KEY is not configured."); return 2
        question = redact(args.question, _redaction_secrets(paths))
        print(agent.investigate(target.name, question)); return 0
    try: host = registry.resolve(args.host)
    except KeyError:
        print(f"ERROR HOST_NOT_FOUND: {args.host}", file=sys.stderr); return 2
    if not host.enabled:
        print(f"ERROR HOST_NOT_FOUND: host '{host.name}' is disabled", file=sys.stderr); return 2
    if args.command == "trust-host":
        try: ok, message = trust_host(host, paths.known_hosts)
        except (OSError, TimeoutError) as exc: ok, message = False, str(exc)
        print(message); return 0 if ok else 2
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
