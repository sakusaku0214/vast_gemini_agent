from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from vast_agent.execution.base import Executor, redact
from vast_agent.inspector import FULL, GROUPS
from vast_agent.models.host import Host
from vast_agent.models.observation import Observation
from vast_agent.models.tool_result import ToolResult
from vast_agent.storage.database import Database
from vast_agent.tools.registry import run_tool

INCIDENT_SIGNATURES = {
    "SSH_UNREACHABLE",
    "NVML_UNAVAILABLE",
    "GPU_UNBOUND",
    "GPU_STUCK_D3",
    "NVIDIA_FALLEN_OFF_BUS",
    "NVIDIA_UVM_FATAL",
    "NVIDIA_GSP_FAILURE",
    "VAST_OFFLINE",
    "DOCKER_FAILED",
    "FILESYSTEM_FULL",
    "SYSTEMD_FAILED_UNIT",
}


@dataclass(frozen=True)
class InspectionRecord:
    observation: Observation
    observation_id: int
    tool_results: dict[str, ToolResult]
    incident_ids: dict[str, int]


class InspectionService:
    def __init__(
        self, database: Database, log_root: Path, secrets: Iterable[str] = (),
    ) -> None:
        self.database = database
        self.log_root = log_root
        self.secrets = tuple(secrets)

    def inspect_and_record(
        self, host: Host, executor: Executor, group: str | None = None,
    ) -> InspectionRecord:
        from vast_agent.diagnostics.parser import build_observation

        self.database.migrate()
        self.database.upsert_host(host)
        names = GROUPS[group] if group else FULL
        ping = run_tool("host_ping", host, executor)
        results = {"host_ping": ping}
        if ping.success:
            results.update({
                name: run_tool(name, host, executor)
                for name in names
                if name != "host_ping"
            })
        observation = self._redact_observation(build_observation(host.name, results))
        log_paths = {
            name: self._save_raw_log(observation, name, result)
            for name, result in results.items()
        }
        directory = next((path.parent for path in log_paths.values() if path), None)
        observation_id = self.database.save_observation(
            observation, str(directory) if directory else None,
        )
        for name, result in results.items():
            path = log_paths[name]
            self.database.save_tool_run(
                observation_id, name, result, str(path) if path else None,
            )
        incidents = {
            signature: self.database.open_incident(
                host.name, signature, "high", f"{signature} detected on {host.name}",
            )
            for signature in observation.signatures
            if signature in INCIDENT_SIGNATURES
        }
        return InspectionRecord(observation, observation_id, results, incidents)

    def _redact_observation(self, observation: Observation) -> Observation:
        def clean(value: object) -> object:
            if isinstance(value, str):
                return redact(value, self.secrets)
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            return value

        return Observation.model_validate(clean(observation.model_dump()))

    def _save_raw_log(
        self, observation: Observation, tool_name: str, result: ToolResult,
    ) -> Path | None:
        sections = []
        if result.stdout:
            sections.append(f"[stdout]\n{result.stdout}")
        if result.stderr:
            sections.append(f"[stderr]\n{result.stderr}")
        if not sections:
            return None
        safe_host = re.sub(r"[^A-Za-z0-9_.-]", "_", observation.host)
        stamp = observation.observed_at.strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.log_root / observation.observed_at.strftime("%Y%m%d") / safe_host
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{tool_name}-{stamp}.log"
        target.write_text(redact("\n\n".join(sections), self.secrets), encoding="utf-8")
        return target


def load_redaction_secrets(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    values = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        value = line.split("=", 1)[1].strip().strip("'\"")
        if value:
            values.append(value)
    return tuple(values)
