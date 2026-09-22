from __future__ import annotations

import json
from typing import Final

from pydantic import ValidationError

from vast_agent.agent.models import CollectEvidenceArgs, InspectHostArgs, RecentIncidentsArgs
from vast_agent.config import HostRegistry
from vast_agent.execution.base import Executor, redact
from vast_agent.services.inspection import InspectionService
from vast_agent.storage.database import Database

EVIDENCE_GROUP: Final = {
    "gpu_status": "gpu", "gpu_processes": "gpu", "pci": "pci",
    "kernel_gpu_errors": "gpu", "d_state": "system", "vast_status": "vast",
    "vast_logs": "vast", "docker": "docker", "vm": "vm", "services": "system",
    "journal_errors": "system", "system_health": "system",
}

FUNCTION_DECLARATIONS = [
    {"type": "function", "name": "inspect_host", "description": "Inspect an allowlisted host scope read-only.",
     "parameters": InspectHostArgs.model_json_schema()},
    {"type": "function", "name": "collect_evidence", "description": "Collect a fixed type of read-only evidence.",
     "parameters": CollectEvidenceArgs.model_json_schema()},
    {"type": "function", "name": "get_recent_incidents", "description": "Read compact incident summaries when history is relevant.",
     "parameters": RecentIncidentsArgs.model_json_schema()},
]


class FunctionExecutor:
    def __init__(self, registry: HostRegistry, inspection: InspectionService, database: Database,
                 remote: Executor, max_chars: int = 2500, secrets: tuple[str, ...] = ()) -> None:
        self.registry = registry; self.inspection = inspection; self.database = database
        self.remote = remote; self.max_chars = max_chars; self.secrets = secrets

    def execute(self, name: str, arguments: dict[str, object], target_host: str) -> dict[str, object]:
        models = {"inspect_host": InspectHostArgs, "collect_evidence": CollectEvidenceArgs,
                  "get_recent_incidents": RecentIncidentsArgs}
        if name not in models:
            return {"error": "FUNCTION_NOT_ALLOWED", "function": name}
        try:
            args = models[name].model_validate(arguments)
        except ValidationError as exc:
            return {"error": "INVALID_ARGUMENTS", "details": exc.errors(include_input=False)}
        if args.host.casefold() != target_host.casefold():
            return {"error": "HOST_MISMATCH"}
        try: host = self.registry.resolve(target_host)
        except KeyError: return {"error": "HOST_NOT_FOUND"}
        if name == "get_recent_incidents":
            result = {"incidents": self.database.recent_incidents(host.name, args.signature, args.limit)}
        else:
            scope = args.scope if name == "inspect_host" else EVIDENCE_GROUP[args.evidence_type]
            record = self.inspection.inspect_and_record(
                host, self.remote, None if scope in {"full", "system"} else scope,
            )
            observation = record.observation
            result = {"observation": {
                "host": observation.host, "observed_at": observation.observed_at.isoformat(),
                "ssh_ok": observation.ssh_ok, "gpu": observation.gpu.model_dump(mode="json"),
                "system": observation.system.model_dump(mode="json"),
                "services": observation.services, "signatures": observation.signatures,
            }}
        compact = redact(json.dumps(result, ensure_ascii=False, default=str), self.secrets)
        if len(compact) > self.max_chars:
            compact = compact[:self.max_chars] + "…[truncated]"
        return {"untrusted_evidence": json.loads(compact) if not compact.endswith("[truncated]") else compact}
