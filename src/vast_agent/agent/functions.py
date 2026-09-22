from __future__ import annotations

import json
from typing import Final

from pydantic import ValidationError

from vast_agent.agent.host_read import (
    FUNCTION_DECLARATIONS as FUNCTION_DECLARATIONS,
)
from vast_agent.agent.host_read import (
    HOST_READ_CAPABILITIES,
    discover,
    execute_diagnostic,
    execute_generic,
)
from vast_agent.config import HostRegistry
from vast_agent.execution.base import Executor, redact
from vast_agent.services.inspection import InspectionService
from vast_agent.storage.database import Database
from vast_agent.tools.registry import run_tool

EVIDENCE_TOOL: Final = {
    "gpu_status": "get_gpu_status", "gpu_processes": "get_gpu_processes",
    "pci": "get_pci_status", "kernel_gpu_errors": "get_kernel_gpu_errors",
    "d_state": "get_d_state_processes", "vast_status": "get_vast_status",
    "vast_logs": "get_vast_logs", "docker": "get_docker_status", "vm": "get_vm_status",
    "services": "get_service_status", "journal_errors": "get_journal_errors",
    "system_health": "get_system_health",
}

class FunctionExecutor:
    def __init__(self, registry: HostRegistry, inspection: InspectionService, database: Database,
                 remote: Executor, max_chars: int = 2500, secrets: tuple[str, ...] = ()) -> None:
        self.registry = registry; self.inspection = inspection; self.database = database
        self.remote = remote; self.max_chars = max_chars; self.secrets = secrets

    def execute(self, name: str, arguments: dict[str, object], target_host: str) -> dict[str, object]:
        capability = HOST_READ_CAPABILITIES.get(name)
        if capability is None or not capability.available:
            return {"error": "FUNCTION_NOT_ALLOWED", "function": name}
        try:
            args = capability.argument_model.model_validate(arguments)
        except ValidationError as exc:
            return {"error": "INVALID_ARGUMENTS", "details": exc.errors(include_input=False)}
        if args.host.casefold() != target_host.casefold():
            return {"error": "HOST_MISMATCH"}
        try: host = self.registry.resolve(target_host)
        except KeyError: return {"error": "HOST_NOT_FOUND"}
        if not host.enabled:
            return {"error": "HOST_DISABLED"}
        if any(not getattr(host.capabilities, value, False)
               for value in capability.required_host_capabilities):
            return {"error": "TOOL_UNSUPPORTED"}
        if name == "list_host_capabilities":
            result = discover(host)
        elif capability.executor == "diagnostic":
            result = execute_diagnostic(name, host, self.remote)
        elif capability.executor == "generic":
            result = execute_generic(name, args, host, self.remote)
        elif name == "get_recent_incidents":
            result = {"incidents": self.database.recent_incidents(host.name, args.signature, args.limit)}
        elif name == "collect_evidence":
            tool_name = EVIDENCE_TOOL[args.evidence_type]
            tool_result = run_tool(tool_name, host, self.remote)
            result = {
                "evidence_type": args.evidence_type,
                "tool": tool_name,
                "success": tool_result.success,
                "exit_code": tool_result.exit_code,
                "error_code": tool_result.error_code,
                "evidence_excerpt": "\n".join(
                    part for part in (tool_result.stdout, tool_result.stderr) if part
                ),
            }
        else:
            scope = args.scope
            record = self.inspection.inspect_and_record(
                host, self.remote, None if scope == "full" else scope,
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
