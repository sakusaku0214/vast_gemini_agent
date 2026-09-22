from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from pydantic import TypeAdapter

from vast_agent.actions.executor import ActionVerifier, TypedActionExecutor
from vast_agent.actions.models import (
    ActionParameters,
    ActionProposal,
    ActionRequest,
    ActionType,
    ProposalStatus,
    VerificationResult,
)
from vast_agent.actions.policies import PolicyDecision, PolicyEngine
from vast_agent.actions.preflight import PreflightProvider
from vast_agent.actions.registry import ActionRegistry
from vast_agent.config import HostRegistry, OperationsSettings
from vast_agent.jobs.locks import HostLockManager, LockClass
from vast_agent.storage.database import Database

type Outcome = tuple[bool, str]


class ActionCoordinator:
    """Proposal/approval/TOCTOU/audit boundary for every non-reboot write."""

    def __init__(self, database: Database, hosts: HostRegistry, settings: OperationsSettings,
                 preflight: PreflightProvider, executor: TypedActionExecutor,
                 verifier: ActionVerifier, owner_id: int, channel_id: int,
                 locks: HostLockManager | None = None) -> None:
        self.db, self.hosts, self.settings = database, hosts, settings
        self.preflight, self.executor, self.verifier = preflight, executor, verifier
        self.owner_id, self.channel_id = owner_id, channel_id
        self.registry, self.policy = ActionRegistry(), PolicyEngine()
        self.locks = locks or HostLockManager()

    def propose(self, request: ActionRequest, created_by: str,
                now: datetime | None = None) -> ActionProposal:
        self.registry.validate_consistency(request)
        host = self.hosts.resolve(request.host)
        state = self.preflight.collect(host, request)
        policy = self.policy.evaluate(request, state, self.settings, execution=False)
        moment = now or datetime.now(UTC)
        status = ProposalStatus.BLOCKED if policy.decision == PolicyDecision.BLOCK else ProposalStatus.PENDING
        proposal = ActionProposal(
            host=host.name, action_type=request.action_type, parameters=request.parameters,
            risk_class=self.registry.risk(request), status=status, created_at=moment,
            expires_at=moment + timedelta(seconds=self.settings.approval_ttl_seconds),
            created_by=created_by, preflight_summary=state,
            preflight_fingerprint=state.fingerprint(request),
        )
        proposal.id = self.db.create_action_proposal(proposal)
        self.db.add_action_event(proposal.id, "PROPOSED", policy.decision)
        return proposal

    def approve(self, proposal_id: int, *, user_id: int, channel_id: int,
                is_bot: bool = False, now: datetime | None = None) -> Outcome:
        if is_bot or user_id != self.owner_id or channel_id != self.channel_id:
            return False, "NOT_AUTHORIZED"
        moment = now or datetime.now(UTC)
        if not self.db.approve_action_proposal(proposal_id, str(user_id), moment):
            return False, "NOT_PENDING_OR_EXPIRED"
        return self._consume(proposal_id)

    def reject(self, proposal_id: int, *, user_id: int, channel_id: int,
               is_bot: bool = False, now: datetime | None = None) -> Outcome:
        if is_bot or user_id != self.owner_id or channel_id != self.channel_id:
            return False, "NOT_AUTHORIZED"
        won = self.db.reject_action_proposal(proposal_id, str(user_id), now or datetime.now(UTC))
        return (won, "REJECTED" if won else "NOT_PENDING_OR_EXPIRED")

    def _consume(self, proposal_id: int) -> Outcome:
        row = self.db.get_action_proposal(proposal_id)
        if not row: return False, "PROPOSAL_NOT_FOUND"
        request = ActionRequest(host=str(row["host"]), action_type=ActionType(str(row["action_type"])),
            parameters=TypeAdapter(ActionParameters).validate_python(json.loads(str(row["params_json"]))))
        if request.action_type == ActionType.HOST_REBOOT:
            self.db.set_proposal_status(proposal_id, ProposalStatus.INVALIDATED, "REBOOT_COORDINATOR_REQUIRED")
            return False, "REBOOT_COORDINATOR_REQUIRED"
        host = self.hosts.resolve(request.host)
        with self.locks.acquire(host.name, LockClass.WRITE):
            fresh = self.preflight.collect(host, request)
            result = self.policy.evaluate(request, fresh, self.settings, execution=True)
            if result.decision == PolicyDecision.BLOCK:
                self.db.set_proposal_status(proposal_id, ProposalStatus.INVALIDATED, ",".join(result.reasons))
                return False, "PREFLIGHT_BLOCKED"
            if fresh.fingerprint(request) != row["preflight_fingerprint"]:
                self.db.set_proposal_status(proposal_id, ProposalStatus.INVALIDATED, "PREFLIGHT_CHANGED")
                return False, "PREFLIGHT_CHANGED"
            run_id = self.db.create_action_run(proposal_id)
            self.db.set_proposal_status(proposal_id, ProposalStatus.EXECUTING)
            command = self.executor.execute(host, request)  # exactly once; never retried
            verification: VerificationResult = self.verifier.verify(host, request)
            if command.success and verification.success:
                status, code = ProposalStatus.SUCCEEDED, None
            elif command.timed_out:
                status, code = ProposalStatus.FAILED, "UNKNOWN_AFTER_TIMEOUT"
            elif command.success:
                status, code = ProposalStatus.FAILED_VERIFY, "FAILED_VERIFY"
            else:
                status, code = ProposalStatus.FAILED, str(command.error_code or "COMMAND_FAILED")
            self.db.finish_action_run(run_id, status, command.exit_code, code, verification.summary)
            self.db.set_proposal_status(proposal_id, status)
            return status == ProposalStatus.SUCCEEDED, status
