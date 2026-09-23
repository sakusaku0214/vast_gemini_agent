from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vast_agent.storage.database import Database


@dataclass(frozen=True)
class Proposal:
    id: int
    owner_id: str
    channel_id: str
    host: str
    argv: tuple[str, ...]
    reason: str
    timeout: int
    status: str


class ProposalStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(
        self,
        owner_id: str,
        channel_id: str,
        host: str,
        argv: tuple[str, ...],
        reason: str,
        timeout: int,
    ) -> Proposal:
        proposal_id = self.database.create_proposal(
            owner_id, channel_id, host, list(argv), reason, timeout, datetime.now(UTC),
        )
        return Proposal(
            proposal_id, owner_id, channel_id, host, argv, reason, timeout, "PENDING",
        )

    def get(self, proposal_id: int) -> Proposal | None:
        row = self.database.get_proposal(proposal_id)
        return self._from_row(row) if row else None

    def pending_for(self, owner_id: str, channel_id: str) -> list[Proposal]:
        return [
            self._from_row(row)
            for row in self.database.pending_proposals(owner_id, channel_id)
        ]

    def reject(self, proposal_id: int, owner_id: str, channel_id: str) -> bool:
        return self.database.reject_proposal(proposal_id, owner_id, channel_id)

    def claim(self, proposal_id: int, owner_id: str, channel_id: str) -> Proposal | None:
        row = self.database.claim_proposal(proposal_id, owner_id, channel_id)
        return self._from_row(row) if row else None

    def mark_executed(self, proposal_id: int) -> None:
        self.database.set_proposal_status(proposal_id, "EXECUTED")

    def mark_failed(self, proposal_id: int) -> None:
        self.database.set_proposal_status(proposal_id, "FAILED")

    @staticmethod
    def _from_row(row: dict[str, object]) -> Proposal:
        return Proposal(
            id=int(row["id"]),
            owner_id=str(row["owner_id"]),
            channel_id=str(row["channel_id"]),
            host=str(row["host"]),
            argv=tuple(json.loads(str(row["argv_json"]))),
            reason=str(row["reason"]),
            timeout=int(row["timeout"]),
            status=str(row["status"]),
        )
