from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from vast_agent.storage.database import Database


@dataclass(frozen=True)
class Proposal:
    id: int
    owner_id: str
    channel_id: str
    host: str
    argv: tuple[str, ...]
    reason: str
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
    ) -> Proposal:
        proposal_id = self.database.create_proposal(
            owner_id, channel_id, host, list(argv), reason, datetime.now(UTC),
        )
        return Proposal(proposal_id, owner_id, channel_id, host, argv, reason, "PENDING")

    def get(self, proposal_id: int) -> Proposal | None:
        row = self.database.get_proposal(proposal_id)
        if row is None:
            return None
        return Proposal(
            id=int(row["id"]),
            owner_id=str(row["owner_id"]),
            channel_id=str(row["channel_id"]),
            host=str(row["host"]),
            argv=tuple(json.loads(str(row["argv_json"]))),
            reason=str(row["reason"]),
            status=str(row["status"]),
        )

    def pending_for(self, owner_id: str, channel_id: str) -> list[Proposal]:
        return [self._from_row(row) for row in self.database.pending_proposals(owner_id, channel_id)]

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
            status=str(row["status"]),
        )
