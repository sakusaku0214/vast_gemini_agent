from __future__ import annotations

from dataclasses import dataclass

from vast_agent.storage.database import Database


@dataclass
class ConversationState:
    last_host: str | None = None
    last_job_id: int | None = None
    last_scope: str | None = None
    last_recommended_action: str | None = None


class ConversationStore:
    def __init__(self, database: Database) -> None:
        self.database = database
        database.migrate()

    def get(self, owner: int | str, channel: int | str) -> ConversationState:
        row = self.database.load_conversation(str(owner), str(channel))
        return ConversationState(
            last_host=row.get("last_host") if row else None,
            last_job_id=row.get("last_job_id") if row else None,
            last_scope=row.get("last_scope") if row else None,
            last_recommended_action=row.get("last_recommended_action") if row else None,
        )

    def save(self, owner: int | str, channel: int | str, state: ConversationState) -> None:
        self.database.save_conversation(str(owner), str(channel), state.last_host,
                                        state.last_job_id, state.last_scope,
                                        state.last_recommended_action)
