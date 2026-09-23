from __future__ import annotations

import json
from dataclasses import dataclass, field

from vast_agent.storage.database import Database


@dataclass
class ConversationState:
    last_host: str | None = None
    last_job_id: int | None = None
    last_scope: str | None = None
    last_recommended_action: str | None = None
    write_context_host: str | None = None
    current_hosts: tuple[str, ...] = ()
    context_kind: str = "none"
    context_source: str | None = None
    investigation_context: dict[str, object] = field(default_factory=dict)


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
            write_context_host=row.get("write_context_host") if row else None,
            current_hosts=tuple(json.loads(str(row.get("context_hosts_json") or "[]"))) if row else (),
            context_kind=str(row.get("context_kind") or "none") if row else "none",
            context_source=row.get("context_source") if row else None,
            investigation_context=json.loads(
                str(row.get("investigation_context_json") or "{}"),
            ) if row else {},
        )

    def save(self, owner: int | str, channel: int | str, state: ConversationState) -> None:
        self.database.save_conversation(str(owner), str(channel), state.last_host,
                                        state.last_job_id, state.last_scope,
                                        state.last_recommended_action, state.write_context_host,
                                        json.dumps(state.current_hosts), state.context_kind,
                                        state.context_source,
                                        json.dumps(state.investigation_context, ensure_ascii=False))
