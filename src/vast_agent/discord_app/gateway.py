from __future__ import annotations

import asyncio
import logging

from vast_agent.discord_app.formatting import split_messages


class DiscordGateway:
    """Framework-light message adapter, directly testable with fake Discord objects."""

    def __init__(self, guard, service) -> None:
        self.guard = guard; self.service = service
        self._tasks: set[asyncio.Task] = set()
        self.log = logging.getLogger("vast_agent.discord")

    async def on_message(self, message) -> None:
        if not self.guard.accepts_message(message): return
        task = asyncio.create_task(self._handle(message))
        self._tasks.add(task); task.add_done_callback(self._tasks.discard)

    async def _handle(self, message) -> None:
        try:
            await message.channel.send("🔎 リクエストを受け付けました。調査中...")
            reply = await self.service.handle_question(
                message.content, message.author.id, message.channel.id,
            )
            for chunk in split_messages(reply.text):
                await message.channel.send(chunk)
        except Exception:
            self.log.exception("Discord request failed")
            await message.channel.send("リクエスト処理に失敗しました。詳細はagent logを確認してください。")

    async def drain(self) -> None:
        if self._tasks: await asyncio.gather(*self._tasks, return_exceptions=True)
