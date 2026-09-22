from __future__ import annotations

import discord

from vast_agent.discord_app.auth import DiscordGuard
from vast_agent.discord_app.gateway import DiscordGateway


def create_bot(owner_id: int, channel_id: int, service) -> discord.Client:
    intents = discord.Intents.default(); intents.message_content = True
    client = discord.Client(intents=intents)
    gateway = DiscordGateway(DiscordGuard(owner_id, channel_id), service)

    @client.event
    async def on_message(message):
        await gateway.on_message(message)

    return client
