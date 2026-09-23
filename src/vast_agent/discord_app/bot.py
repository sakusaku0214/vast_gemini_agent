from __future__ import annotations

import discord

from vast_agent.discord_app.auth import DiscordGuard
from vast_agent.discord_app.formatting import split_messages, wants_continue_button
from vast_agent.discord_app.gateway import DiscordGateway


class ContinueView(discord.ui.View):
    def __init__(self, guard: DiscordGuard, service, source_job_id: int) -> None:
        super().__init__(timeout=1800)
        self.guard = guard
        self.service = service
        self.source_job_id = source_job_id

    @discord.ui.button(label="続けて", style=discord.ButtonStyle.primary)
    async def continue_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.guard.accepts_interaction(interaction):
            await interaction.response.send_message("OWNERのみ操作できます。", ephemeral=True)
            return

        channel = interaction.channel
        if channel is None or int(channel.id) != self.guard.channel_id:
            await interaction.response.send_message("このchannelでは操作できません。", ephemeral=True)
            return

        state = self.service.conversations.get(self.guard.owner_id, self.guard.channel_id)
        if state.last_job_id != self.source_job_id:
            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send(
                "この［続けて］ボタンは古くなっています。最新の調査から続行してください。",
                ephemeral=True,
            )
            return

        button.disabled = True
        await interaction.response.edit_message(view=self)
        await channel.send("🔎 続きを調査中...")

        reply = await self.service.handle_question(
            "続けて",
            self.guard.owner_id,
            self.guard.channel_id,
        )
        chunks = split_messages(reply.text)
        show_continue = reply.job_id is not None and wants_continue_button(reply.text)
        for index, chunk in enumerate(chunks):
            if show_continue and index == len(chunks) - 1:
                await channel.send(
                    chunk,
                    view=ContinueView(self.guard, self.service, reply.job_id),
                )
            else:
                await channel.send(chunk)


class ProposalView(discord.ui.View):
    def __init__(self, guard: DiscordGuard, service, proposal_id: int) -> None:
        super().__init__(timeout=1800)
        self.guard = guard
        self.service = service
        self.proposal_id = proposal_id

    def _authorized(self, interaction: discord.Interaction) -> bool:
        channel = interaction.channel
        return (
            self.guard.accepts_interaction(interaction)
            and channel is not None
            and int(channel.id) == self.guard.channel_id
        )

    def _disable_all(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

    @discord.ui.button(label="承認", style=discord.ButtonStyle.success)
    async def approve_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._authorized(interaction):
            await interaction.response.send_message("OWNERのみ操作できます。", ephemeral=True)
            return

        await interaction.response.defer()
        reply = await self.service.handle_question(
            f"承認 #{self.proposal_id}",
            self.guard.owner_id,
            self.guard.channel_id,
        )
        self._disable_all()
        await interaction.edit_original_response(view=self)
        channel = interaction.channel
        if channel is not None:
            for chunk in split_messages(reply.text):
                await channel.send(chunk)

    @discord.ui.button(label="拒否", style=discord.ButtonStyle.danger)
    async def reject_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._authorized(interaction):
            await interaction.response.send_message("OWNERのみ操作できます。", ephemeral=True)
            return

        reply = await self.service.handle_question(
            f"拒否 #{self.proposal_id}",
            self.guard.owner_id,
            self.guard.channel_id,
        )
        self._disable_all()
        await interaction.response.edit_message(view=self)
        await interaction.followup.send(reply.text)



def create_bot(owner_id: int, channel_id: int, service) -> discord.Client:
    intents = discord.Intents.default(); intents.message_content = True
    client = discord.Client(intents=intents)
    guard = DiscordGuard(owner_id, channel_id)
    gateway = DiscordGateway(
        guard,
        service,
        continue_view_factory=lambda job_id: ContinueView(guard, service, job_id),
        proposal_view_factory=lambda proposal_id: ProposalView(guard, service, proposal_id),
    )

    @client.event
    async def on_message(message):
        await gateway.on_message(message)

    return client
