from __future__ import annotations

import asyncio

import discord


class ApprovalView(discord.ui.View):
    """Thin UI: custom IDs carry only the DB proposal ID."""

    def __init__(self, proposal_id: int, coordinator, timeout: float = 600) -> None:
        super().__init__(timeout=timeout)
        self.proposal_id, self.coordinator = proposal_id, coordinator
        approve = discord.ui.Button(label="Approve", style=discord.ButtonStyle.danger,
                                    custom_id=f"proposal:approve:{proposal_id}")
        reject = discord.ui.Button(label="Reject", style=discord.ButtonStyle.secondary,
                                   custom_id=f"proposal:reject:{proposal_id}")
        approve.callback = self._approve
        reject.callback = self._reject
        self.add_item(approve); self.add_item(reject)

    async def _approve(self, interaction: discord.Interaction) -> None:
        channel_id = interaction.channel_id or 0
        ok, message = await asyncio.to_thread(
            self.coordinator.approve, self.proposal_id, user_id=interaction.user.id,
            channel_id=channel_id, is_bot=interaction.user.bot,
        )
        if ok or message != "NOT_AUTHORIZED": self._disable()
        await interaction.response.edit_message(content=message, view=self)

    async def _reject(self, interaction: discord.Interaction) -> None:
        channel_id = interaction.channel_id or 0
        ok, message = await asyncio.to_thread(
            self.coordinator.reject, self.proposal_id, user_id=interaction.user.id,
            channel_id=channel_id, is_bot=interaction.user.bot,
        )
        if ok: self._disable()
        await interaction.response.edit_message(content=message, view=self)

    def _disable(self) -> None:
        for child in self.children:
            child.disabled = True
        self.stop()
