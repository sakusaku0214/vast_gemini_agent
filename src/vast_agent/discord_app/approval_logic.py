from __future__ import annotations

import asyncio
from collections.abc import Callable


async def handle_approval(interaction, coordinator, proposal_id: int,
                          disable: Callable[[], None]) -> None:
    """ACK an authorized interaction before running potentially multi-minute work."""
    channel_id = interaction.channel_id or 0
    if (interaction.user.bot or interaction.user.id != coordinator.owner_id or
            channel_id != coordinator.channel_id):
        await interaction.response.send_message("OWNERのみ承認できます。", ephemeral=True)
        return
    await interaction.response.defer()
    await interaction.edit_original_response(
        content=f"⚙️ Proposal #{proposal_id} approved. Preflightを再確認しています...",
        view=getattr(disable, "__self__", None),
    )
    _, message = await asyncio.to_thread(
        coordinator.approve, proposal_id, user_id=interaction.user.id,
        channel_id=channel_id, is_bot=interaction.user.bot,
    )
    disable()
    await interaction.edit_original_response(
        content=message, view=getattr(disable, "__self__", None),
    )
