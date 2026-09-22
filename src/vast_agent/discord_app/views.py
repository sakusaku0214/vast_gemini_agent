from __future__ import annotations

import discord

from vast_agent.discord_app.auth import DiscordGuard


class CancelView(discord.ui.View):
    def __init__(self, job_id: int, jobs, guard: DiscordGuard, timeout: float = 900) -> None:
        super().__init__(timeout=timeout); self.job_id = job_id; self.jobs = jobs; self.guard = guard

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.guard.accepts_interaction(interaction): return True
        await interaction.response.send_message("OWNERのみ操作できます。", ephemeral=True)
        return False

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        result = self.jobs.cancel(self.job_id)
        await interaction.response.send_message(f"Job #{self.job_id}: {result}", ephemeral=True)
        self.stop()
