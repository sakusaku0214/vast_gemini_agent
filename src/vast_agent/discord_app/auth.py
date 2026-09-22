from __future__ import annotations


class DiscordGuard:
    def __init__(self, owner_id: int, channel_id: int) -> None:
        self.owner_id = owner_id; self.channel_id = channel_id

    def accepts_message(self, message) -> bool:
        return (not bool(message.author.bot) and int(message.author.id) == self.owner_id
                and int(message.channel.id) == self.channel_id)

    def accepts_interaction(self, interaction) -> bool:
        return int(interaction.user.id) == self.owner_id
