from __future__ import annotations


def split_messages(text: str, limit: int = 1900) -> list[str]:
    """Split on lines for Discord's 2000-character message limit."""
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit and current:
            chunks.append(current.rstrip()); current = ""
        while len(line) > limit:
            chunks.append(line[:limit]); line = line[limit:]
        current += line
    if current or not chunks: chunks.append(current.rstrip())
    return chunks
