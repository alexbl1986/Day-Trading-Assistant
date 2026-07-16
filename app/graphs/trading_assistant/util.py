"""Small message helpers shared across nodes."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AnyMessage, HumanMessage


def message_text(msg: AnyMessage) -> str:
    """Plain text of a message, tolerant of streaming content-block lists."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(parts)


def latest_user_text(messages: Sequence[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return message_text(msg)
    return message_text(messages[-1]) if messages else ""
