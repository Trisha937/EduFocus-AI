"""Chat memory management for EduFocus.

Implements a 3-turn sliding window chat history to maintain dialogue context
without overwhelming the model with long conversations.
"""

from typing import Any

from langchain_core.messages import HumanMessage, AIMessage


# Maximum number of turns to keep in sliding window
MAX_HISTORY_TURNS = 3


def format_chat_history(messages: list[dict[str, str]]) -> list[Any]:
    """Convert chat history dicts to LangChain message objects.

    Args:
        messages: List of {'role': 'human'|'ai', 'content': str} dicts.

    Returns:
        List of HumanMessage and AIMessage objects for LangChain.
    """
    # Map string role to LangChain message class
    # This allows LangChain's chat models to understand the chat structure
    formatted = []
    for msg in messages:
        if msg["role"] == "human":
            formatted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            formatted.append(AIMessage(content=msg["content"]))
    return formatted


def slice_sliding_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the most recent N turns in chat history.

    A "turn" is a pair: (human question, ai response).
    This implements the 3-turn sliding window memory.

    Args:
        messages: Full chat history list.

    Returns:
        Trimmed history keeping only last 3 turns (6 messages max).
    """
    # Each turn = 2 messages (human + ai). 3 turns = 6 messages max
    # We use negative indexing to get the last N*2 messages efficiently
    return messages[-(MAX_HISTORY_TURNS * 2):]


def add_message(messages: list[dict[str, str]], role: str, content: str) -> list[dict[str, str]]:
    """Append a new message and return updated history.

    Args:
        messages: Current chat history list.
        role: 'human' or 'ai'.
        content: Message content.

    Returns:
        Updated chat history with sliding window applied.
    """
    # Create a new message dict and append it
    messages.append({"role": role, "content": content})

    # Apply sliding window to limit context size
    return slice_sliding_history(messages)