"""Conversation-state resolution for multi-turn support requests."""

from __future__ import annotations

import re
from typing import Iterable

from .models import ConversationTurn

FOLLOW_UP_PATTERNS = [
    r"\bwhat should i try\b", r"\bwhat can i try\b", r"\bhow (?:do|can) i fix (?:it|this|that)\b",
    r"\bwhat should i do\b", r"\banything else i can try\b", r"\bit still (?:doesn't|does not|won't|will not) work\b",
    r"\bi already (?:tried|did|restarted|reset)\b", r"\bhow do i do that\b", r"\bwill .*\b(?:delete|remove|erase)\b",
    r"\bdoes .*\b(?:delete|remove|erase|affect)\b", r"\bwhy is (?:it|this) happening\b", r"\bdoes that affect\b",
]
NEW_TOPIC_MARKERS = re.compile(r"\b(?:also|another|different|separately|by the way)\b", re.I)
REFERENTIAL = re.compile(r"\b(?:it|this|that|still|already|there|same)\b", re.I)


def _looks_like_new_problem(message: str) -> bool:
    return bool(re.search(r"\b(?:won't|can't|cannot|doesn't|isn't|not working|keeps?|unable|broken|crash\w*|freez\w*|disconnect\w*|drain\w*|charg\w*|fail\w*|problem|issue|stuck|unresponsive)\b", message, re.I))


def _is_follow_up(message: str) -> bool:
    text = message.strip().lower()
    if text.isdigit():
        return False
    if any(re.search(pattern, text, re.I) for pattern in FOLLOW_UP_PATTERNS):
        return True
    if _looks_like_new_problem(message):
        return bool(REFERENTIAL.search(message))
    return len(text.split()) <= 7 and bool(REFERENTIAL.search(message))


def build_contextual_query(message: str, conversation: Iterable[ConversationTurn] | None = None) -> str:
    """Use recent USER turns as context and never use assistant text as customer context."""
    turns = list(conversation or [])
    user_history = [turn.content.strip() for turn in turns if turn.role == "user" and turn.content.strip()]
    current = message.strip()
    if not user_history:
        return current
    if NEW_TOPIC_MARKERS.search(current) and _looks_like_new_problem(current):
        return current
    if _is_follow_up(current):
        return " ".join(user_history[-4:] + [current])
    return current
