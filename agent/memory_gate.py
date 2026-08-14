from __future__ import annotations

import re

_MARKERS = (
    r"\bя\b",
    r"\bмне\b",
    r"\bменя\b",
    r"\bмной\b",
    r"\bмо[йяеё]\b",
    r"\bмои\b",
    r"\bмоего\b",
    r"\bмоей\b",
    r"\bмоих\b",
    r"\bмы\b",
    r"\bнам\b",
    r"\bнас\b",
    r"\bнаш\w*\b",
    r"\bу\s+меня\b",
    r"\bзовут\b",
    r"\bработаю\b",
    r"\bтружусь\b",
    r"\bдолжност\w*\b",
    r"\bя\s+из\b",
    r"\bзапомни\w*\b",
    r"\bзаполни\s+профил\w*\b",
    r"\bне\s+забуд\w*\b",
)

_PATTERN = re.compile("|".join(_MARKERS), re.IGNORECASE)


def mentions_user(text: str) -> bool:
    if not text or not text.strip():
        return False
    return bool(_PATTERN.search(text.replace("ё", "е")))


def worth_extracting(messages: list) -> bool:
    human_texts: list[str] = []
    for message in reversed(messages):
        role = getattr(message, "type", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        if role not in ("human", "user"):
            continue
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        human_texts.append(str(content or ""))
        if len(human_texts) == 2:
            break

    return any(mentions_user(text) for text in human_texts)
