from __future__ import annotations

import logging

from .prompts import REWRITE_PROMPT

logger = logging.getLogger(__name__)

MAX_REWRITE_CHARS = 300


async def rewrite_query(model, question: str) -> str | None:
    from langchain_core.messages import HumanMessage

    text = question.strip()
    if not text:
        return None

    try:
        answer = await model.ainvoke(
            messages=[HumanMessage(REWRITE_PROMPT.format(question=text))]
        )
    except Exception as exc:
        logger.warning("переписывание запроса не выполнено: %s", exc)
        return None

    rewritten = str(getattr(answer, "content", "") or "").strip().strip('"«»')
    rewritten = rewritten.splitlines()[0].strip() if rewritten else ""

    if not rewritten or len(rewritten) > MAX_REWRITE_CHARS:
        logger.info("переписывание отвергнуто: результат пуст или слишком длинный")
        return None
    if rewritten.casefold() == text.casefold():
        logger.info("переписывание ничего не изменило")
        return None

    logger.info("запрос переписан: %r -> %r", text[:60], rewritten[:60])
    return rewritten
