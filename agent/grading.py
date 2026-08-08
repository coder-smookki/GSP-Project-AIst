from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from .prompts import GRADE_PROMPT

logger = logging.getLogger(__name__)

RELEVANT = "Р"
PARTIAL = "Ч"
IRRELEVANT = "Н"

Verdict = Literal["Р", "Ч", "Н"]


class GradeResult(BaseModel):
    v: list[Verdict] = Field(description="вердикты по фрагментам по порядку")


def _format_chunks(items: list[dict], snippet_chars: int) -> str:
    return "\n\n".join(
        f"[{n}] {(item.get('text') or item.get('snippet') or '')[:snippet_chars]}"
        for n, item in enumerate(items, 1)
    )


async def grade_chunks(
    model,
    question: str,
    items: list[dict],
    *,
    keep_partial: bool = True,
    snippet_chars: int = 400,
) -> list[dict] | None:
    if not items:
        return []

    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        structured = model.raw.with_structured_output(GradeResult)
        result = await structured.ainvoke(
            [
                SystemMessage(GRADE_PROMPT.format(count=len(items))),
                HumanMessage(
                    f"ВОПРОС: {question}\n\n"
                    f"ФРАГМЕНТЫ:\n{_format_chunks(items, snippet_chars)}"
                ),
            ]
        )
    except Exception as exc:
        logger.warning("грейдинг не выполнен, выдача остаётся как есть: %s", exc)
        return None

    verdicts = list(getattr(result, "v", []) or [])
    if len(verdicts) != len(items):
        logger.warning(
            "грейдер вернул %d вердиктов на %d фрагментов — выдача остаётся как есть",
            len(verdicts),
            len(items),
        )
        return None

    allowed = {RELEVANT, PARTIAL} if keep_partial else {RELEVANT}
    kept = [
        item
        for item, verdict in zip(items, verdicts, strict=True)
        if verdict in allowed
    ]

    logger.info(
        "грейдер: из %d фрагментов оставлено %d (вердикты: %s)",
        len(items),
        len(kept),
        "".join(verdicts),
    )
    return kept
