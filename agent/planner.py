"""Планировщик поиска: решает, искать ли в базе и по каким подзапросам."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import PLAN_PROMPT
from .state import SearchPlan

logger = logging.getLogger(__name__)


def max_subqueries(max_requests: int, retries: int) -> int:
    """Верхняя граница числа подзапросов по бюджету обращений к поиску.

    Каждый подзапрос исполняется веткой-агентом, которая тратит
    ``retries + 1`` обращений (первый вызов плюс повторы). Бюджет делится
    на ветки поровну, но не ниже одного подзапроса.

    Args:
        max_requests: Общий бюджет обращений к поиску на запрос.
        retries: Повторы поиска в ветке-агенте.

    Returns:
        int: Сколько подзапросов максимум может построить планировщик.
    """
    return max(1, max_requests // (retries + 1))


def _clean(text: str) -> str:
    """Убирает пустые края запроса.

    Args:
        text: Строка запроса.

    Returns:
        str: Строка без ведущих и завершающих пробелов.
    """
    return text.strip()


async def plan_search(
    model,
    question: str,
    *,
    max_requests: int = 10,
    retries: int = 1,
    always_answer_with_db: bool = False,
) -> SearchPlan:
    """Строит план поиска: режим и подзапросы.

    При любом сбое структурированного вывода возвращается безопасный
    фолбэк — искать по исходному вопросу. Флаг ``always_answer_with_db``
    принудительно включает kb-режим без обращения к модели: он закрывает
    случай, когда модель не вызывает поиск там, где он нужен.

    Args:
        model: Клиент модели со свойством ``raw`` (структурированный вывод).
        question: Исходный запрос пользователя.
        max_requests: Бюджет обращений к поиску на запрос.
        retries: Повторы поиска в ветке-агенте.
        always_answer_with_db: Всегда искать в базе, игнорируя mode="chat".

    Returns:
        SearchPlan: План: режим и подзапросы, не длиннее бюджета.
    """
    if always_answer_with_db:
        return SearchPlan(mode="kb", subqueries=[_clean(question)])

    cap = max_subqueries(max_requests, retries)
    try:
        structured = model.raw.with_structured_output(SearchPlan)
        plan = await structured.ainvoke(
            [
                SystemMessage(PLAN_PROMPT.format(max_subqueries=cap)),
                HumanMessage(question),
            ]
        )
    except Exception as exc:
        logger.warning("планировщик не выполнен, ищем по исходному вопросу: %s", exc)
        return SearchPlan(mode="kb", subqueries=[_clean(question)])

    if plan is None:
        logger.warning("планировщик вернул пусто — ищем по исходному вопросу")
        return SearchPlan(mode="kb", subqueries=[_clean(question)])

    if plan.mode == "chat":
        return SearchPlan(mode="chat")

    subqueries = [_clean(q) for q in plan.subqueries if _clean(q)]
    if not subqueries:
        subqueries = [_clean(question)]
    return SearchPlan(mode="kb", subqueries=subqueries[:cap])