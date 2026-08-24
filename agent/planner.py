"""Планировщик поиска: решает, искать ли в базе и по каким подзапросам."""

from __future__ import annotations

import json
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


def parse_plan_json(text: str) -> SearchPlan | None:
    """Разбирает план из свободного текста модели.

    Запасной путь на случай, когда структурированный вывод не сработал.
    Модели по-разному оформляют JSON: заворачивают в ограждение ```json,
    добавляют пояснение до или после, а gpt-oss при выключенных
    размышлениях возвращает пустую строку. Поэтому JSON вырезается из
    текста, а не парсится целиком.

    Args:
        text: Ответ модели как есть.

    Returns:
        SearchPlan | None: Разобранный план либо ``None``, если в тексте
        нет пригодного JSON.
    """
    if not text or not text.strip():
        return None

    body = text.strip()
    if "```" in body:
        parts = body.split("```")
        body = parts[1] if len(parts) > 1 else body
        body = body.removeprefix("json").strip()

    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        data = json.loads(body[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    mode = data.get("mode")
    if mode not in ("chat", "kb"):
        return None
    raw = data.get("subqueries") or []
    if not isinstance(raw, list):
        return None
    return SearchPlan(mode=mode, subqueries=[str(q) for q in raw if str(q).strip()])


async def _plan_via_text(model, question: str, cap: int) -> SearchPlan | None:
    """Просит план обычным сообщением, без структурированного вывода.

    Args:
        model: Клиент модели.
        question: Исходный запрос пользователя.
        cap: Предел числа подзапросов.

    Returns:
        SearchPlan | None: План либо ``None``, если и так не вышло.
    """
    try:
        answer = await model.ainvoke(
            messages=[
                SystemMessage(PLAN_PROMPT.format(max_subqueries=cap)),
                HumanMessage(question),
            ]
        )
    except Exception as exc:
        logger.warning("запасной разбор плана не выполнен: %s", exc)
        return None
    return parse_plan_json(str(getattr(answer, "content", "") or ""))


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
        # Структурированный вывод ломается по-разному у разных моделей: у
        # gpt-oss при выключенных размышлениях ответ приходит пустым, и
        # разбор падает с «Invalid json output:» без содержимого. Раньше это
        # молча уводило на поиск по исходному вопросу, то есть декомпозиция
        # не работала вовсе, а в логе оставалось одно предупреждение. Пробуем
        # получить план обычным сообщением: разбор JSON из текста переживает
        # и ограждения ```json, и пояснения вокруг.
        logger.warning("структурированный план не получен (%s), пробуем текстом", exc)
        plan = await _plan_via_text(model, question, cap)

    if plan is None:
        logger.warning(
            "планировщик не дал плана: декомпозиция пропущена, "
            "ищем по исходному вопросу целиком"
        )
        return SearchPlan(mode="kb", subqueries=[_clean(question)])

    if plan.mode == "chat":
        return SearchPlan(mode="chat")

    subqueries = [_clean(q) for q in plan.subqueries if _clean(q)]
    if not subqueries:
        subqueries = [_clean(question)]
    return SearchPlan(mode="kb", subqueries=subqueries[:cap])
