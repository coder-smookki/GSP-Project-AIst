"""Настройки чата доходят до запроса к агенту.

Настройка, которую видно в интерфейсе, но которая не влияет на поиск, хуже
отсутствующей: человек думает, что сузил выдачу, а она прежняя.
"""

from __future__ import annotations

from gui.contract import build_input


def test_section_types_reach_the_request() -> None:
    """Выбранные типы разделов уходят в запрос.

    Поиск умел фильтровать по ним с самого начала, но интерфейс их не
    отдавал - фильтр существовал и был недостижим.
    """
    payload = build_input(
        "Какой порядок допуска?",
        sections=["procedura"],
        areas=["ohrana_truda"],
    )

    assert payload["sections"] == ["procedura"]
    assert payload["areas"] == ["ohrana_truda"]


def test_empty_selection_means_the_whole_base() -> None:
    """Пустой выбор не превращается в фильтр по пустому множеству."""
    payload = build_input("вопрос", sections=[], areas=[])

    assert "sections" not in payload
    assert "areas" not in payload


def test_forced_search_reaches_the_request() -> None:
    """Переключатель «всегда искать в базе» доходит до графа."""
    payload = build_input("спасибо", always_answer_with_db=True)

    assert payload["always_answer_with_db"] is True
