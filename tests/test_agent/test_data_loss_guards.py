"""Защита от потери данных на неудачном ответе вспомогательной модели.

Два узла графа удаляют данные по итогам генерации: ``condense_context``
выбрасывает сообщения, а ``write_memory`` - записи долгосрочной памяти. Оба
удаления необратимы, поэтому пустой ответ модели не должен их запускать.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from agent.graph import create_app_agent
from agent.state import SearchPlan


class _Planner:
    """Планировщик, уводящий разговор мимо поиска."""

    def with_structured_output(self, schema, **kwargs):
        """Отдаёт себя же.

        Args:
            schema: Схема структурированного вывода.
            **kwargs: Прочие настройки.

        Returns:
            _Planner: Тот же объект.
        """
        return self

    async def ainvoke(self, messages, **kwargs):
        """Возвращает план беседы без обращения к базе.

        Args:
            messages: История сообщений.
            **kwargs: Прочие настройки.

        Returns:
            SearchPlan: План режима «беседа».
        """
        return SearchPlan(mode="chat")


class _SupportModel:
    """Вспомогательная модель с программируемым ответом про память."""

    def __init__(self, memory_answer: str) -> None:
        """Готовит модель.

        Args:
            memory_answer: Текст, который модель вернёт узлу памяти.
        """
        self.raw = _Planner()
        self._memory_answer = memory_answer

    async def ainvoke(self, messages=None, tools=None, **kwargs):
        """Отвечает выжимкой памяти.

        Args:
            messages: История сообщений.
            tools: Доступные инструменты.
            **kwargs: Прочие настройки.

        Returns:
            AIMessage: Ответ модели.
        """
        return AIMessage(content=self._memory_answer)


class _Model:
    """Модель ответа."""

    async def ainvoke(self, *args, **kwargs):
        """Возвращает готовый ответ.

        Returns:
            AIMessage: Ответ ассистента.
        """
        return AIMessage(content="Ответ.")


class _EmptySummaryContext:
    """Сворачивание истории, возвращающее пустое резюме."""

    keep_last = 1

    def should_condense(self, messages, summary) -> bool:
        """Считает, что сворачивать пора.

        Args:
            messages: История сообщений.
            summary: Текущее резюме.

        Returns:
            bool: Всегда ``True``.
        """
        return True

    async def condense(self, messages, existing_summary=""):
        """Возвращает пустое резюме, имитируя сбой модели.

        Args:
            messages: История сообщений.
            existing_summary: Прежнее резюме.

        Returns:
            str: Пустая строка.
        """
        return "   "


@pytest.mark.asyncio
async def test_empty_summary_does_not_delete_the_conversation() -> None:
    """Пустое резюме не повод удалять историю диалога.

    Проверялось только ``None``, и пустая строка проходила дальше: сообщения
    удалялись, а взамен не оставалось ничего.
    """
    app = create_app_agent(
        model=_Model(),
        support_model=_SupportModel("[]"),
        context_manager=_EmptySummaryContext(),
        store=InMemoryStore(),
        search=None,
        verify_numbers=False,
    )
    history = [
        HumanMessage(content="Первый вопрос"),
        AIMessage(content="Первый ответ"),
        HumanMessage(content="Второй вопрос"),
    ]
    result = await app.ainvoke({"messages": history})

    kept = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert len(kept) == 2, "оба вопроса пользователя остались в истории"
    assert not result.get("summary")


@pytest.mark.asyncio
async def test_empty_extraction_does_not_wipe_long_term_memory() -> None:
    """Пустая выжимка не стирает уже накопленную память.

    Узел удаляет каждую запись, которой нет в новом наборе, поэтому один
    неудачный ответ модели уносил бы всю память пользователя безвозвратно.
    """
    store = InMemoryStore()
    namespace = ("memories", "ivanov")
    await store.aput(namespace, "k1", {"content": "Имя: Иван"})
    await store.aput(namespace, "k2", {"content": "Обращение: на вы"})

    app = create_app_agent(
        model=_Model(),
        support_model=_SupportModel('{"memories": []}'),
        store=store,
        search=None,
        verify_numbers=False,
    )
    await app.ainvoke(
        {"messages": [HumanMessage(content="Меня зовут Иван")]},
        {"configurable": {"user_id": "ivanov"}},
    )

    remaining = await store.asearch(namespace)
    assert len(remaining) == 2, "обе записи памяти на месте"
