"""Изоляция ходов диалога: второй вопрос не отвечает выдержками к первому.

Замеры качества берут новый ``thread_id`` на каждый вопрос, поэтому накопление
состояния между ходами они увидеть не могли в принципе. Здесь один и тот же
тред используется несколько раз - как в живом интерфейсе.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from agent.graph import create_app_agent
from agent.state import RESET_CHANNEL, SearchPlan, merge_or_reset
from infrastructure.retrieval.qdrant_search import SearchUnavailable
from infrastructure.retrieval.search import Chunk


class _Planner:
    """Планировщик, доступный модели через ``.raw``."""

    def __init__(self, subquery: str) -> None:
        """Готовит планировщик под один подзапрос.

        Args:
            subquery: Подзапрос, который вернёт план.
        """
        self._subquery = subquery

    def with_structured_output(self, schema, **kwargs):
        """Отдаёт себя же: план фиксирован.

        Args:
            schema: Схема структурированного вывода.
            **kwargs: Прочие настройки, не используются.

        Returns:
            _Planner: Тот же объект.
        """
        return self

    async def ainvoke(self, messages, **kwargs):
        """Возвращает план поиска.

        Args:
            messages: История сообщений.
            **kwargs: Прочие настройки.

        Returns:
            SearchPlan: План с одним подзапросом.
        """
        return SearchPlan(mode="kb", subqueries=[self._subquery])


class _SupportModel:
    """Модель планирования (через ``.raw``) и веточного поиска."""

    def __init__(self, subquery: str) -> None:
        """Готовит модель.

        Args:
            subquery: Подзапрос, который вернёт планировщик.
        """
        self.raw = _Planner(subquery)
        self._subquery = subquery
        self._called: set[int] = set()

    async def ainvoke(self, messages=None, tools=None, **kwargs):
        """Просит поиск один раз на ветку, затем завершает её.

        Args:
            messages: История сообщений ветки.
            tools: Доступные инструменты.
            **kwargs: Прочие настройки.

        Returns:
            AIMessage: Вызов инструмента или завершающий ответ.
        """
        if not tools:
            return AIMessage(content="Ответ.")
        if any(getattr(m, "type", "") == "tool" for m in (messages or [])):
            return AIMessage(content="Готово.")
        return AIMessage(
            content="ищу",
            tool_calls=[
                {
                    "name": "RetrieveQuery",
                    "args": {"query": self._subquery},
                    "id": "call-1",
                }
            ],
        )


class _Model:
    """Модель ответа."""

    async def ainvoke(self, *args, **kwargs):
        """Возвращает готовый ответ.

        Returns:
            AIMessage: Ответ ассистента.
        """
        return AIMessage(content="Ответ.")


def _chunk(n: int) -> Chunk:
    """Собирает фрагмент, привязанный к номеру хода.

    Args:
        n: Номер хода.

    Returns:
        Chunk: Фрагмент базы знаний.
    """
    return Chunk(
        chunk_id=f"R-U2-{n:02d}#c001",
        doc_code=f"R-U2-{n:02d}",
        doc_title=f"Документ хода {n}",
        section="4",
        section_type="procedura",
        clause="4.1",
        text=f"выдержка хода {n}",
    )


class _TurnSearch:
    """Поиск, отдающий на каждый ход свой фрагмент."""

    def __init__(self, *, fail_first: bool = False) -> None:
        """Готовит поиск.

        Args:
            fail_first: Уронить первый вызов, имитируя недоступность индекса.
        """
        self.turn = 0
        self._fail_first = fail_first

    async def query(self, query, sections=None, limit=None, areas=None, docs=None):
        """Возвращает один фрагмент, свой для каждого хода.

        Args:
            query: Текст подзапроса.
            sections: Типы разделов.
            limit: Предел выдачи.
            areas: Области знаний.
            docs: Коды документов.

        Returns:
            list[tuple[Chunk, float]]: Одна пара «фрагмент, оценка».

        Raises:
            SearchUnavailable: На первом ходе, если так задано.
        """
        self.turn += 1
        if self._fail_first and self.turn == 1:
            raise SearchUnavailable("индекс недоступен")
        return [(_chunk(self.turn), 0.9)]


def _app(search):
    """Собирает агента с чекпоинтером в памяти.

    Args:
        search: Реализация поиска.

    Returns:
        Скомпилированный граф с сохранением состояния между ходами.
    """
    return create_app_agent(
        model=_Model(),
        support_model=_SupportModel("подзапрос"),
        store=InMemoryStore(),
        checkpointer=MemorySaver(),
        search=search,
        max_requests=10,
        branch_retries=0,
        top_k=5,
        verify_numbers=False,
    )


@pytest.mark.asyncio
async def test_second_question_does_not_answer_with_first_questions_sources() -> None:
    """Источники прошлого хода не попадают в ответ на следующий вопрос.

    Каналы ``branch_retrieved`` и ``failures`` складываются reducer-сумматором,
    и возврат пустого списка их не очищал: ``current + []`` это ``current``.
    Фрагменты копились от вопроса к вопросу, а ``_merge_branches`` раздавал
    слоты по кругу, поэтому со второго вопроса большую часть контекста
    занимали выдержки к предыдущим.
    """
    app = _app(_TurnSearch())
    config = {"configurable": {"thread_id": "один-диалог"}}

    first = await app.ainvoke(
        {"messages": [{"role": "user", "content": "Первый вопрос?"}]}, config
    )
    assert [item["doc_code"] for item in first["retrieved"]] == ["R-U2-01"]

    second = await app.ainvoke(
        {"messages": [{"role": "user", "content": "Второй вопрос?"}]}, config
    )
    assert [item["doc_code"] for item in second["retrieved"]] == ["R-U2-02"]

    third = await app.ainvoke(
        {"messages": [{"role": "user", "content": "Третий вопрос?"}]}, config
    )
    assert [item["doc_code"] for item in third["retrieved"]] == ["R-U2-03"]


@pytest.mark.asyncio
async def test_past_search_failure_does_not_poison_later_turns() -> None:
    """Разовый сбой поиска не превращает будущие ответы в «поиск недоступен».

    ``retrieval_failed`` выводится из накопленных ``failures``. Пока канал не
    очищался, одна давняя авария индекса навсегда подменяла честный ответ
    «в базе знаний информации нет» сообщением о недоступности поиска.
    """
    app = _app(_TurnSearch(fail_first=True))
    config = {"configurable": {"thread_id": "диалог-со-сбоем"}}

    failed = await app.ainvoke(
        {"messages": [{"role": "user", "content": "Вопрос во время аварии?"}]}, config
    )
    assert failed["retrieval_failed"] is True

    recovered = await app.ainvoke(
        {"messages": [{"role": "user", "content": "Вопрос после починки?"}]}, config
    )
    assert recovered["retrieval_failed"] is False
    assert recovered["retrieved"]


def test_empty_branch_does_not_erase_its_siblings() -> None:
    """Пустой список - законный вклад ветки, а не сигнал сброса.

    Поэтому сброс выражен отдельным значением: если бы сбрасывал пустой
    список, ветка, честно ничего не нашедшая, стирала бы находки соседних.
    """
    assert merge_or_reset([{"n": 1}], []) == [{"n": 1}]
    assert merge_or_reset([{"n": 1}], [{"n": 2}]) == [{"n": 1}, {"n": 2}]
    assert merge_or_reset([{"n": 1}], RESET_CHANNEL) == []
