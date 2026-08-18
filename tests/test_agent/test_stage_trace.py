"""События этапов графа: узлы сообщают о начале и о завершении работы.

Интерфейс рисует ход выполнения по событиям ``trace`` (см. docs/GUI.md).
Событие о завершении приходит, когда узел уже отработал, поэтому одного его
мало: без парного «start» пользователь всё так же ждал бы молча. Тесты держат
этот контракт: у долгих узлов есть оба события, а у веток поиска они ещё и
разделены подзапросом.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from agent.graph import create_app_agent
from agent.state import SearchPlan
from domain.context import ContextManager
from infrastructure.retrieval.mock_search import MockSearch


class _Structured:
    """Обёртка ``.raw`` с заранее заданным планом поиска."""

    def __init__(self, plan: SearchPlan) -> None:
        """Запоминает план, который вернёт планировщик.

        Args:
            plan: План поиска, отдаваемый структурированным выводом.
        """
        self._plan = plan

    def with_structured_output(self, schema, **kwargs):
        """Возвращает себя же вместо связанной модели.

        Args:
            schema: Схема структурированного вывода.
            **kwargs: Прочие параметры вызова.

        Returns:
            _Structured: Тот же объект.
        """
        return self

    async def ainvoke(self, messages, **kwargs) -> SearchPlan:
        """Отдаёт заранее заданный план.

        Args:
            messages: Сообщения запроса.
            **kwargs: Прочие параметры вызова.

        Returns:
            SearchPlan: План поиска.
        """
        return self._plan


class _SupportModel:
    """Модель планировщика, веток поиска, свёртки истории и памяти."""

    def __init__(self, plan: SearchPlan) -> None:
        """Собирает модель-заглушку.

        Args:
            plan: План, который вернёт планировщик.
        """
        self.raw = _Structured(plan)

    async def ainvoke(self, messages=None, **kwargs) -> AIMessage:
        """Отвечает без вызова инструментов: ветка сразу завершается.

        Args:
            messages: Сообщения запроса.
            **kwargs: Прочие параметры вызова.

        Returns:
            AIMessage: Ответ без tool_calls.
        """
        return AIMessage(content="Готово.")


class _AnswerModel:
    """Модель финального ответа."""

    async def ainvoke(self, messages=None, **kwargs) -> AIMessage:
        """Возвращает фиксированный ответ.

        Args:
            messages: Сообщения запроса.
            **kwargs: Прочие параметры вызова.

        Returns:
            AIMessage: Ответ модели.
        """
        return AIMessage(content="Ответ по документам.")


def _app(plan: SearchPlan, context_manager=None):
    """Собирает граф с заглушками моделей и поиска.

    Args:
        plan: План поиска, который вернёт планировщик.
        context_manager: Менеджер свёртки истории (опционально).

    Returns:
        CompiledStateGraph: Скомпилированный граф.
    """
    return create_app_agent(
        model=_AnswerModel(),
        support_model=_SupportModel(plan),
        context_manager=context_manager,
        store=InMemoryStore(),
        search=MockSearch(),
    )


async def _traces(app, state) -> list[dict]:
    """Собирает полезные нагрузки событий trace из потока графа.

    Args:
        app: Скомпилированный граф.
        state: Входное состояние запуска.

    Returns:
        list[dict]: Полезные нагрузки событий kind="trace" по порядку.
    """
    out: list[dict] = []
    async for _mode, data in app.astream(state, stream_mode=["custom"]):
        env = (data or {}).get("aist") or {}
        if env.get("kind") == "trace":
            out.append(env.get("payload") or {})
    return out


def _pairs(traces: list[dict]) -> list[tuple[str, str]]:
    """Сводит события к парам «этап, статус».

    Args:
        traces: Полезные нагрузки событий trace.

    Returns:
        list[tuple[str, str]]: Пары этапа и статуса по порядку.
    """
    return [(t.get("stage", ""), t.get("status", "")) for t in traces]


class TestStageTrace:
    @pytest.mark.asyncio
    async def test_plan_reports_start_and_done(self):
        app = _app(SearchPlan(mode="kb", subqueries=["СОУТ"]))
        traces = await _traces(
            app,
            {"messages": [{"role": "user", "content": "Как часто проводится СОУТ?"}]},
        )
        pairs = _pairs(traces)

        assert pairs.index(("plan", "start")) < pairs.index(("plan", "done"))

    @pytest.mark.asyncio
    async def test_each_branch_reports_its_subquery(self):
        app = _app(SearchPlan(mode="kb", subqueries=["наряд-допуск", "СИЗ"]))
        traces = await _traces(
            app, {"messages": [{"role": "user", "content": "Наряд-допуск и СИЗ"}]}
        )

        started = [
            t.get("query")
            for t in traces
            if t.get("stage") == "search_agent" and t.get("status") == "start"
        ]
        finished = [
            t.get("query")
            for t in traces
            if t.get("stage") == "search_agent" and t.get("status") == "done"
        ]

        assert sorted(started) == ["СИЗ", "наряд-допуск"]
        assert sorted(finished) == ["СИЗ", "наряд-допуск"]

    @pytest.mark.asyncio
    async def test_finalize_and_assistant_reported(self):
        app = _app(SearchPlan(mode="kb", subqueries=["СОУТ"]))
        traces = await _traces(
            app,
            {"messages": [{"role": "user", "content": "Как часто проводится СОУТ?"}]},
        )
        pairs = _pairs(traces)

        assert ("finalize", "done") in pairs
        assert ("assistant", "start") in pairs
        assert ("assistant", "done") in pairs

    @pytest.mark.asyncio
    async def test_chat_mode_reports_plan_and_assistant_only(self):
        app = _app(SearchPlan(mode="chat"))
        traces = await _traces(
            app, {"messages": [{"role": "user", "content": "Привет!"}]}
        )
        stages = {t.get("stage") for t in traces}

        assert "search_agent" not in stages
        assert "finalize" not in stages
        assert {"plan", "assistant"} <= stages

    @pytest.mark.asyncio
    async def test_no_condense_stage_without_context_manager(self):
        app = _app(SearchPlan(mode="chat"))
        traces = await _traces(
            app, {"messages": [{"role": "user", "content": "Привет!"}]}
        )

        assert all(t.get("stage") != "condense" for t in traces)

    @pytest.mark.asyncio
    async def test_condense_stays_silent_when_history_fits(self):
        manager = ContextManager(
            llm=_SupportModel(SearchPlan(mode="chat")), budget=10**6
        )
        app = _app(SearchPlan(mode="chat"), context_manager=manager)
        traces = await _traces(
            app, {"messages": [{"role": "user", "content": "Привет!"}]}
        )

        assert all(t.get("stage") != "condense" for t in traces)

    @pytest.mark.asyncio
    async def test_condense_reports_start_and_done(self):
        manager = ContextManager(
            llm=_SupportModel(SearchPlan(mode="chat")), keep_last=1, budget=0
        )
        app = _app(SearchPlan(mode="chat"), context_manager=manager)
        traces = await _traces(
            app,
            {
                "messages": [
                    {"role": "user", "content": "Первый вопрос про охрану труда"},
                    {"role": "user", "content": "Второй вопрос про охрану труда"},
                ]
            },
        )
        pairs = _pairs(traces)

        assert ("condense", "start") in pairs
        assert ("condense", "done") in pairs


class TestShouldCondense:
    def test_history_within_budget_is_not_condensed(self):
        manager = ContextManager(llm=None, budget=10**6)
        assert manager.should_condense([AIMessage(content="привет")]) is False

    def test_short_history_is_not_condensed_even_over_budget(self):
        manager = ContextManager(llm=None, keep_last=5, budget=0)
        assert manager.should_condense([AIMessage(content="привет")]) is False

    def test_long_history_over_budget_is_condensed(self):
        manager = ContextManager(llm=None, keep_last=1, budget=0)
        messages = [AIMessage(content="привет"), AIMessage(content="и снова")]
        assert manager.should_condense(messages) is True

    def test_existing_summary_counts_towards_budget(self):
        manager = ContextManager(llm=None, keep_last=1, budget=10)
        messages = [AIMessage(content="а"), AIMessage(content="б")]
        assert manager.should_condense(messages) is False
        assert manager.should_condense(messages, existing_summary="я" * 100) is True
