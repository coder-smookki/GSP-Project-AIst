"""Тесты веточного поискового агента.

Проверяется поведение ReAct-петли: когда модель вызывает инструмент,
когда останавливается, как работает лимит раундов и что происходит при
недоступности поиска. Слабые модели склонны вызывать инструмент без
остановки или вовсе не вызывать, поэтому границы тестируются явно.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agent.search_agent import RetrieveQuery, build_search_agent
from infrastructure.retrieval.qdrant_search import SearchUnavailable
from infrastructure.retrieval.search import Chunk
from tests.fakes.fake_search import MockSearch

CHUNK = Chunk(
    chunk_id="R-U2-23#c001",
    doc_code="R-U2-23",
    doc_title="Проведение специальной оценки условий труда",
    section="3.7",
    section_type="procedura",
    clause="3.7",
    text="Специальная оценка условий труда проводится не реже одного раза в пять лет.",
)


def _tool_call(query: str, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RetrieveQuery",
                "args": {"query": query},
                "id": call_id,
            }
        ],
    )


class _ScriptedModel:
    """Модель, отыгрывающая заранее заданную последовательность ответов."""

    def __init__(self, *script: AIMessage):
        self._script = list(script)
        self.calls = 0
        self.seen_messages = []
        self.seen_kwargs = []

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.seen_messages.append(messages)
        self.seen_kwargs.append(kwargs)
        if self._script:
            return self._script.pop(0)
        return AIMessage(content="Готово.")


class _RecordingSearch(MockSearch):
    """Поиск, записывающий запросы; может падать или отдавать заданную выдачу."""

    def __init__(self, hits: list[tuple[Chunk, float]] | None = None, error=False):
        self.queries: list[str] = []
        self.sections = None
        self._hits = hits
        self._error = error

    async def query(self, query, sections=None, limit=None, areas=None, docs=None):
        self.queries.append(query)
        self.sections = sections
        self.docs = docs
        if self._error:
            raise SearchUnavailable("поиск лежит")
        if self._hits is not None:
            return self._hits
        return await super().query(
            query, sections=sections, limit=limit, areas=areas, docs=docs
        )


class TestSearchAgent:
    def _agent(self, model, search, max_rounds: int = 2):
        return build_search_agent(model, search, max_rounds=max_rounds)

    @pytest.mark.asyncio
    async def test_model_without_tool_call_skips_search(self):
        model = _ScriptedModel(AIMessage(content="Ничего не нужно."))
        search = _RecordingSearch()
        app = self._agent(model, search)
        out = await app.ainvoke({"query": "СОУТ", "rounds": 0})
        assert search.queries == []
        assert out.get("retrieved") == []
        assert out.get("failures") == []

    @pytest.mark.asyncio
    async def test_single_tool_call_returns_citations(self):
        model = _ScriptedModel(_tool_call("частота СОУТ"))
        search = _RecordingSearch(hits=[(CHUNK, 0.9)])
        app = self._agent(model, search)
        out = await app.ainvoke({"query": "СОУТ", "rounds": 0})

        assert search.queries == ["частота СОУТ"]
        retrieved = out.get("retrieved")
        assert retrieved
        assert retrieved[0]["doc_code"] == "R-U2-23"
        assert retrieved[0]["chunk_id"] == "R-U2-23#c001"

    @pytest.mark.asyncio
    async def test_tool_gets_query_from_args(self):
        model = _ScriptedModel(_tool_call("периодичность СОУТ"))
        search = _RecordingSearch()
        app = self._agent(model, search)
        await app.ainvoke({"query": "исходный", "rounds": 0})
        assert search.queries == ["периодичность СОУТ"]

    @pytest.mark.asyncio
    async def test_empty_args_falls_back_to_subquery(self):
        model = _ScriptedModel(_tool_call("  "))
        search = _RecordingSearch()
        app = self._agent(model, search)
        await app.ainvoke({"query": "подзапрос", "rounds": 0})
        assert search.queries == ["подзапрос"]

    @pytest.mark.asyncio
    async def test_retry_after_empty_results(self):
        model = _ScriptedModel(
            _tool_call("СОУТ", "call-1"),
            _tool_call("специальная оценка условий труда", "call-2"),
        )
        search = _RecordingSearch(hits=[(CHUNK, 0.9)])
        app = self._agent(model, search)
        out = await app.ainvoke({"query": "СОУТ", "rounds": 0})

        assert search.queries == ["СОУТ", "специальная оценка условий труда"]
        assert out.get("retrieved")

    @pytest.mark.asyncio
    async def test_max_rounds_limits_search_calls(self):
        model = _ScriptedModel(
            _tool_call("а", "call-1"),
            _tool_call("б", "call-2"),
            _tool_call("в", "call-3"),
        )
        search = _RecordingSearch()
        app = self._agent(model, search, max_rounds=2)
        await app.ainvoke({"query": "q", "rounds": 0})

        assert search.queries == ["а", "б"]
        assert model.calls == 2

    @pytest.mark.asyncio
    async def test_retry_fails_when_search_unavailable(self):
        model = _ScriptedModel(_tool_call("СОУТ"))
        search = _RecordingSearch(error=True)
        app = self._agent(model, search)
        out = await app.ainvoke({"query": "СОУТ", "rounds": 0})

        assert out.get("failures") == ["СОУТ"]
        assert out.get("retrieval_failed") is True

    @pytest.mark.asyncio
    async def test_sections_reach_search(self):
        model = _ScriptedModel(_tool_call("СОУТ"))
        search = _RecordingSearch()
        app = self._agent(model, search)
        await app.ainvoke({"query": "СОУТ", "rounds": 0, "sections": ["procedura"]})
        assert search.sections == ["procedura"]

    @pytest.mark.asyncio
    async def test_selected_documents_reach_search(self):
        """Выбранные документы доходят до поиска через состояние ветки.

        Третий уровень дерева областей знаний бесполезен, если отбор теряется
        по дороге к веточному агенту: пользователь ограничил бы выдачу одним
        регламентом, а поиск всё равно шёл бы по всей базе.
        """
        model = _ScriptedModel(_tool_call("СОУТ"))
        search = _RecordingSearch()
        app = self._agent(model, search)
        await app.ainvoke(
            {"query": "СОУТ", "rounds": 0, "docs": ["Р-У2-59", "Р-У2-23"]}
        )
        assert search.docs == ["Р-У2-59", "Р-У2-23"]

    @pytest.mark.asyncio
    async def test_prompt_carries_subquery_and_rounds(self):
        model = _ScriptedModel(AIMessage(content="Готово."))
        search = _RecordingSearch()
        app = self._agent(model, search, max_rounds=2)
        await app.ainvoke({"query": "мой подзапрос", "rounds": 0})

        system = model.seen_messages[0][0]
        assert "мой подзапрос" in system.content
        assert "2" in system.content

    @pytest.mark.asyncio
    async def test_tool_schema_bound_to_model(self):
        model = _ScriptedModel(_tool_call("СОУТ"))
        search = _RecordingSearch()
        app = self._agent(model, search)
        await app.ainvoke({"query": "СОУТ", "rounds": 0})
        assert model.seen_kwargs[0]["tools"] == [RetrieveQuery]

    @pytest.mark.asyncio
    async def test_model_sees_tool_results_before_retry(self):
        model = _ScriptedModel(
            _tool_call("СОУТ", "call-1"),
            _tool_call("специальная оценка", "call-2"),
        )
        search = _RecordingSearch()
        app = self._agent(model, search)
        await app.ainvoke({"query": "СОУТ", "rounds": 0})

        second_call = model.seen_messages[1]
        tool_message = second_call[-1]
        assert tool_message.type == "tool"
        assert tool_message.tool_call_id == "call-1"