"""Тесты графа агента: планирование → ветки → сборка → ответ.

Покрываются критические для жюри сценарии: приветствие (без поиска), одиночный
поиск, несколько подзапросов (ветки параллельно), дедуп/сортировка/нумерация
финализатора, недоступность поиска и падение модели. Стихийные модели —
заглушки с ``.raw.with_structured_output`` для планировщика и tool-calling
для веток.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from agent.graph import create_app_agent
from agent.state import AgentState, SearchPlan
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


class _Structured:
    """Обёртка ``.raw`` с программируемым структурированным выводом."""

    def __init__(self, plans):
        self._plans = list(plans)
        self.calls = 0
        self.schema = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        plan = self._plans.pop(0) if self._plans else None
        if isinstance(plan, Exception):
            raise plan
        return plan


class _SupportModel:
    """support_model: планировщик (через ``.raw``) и ветки (через ainvoke)."""

    def __init__(self, plans=None, turns=None):
        self.raw = _Structured(plans if plans else [])
        self._turns = list(turns if turns else [])
        self.branch_calls = 0
        self.branch_messages = []

    async def ainvoke(self, messages, **kwargs):
        self.branch_calls += 1
        self.branch_messages.append(messages)
        tools = kwargs.get("tools")
        if tools:
            if self._has_hit(messages):
                return AIMessage(content="Готово.")
            sys = messages[0].content
            m = re.search(r"Подзапрос пользователя: «(.+?)»", sys)
            q = m.group(1) if m else "СОУТ"
            return _tool_call("найдено", f"t{self.branch_calls}", query=q)
        if self._turns:
            return self._turns.pop(0)
        return AIMessage(content="Ответ по документам.")

    @staticmethod
    def _has_hit(messages) -> bool:
        """Есть ли в диалоге ToolMessage с непустой выдачей (без подсказки к retry)."""
        for msg in messages:
            txt = getattr(msg, "content", "") or ""
            if isinstance(msg, ToolMessage) and txt and "ничего не найдено" not in txt:
                return True
        return False


class FakeModel:
    """Финальная модель ответа (без tools)."""

    def __init__(self, answer: str = "Ответ по документам."):
        self.answer = answer

    async def ainvoke(self, messages=None, **kwargs):
        return AIMessage(content=self.answer)


def _tool_call(content: str, call_id: str, query: str = "СОУТ") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": "RetrieveQuery",
                "args": {"query": query},
                "id": call_id,
            }
        ],
    )


class _RecordingSearch(MockSearch):
    """Поиск, запоминающий запросы и выдачу по запросу."""

    def __init__(self, corpus=None, error=False):
        self.queries: list[str] = []
        self._corpus = corpus
        self._error = error

    async def query(self, query, sections=None, limit=None, areas=None):
        self.queries.append(query)
        if self._error:
            raise SearchUnavailable("недоступен")
        if self._corpus is not None:
            return self._corpus
        return await MockSearch().query(
            query, sections=sections, limit=limit, areas=areas
        )


def _app(**overrides):
    """Собирает агента с фиктивными моделями и поиском."""
    opts = {
        "model": FakeModel(),
        "support_model": _SupportModel(),
        "store": InMemoryStore(),
        "search": MockSearch(),
        "max_requests": 10,
        "branch_retries": 1,
        "top_k": 5,
    }
    opts.update(overrides)
    return create_app_agent(**opts)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_chat_mode_skips_search(self):
        support = _SupportModel(plans=[SearchPlan(mode="chat")])
        app = _app(support_model=support, search=_RecordingSearch())
        state: AgentState = {
            "messages": [{"role": "user", "content": "Привет, как дела?"}]
        }
        result = await app.ainvoke(state)

        assert result["messages"][-1].content == "Ответ по документам."
        assert result.get("retrieved") == []
        assert support.branch_calls == 0

    @pytest.mark.asyncio
    async def test_single_kb_query_returns_results(self):
        support = _SupportModel(
            plans=[SearchPlan(mode="kb", subqueries=["СОУТ"])],
            turns=[],
        )
        search = _RecordingSearch()
        app = _app(support_model=support, search=search)
        state: AgentState = {
            "messages": [{"role": "user", "content": "Как часто проводится СОУТ?"}]
        }
        result = await app.ainvoke(state)

        assert search.queries == ["СОУТ"]
        retrieved = result.get("retrieved")
        assert retrieved
        assert retrieved[0]["doc_code"] == "R-U2-23"

    @pytest.mark.asyncio
    async def test_multiple_subqueries_search_in_parallel(self):
        support = _SupportModel(
            plans=[
                SearchPlan(
                    mode="kb",
                    subqueries=["нормы СОУТ", "нормы ГОСТ 4.512"],
                )
            ],
            turns=[],
        )
        search = _RecordingSearch()
        app = _app(support_model=support, search=search)
        state: AgentState = {
            "messages": [
                {"role": "user", "content": "Нормы и как носить каски"}
            ]
        }
        result = await app.ainvoke(state)

        assert "нормы СОУТ" in search.queries
        assert "нормы ГОСТ 4.512" in search.queries
        assert result.get("retrieved")

    @pytest.mark.asyncio
    async def test_sources_event_emitted_before_answer(self):
        support = _SupportModel(
            plans=[SearchPlan(mode="kb", subqueries=["СОУТ"])],
        )
        search = _RecordingSearch()
        app = _app(support_model=support, search=search)
        state: AgentState = {
            "messages": [{"role": "user", "content": "Как часто проводится СОУТ?"}]
        }

        kinds = []
        async for chunk in app.astream(state, stream_mode=["custom"]):
            if isinstance(chunk, tuple):
                kinds.append(chunk[1]["aist"]["kind"])

        assert "sources" in kinds
        assert "trace" in kinds

    @pytest.mark.asyncio
    async def test_followups_emitted(self):
        support = _SupportModel(
            plans=[SearchPlan(mode="kb", subqueries=["СОУТ"])],
        )
        app = _app(support_model=support, search=_RecordingSearch())
        state: AgentState = {
            "messages": [{"role": "user", "content": "Как часто проводится СОУТ?"}]
        }
        kinds = []
        async for chunk in app.astream(state, stream_mode=["custom"]):
            if isinstance(chunk, tuple):
                kinds.append(chunk[1]["aist"]["kind"])
        assert "followups" in kinds
        assert "sources" in kinds


class TestFinalize:
    @pytest.mark.asyncio
    async def test_retrieved_renumbered_after_merge(self):
        support = _SupportModel(
            plans=[SearchPlan(mode="kb", subqueries=["а", "б"])],
        )
        search = _RecordingSearch()
        app = _app(support_model=support, search=search)
        state: AgentState = {
            "messages": [{"role": "user", "content": "вопрос"}]
        }
        result = await app.ainvoke(state)

        numbers = [item["n"] for item in result["retrieved"]]
        assert numbers == list(range(1, len(numbers) + 1))

    @pytest.mark.asyncio
    async def test_retrieved_capped_by_top_k(self):
        support = _SupportModel(
            plans=[SearchPlan(mode="kb", subqueries=["СОУТ"])],
        )
        app = _app(support_model=support, search=_RecordingSearch(), top_k=3)
        state: AgentState = {
            "messages": [{"role": "user", "content": "вопрос"}]
        }
        result = await app.ainvoke(state)

        assert len(result["retrieved"]) <= 3

    @pytest.mark.asyncio
    async def test_duplicates_merged(self):
        support = _SupportModel(
            plans=[SearchPlan(mode="kb", subqueries=["СОУТ", "СОУТ"])],
        )
        app = _app(support_model=support, search=_RecordingSearch())
        state: AgentState = {
            "messages": [{"role": "user", "content": "вопрос"}]
        }
        result = await app.ainvoke(state)

        ids = [c["chunk_id"] for c in result["retrieved"]]
        assert len(ids) == len(set(ids)), "дубли по chunk_id должны срастаться"


class TestPlanFailure:
    @pytest.mark.asyncio
    async def test_model_crash_falls_back_to_search_by_question(self):
        support = _SupportModel(plans=[RuntimeError("модель упала")])
        search = _RecordingSearch()
        app = _app(support_model=support, search=search)
        state: AgentState = {
            "messages": [{"role": "user", "content": "Как часто проводится СОУТ?"}]
        }
        await app.ainvoke(state)

        assert search.queries == ["Как часто проводится СОУТ?"]


class TestModelFailure:
    @pytest.mark.asyncio
    async def test_assistant_model_timeout_yields_service_message(self):
        import httpx

        app = _app(
            model=_FailingModel(httpx.ReadTimeout("время вышло")),
            support_model=_SupportModel(plans=[SearchPlan(mode="chat")]),
        )
        state: AgentState = {
            "messages": [{"role": "user", "content": "Привет"}]
        }
        result = await app.ainvoke(state)
        answer = result["messages"][-1].content

        assert "недоступен" in answer
        assert "В базе знаний информации нет" not in answer

    @pytest.mark.asyncio
    async def test_search_unavailable_yields_honest_answer(self):
        support = _SupportModel(
            plans=[SearchPlan(mode="kb", subqueries=["СОУТ"])],
        )
        search = _RecordingSearch(error=True)
        app = _app(support_model=support, search=search, model=FakeModel("Ответ."))
        state: AgentState = {
            "messages": [{"role": "user", "content": "Как часто проводится СОУТ?"}]
        }
        result = await app.ainvoke(state)

        assert result.get("retrieval_failed") is True
        assert result.get("retrieved") == []

    @pytest.mark.asyncio
    async def test_search_unavailable_context_instructs_model(self):
        """Контекст запрещает отвечать по памяти, а выдача пуста."""
        support = _SupportModel(
            plans=[SearchPlan(mode="kb", subqueries=["СОУТ"])],
        )
        search = _RecordingSearch(error=True)

        seen_contexts = []

        class _SpyModel:
            async def ainvoke(self, messages=None, **kwargs):
                system = messages[0]
                seen_contexts.append(system.content)
                return AIMessage(content="Ответ по документам.")

        app = _app(support_model=support, search=search, model=_SpyModel())
        state: AgentState = {
            "messages": [{"role": "user", "content": "Как часто проводится СОУТ?"}]
        }
        await app.ainvoke(state)

        assert any(
            "ПОИСК ПО БАЗЕ ЗНАНИЙ ВРЕМЕННО НЕДОСТУПЕН" in c
            for c in seen_contexts
        )


class TestQueryRouting:
    @pytest.mark.asyncio
    async def test_always_answer_with_db_forces_kb_on_greeting(self):
        support = _SupportModel(plans=[SearchPlan(mode="chat")])
        search = _RecordingSearch()
        app = _app(support_model=support, search=search)
        state: AgentState = {
            "messages": [{"role": "user", "content": "Привет"}],
            "always_answer_with_db": True,
        }
        result = await app.ainvoke(state)

        assert result.get("retrieved") == []
        assert len(search.queries) >= 1
        assert search.queries[0] == "Привет"


class TestTodayInjection:
    def test_today_is_deterministic_when_supplied(self):
        from agent.graph import today_msk

        config = {"configurable": {"today": "01.02.2026"}}
        assert today_msk(config) == "01.02.2026"

    def test_today_falls_back_to_moscow_clock(self):
        from datetime import datetime

        from agent.graph import MSK, today_msk

        assert today_msk(None) == datetime.now(MSK).strftime("%d.%m.%Y")
        assert today_msk({}) == datetime.now(MSK).strftime("%d.%m.%Y")

    def test_prompt_has_today_placeholder(self):
        from agent.graph import MODEL_SYSTEM_MESSAGE

        assert "{today}" in MODEL_SYSTEM_MESSAGE
        rendered = MODEL_SYSTEM_MESSAGE.format(
            memory="", context="", today="09.08.2026"
        )
        assert "09.08.2026" in rendered


class _FailingModel:
    def __init__(self, error: BaseException):
        self._error = error

    async def ainvoke(self, messages=None, **kwargs):
        raise self._error
