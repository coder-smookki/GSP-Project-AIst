import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from agent.graph import build_followups, create_app_agent
from agent.state import AgentState, SearchPlan
from tests.fakes.fake_search import MockSearch

# (частично дублирует test_retrieve._SupportModel, но локален, чтобы
# тест модуля followups не зависел от коллекции фикстур роутера поиска).


def _tool_call(query: str, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="найдено",
        tool_calls=[
            {"name": "RetrieveQuery", "args": {"query": query}, "id": call_id}
        ],
    )


class _Structured:
    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if isinstance(self.plan, Exception):
            raise self.plan
        return self.plan


def _support(plan=None, answer="Ответ."):
    plan = plan if plan is not None else SearchPlan(
        mode="kb", subqueries=["Как часто проводится СОУТ?"]
    )

    class _Model:
        def __init__(self):
            self.raw = _Structured(plan)

        async def ainvoke(self, messages, **kwargs):
            if kwargs.get("tools"):
                return _tool_call(messages[0].content[:0] or "СОУТ")
            return AIMessage(content=answer)

    return _Model()


class _Model:
    def __init__(self, answer="Ответ."):
        self.answer = answer

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self.answer)


class TestBuildFollowups:
    def test_clause_items_build_detail_questions(self):
        items = [
            {"doc_code": "R-U2-23", "clause": "3.7"},
            {"doc_code": "R-U2-59", "clause": "4.1"},
        ]
        out = build_followups(items)
        assert out == [
            "Подробнее по пункту 3.7 (R-U2-23)",
            "Подробнее по пункту 4.1 (R-U2-59)",
            "Список всех найденных документов по охране труда",
        ]

    def test_doc_without_clause(self):
        items = [{"doc_code": "R-U2-23", "clause": ""}]
        out = build_followups(items)
        assert "Список положений из R-U2-23" in out

    def test_deduplicates_repeated_doc(self):
        items = [
            {"doc_code": "R-U2-23", "clause": "3.7"},
            {"doc_code": "R-U2-23", "clause": "3.8"},
            {"doc_code": "R-U2-23", "clause": "3.9"},
        ]
        out = build_followups(items)
        assert out == [
            "Подробнее по пункту 3.7 (R-U2-23)",
            "Подробнее по пункту 3.8 (R-U2-23)",
            "Подробнее по пункту 3.9 (R-U2-23)",
        ]
        assert "Список положений из R-U2-23" not in out

    def test_limit_respected(self):
        items = [{"doc_code": f"R-{i}", "clause": f"1.{i}"} for i in range(1, 6)]
        assert len(build_followups(items, limit=3)) == 3

    def test_empty_items_returns_fallback(self):
        assert build_followups([]) == ["Что искать в базе знаний?"]


class TestFollowupsInGraph:
    @pytest.mark.asyncio
    async def test_followups_emitted(self):
        app = create_app_agent(
            model=_Model(),
            support_model=_support(),
            store=InMemoryStore(),
            search=MockSearch(),
        )
        state: AgentState = {
            "messages": [{"role": "human", "content": "Как часто проводится СОУТ?"}]
        }

        kinds = []
        async for chunk in app.astream(state, stream_mode=["custom"]):
            if not isinstance(chunk, tuple):
                continue
            env = chunk[1]["aist"]
            kinds.append(env["kind"])

        assert "followups" in kinds
        assert "sources" in kinds

    @pytest.mark.asyncio
    async def test_followups_land_in_state(self):
        app = create_app_agent(
            model=_Model(),
            support_model=_support(),
            store=InMemoryStore(),
            search=MockSearch(),
        )
        state: AgentState = {
            "messages": [{"role": "human", "content": "Как часто проводится СОУТ?"}]
        }
        result = await app.ainvoke(state)

        followups = result.get("followups")
        assert followups
        assert isinstance(followups, list)
        assert all(isinstance(f, str) for f in followups)
