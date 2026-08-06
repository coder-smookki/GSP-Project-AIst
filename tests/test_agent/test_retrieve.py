import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from agent.graph import create_app_agent
from agent.state import AgentState
from tests.fakes.fake_search import MockSearch


class FakeModel:
    def __init__(self, answer: str = "Ответ по документам."):
        self.answer = answer

    async def ainvoke(self, messages):
        return AIMessage(content=self.answer)


class TestRetrieveNode:
    @pytest.mark.asyncio
    async def test_sources_event_emitted_before_answer(self):
        app = create_app_agent(
            model=FakeModel(),
            support_model=FakeModel(),
            store=InMemoryStore(),
            search=MockSearch(),
        )
        state: AgentState = {
            "messages": [{"role": "human", "content": "Как часто проводится СОУТ?"}]
        }

        custom: list[tuple[str, dict]] = []
        async for chunk in app.astream(state, stream_mode=["custom"]):
            if isinstance(chunk, tuple):
                custom.append(chunk)
            else:
                custom.append(("custom", chunk))

        assert any(item["aist"]["kind"] == "sources" for _, item in custom)

    @pytest.mark.asyncio
    async def test_retrieved_lands_in_state(self):
        app = create_app_agent(
            model=FakeModel(),
            support_model=FakeModel(),
            store=InMemoryStore(),
            search=MockSearch(),
        )

        state: AgentState = {
            "messages": [{"role": "human", "content": "Как часто проводится СОУТ?"}]
        }
        result = await app.ainvoke(state)

        retrieved = result.get("retrieved")
        assert retrieved
        first = retrieved[0]
        assert first["doc_code"] == "R-U2-23"
        assert set(
            {
                "n",
                "doc_code",
                "doc_title",
                "clause",
                "section_no",
                "section_title",
                "chunk_id",
                "score",
                "snippet",
                "url",
            }
        ) <= set(first)

    @pytest.mark.asyncio
    async def test_no_search_no_crash(self):
        app = create_app_agent(
            model=FakeModel(),
            support_model=FakeModel(),
            store=InMemoryStore(),
        )
        state: AgentState = {"messages": [{"role": "human", "content": "Привет"}]}
        result = await app.ainvoke(state)
        assert result["messages"][-1].content == "Ответ по документам."
        assert result.get("retrieved") is None

    @pytest.mark.asyncio
    async def test_sections_filter_reaches_search(self):
        app = create_app_agent(
            model=FakeModel(),
            support_model=FakeModel(),
            store=InMemoryStore(),
            search=MockSearch(),
        )
        state: AgentState = {
            "messages": [{"role": "human", "content": "Что такое СОУТ?"}],
            "sections": ["sokrashcheniya"],
        }
        result = await app.ainvoke(state)

        retrieved = result.get("retrieved")
        assert retrieved
        assert all(r["section_title"] == "2.2" for r in retrieved)
