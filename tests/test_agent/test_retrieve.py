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
        assert result.get("retrieved") == []

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


class TestTodayInjection:
    """Дата в рантайме: без неё вопросы о сроках отвечаются вслепую."""

    def test_today_is_deterministic_when_supplied(self):
        """Переданная снаружи дата побеждает системные часы.

        Без этого тест на вопросы вида «в какие сроки» невоспроизводим:
        ответ менялся бы каждый день.
        """
        from agent.graph import today_msk

        config = {"configurable": {"today": "01.02.2026"}}
        assert today_msk(config) == "01.02.2026"

    def test_today_falls_back_to_moscow_clock(self):
        """Без явной даты берётся московское время."""
        from datetime import datetime

        from agent.graph import MSK, today_msk

        assert today_msk(None) == datetime.now(MSK).strftime("%d.%m.%Y")
        assert today_msk({}) == datetime.now(MSK).strftime("%d.%m.%Y")

    def test_prompt_has_today_placeholder(self):
        """Промпт действительно получает дату, а не игнорирует её."""
        from agent.graph import MODEL_SYSTEM_MESSAGE

        assert "{today}" in MODEL_SYSTEM_MESSAGE
        rendered = MODEL_SYSTEM_MESSAGE.format(
            memory="", context="", today="09.08.2026"
        )
        assert "09.08.2026" in rendered


class _FailingModel:
    """Модель, всегда падающая заданной ошибкой."""

    def __init__(self, error: BaseException):
        self._error = error

    async def ainvoke(self, messages=None, **kwargs):
        """Всегда бросает заготовленную ошибку.

        Args:
            messages: Сообщения запроса.
            **kwargs: Прочие параметры вызова.

        Raises:
            BaseException: Заготовленная ошибка.
        """
        raise self._error


class TestModelFailure:
    """Недоступность модели не должна ронять запрос.

    Раньше граф ловил только `SearchUnavailable`, поэтому таймаут генерации
    или выгруженная из памяти модель приводили к трассировке вместо ответа,
    а диалог оставался без реплики ассистента.
    """

    @pytest.mark.asyncio
    async def test_timeout_yields_service_message(self):
        """Таймаут модели превращается в понятное сообщение об аварии."""
        import httpx

        app = create_app_agent(
            model=_FailingModel(httpx.ReadTimeout("время вышло")),
            support_model=FakeModel(),
            store=InMemoryStore(),
            search=MockSearch(),
        )
        state: AgentState = {
            "messages": [{"role": "human", "content": "Как часто проводится СОУТ?"}]
        }

        result = await app.ainvoke(state)
        answer = result["messages"][-1].content

        assert "недоступен" in answer
        # Ответ не должен маскироваться под честный отказ: сведения в базе
        # есть, недоступна именно генерация.
        assert "В базе знаний информации нет" not in answer

    @pytest.mark.asyncio
    async def test_connect_error_yields_service_message(self):
        """Неподнятая Ollama тоже не роняет запрос."""
        import httpx

        app = create_app_agent(
            model=_FailingModel(httpx.ConnectError("соединение отклонено")),
            support_model=FakeModel(),
            store=InMemoryStore(),
            search=MockSearch(),
        )
        state: AgentState = {
            "messages": [{"role": "human", "content": "Как часто проводится СОУТ?"}]
        }

        result = await app.ainvoke(state)
        assert "недоступен" in result["messages"][-1].content


class _FirstQueryEmptySearch(MockSearch):
    """Первый поиск пуст, повторный (после переписывания) находит."""

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    async def query(self, query, sections=None, limit=None, areas=None):
        self.calls.append(query)
        if len(self.calls) == 1:
            return []
        return await super().query(query, sections=sections, limit=limit, areas=areas)


class TestCorrectiveRAGRewrite:
    """Пустой первый поиск уходит в отдельный узел переписывания запроса."""

    @pytest.mark.asyncio
    async def test_loop_runs_search_twice_and_returns_results(self):
        search = _FirstQueryEmptySearch()
        app = create_app_agent(
            model=FakeModel(),
            support_model=FakeModel(answer=""),
            store=InMemoryStore(),
            search=search,
        )
        state: AgentState = {
            "messages": [{"role": "human", "content": "Как часто проводится СОУТ?"}]
        }

        result = await app.ainvoke(state)

        assert len(search.calls) == 2
        assert search.calls[0] == "Как часто проводится СОУТ?"
        assert result.get("retrieved"), "повторный поиск должен дать источники"
        assert not result.get("retrieval_failed")

    @pytest.mark.asyncio
    async def test_loop_stops_after_one_rewrite(self):
        seen: list[str] = []

        class NeverHits(MockSearch):
            async def query(self, *args, **kwargs):
                seen.append("query")
                return []

        app = create_app_agent(
            model=FakeModel(),
            support_model=FakeModel(answer="снова ничего"),
            store=InMemoryStore(),
            search=NeverHits(),
        )
        state: AgentState = {"messages": [{"role": "human", "content": "СОУТ"}]}

        result = await app.ainvoke(state)

        # 1 исходный запрос + 1 после переписывания, без бесконечного цикла
        assert len(seen) <= 2
        assert not result.get("retrieval_failed")

    @pytest.mark.asyncio
    async def test_max_rewrites_zero_disables_correction(self):
        seen: list[str] = []

        class NeverHits(MockSearch):
            async def query(self, *args, **kwargs):
                seen.append("query")
                return []

        app = create_app_agent(
            model=FakeModel(),
            support_model=FakeModel(answer="снова ничего"),
            store=InMemoryStore(),
            search=NeverHits(),
            max_rewrites=0,
        )
        state: AgentState = {"messages": [{"role": "human", "content": "СОУТ"}]}

        await app.ainvoke(state)

        assert len(seen) == 1, "без коррекции узел rewrite_query не должен запускаться"

    @pytest.mark.asyncio
    async def test_max_rewrites_two_allows_additional_retry(self):
        seen: list[str] = []

        class NeverHit(MockSearch):
            async def query(self, *args, **kwargs):
                seen.append("query")
                return []

        app = create_app_agent(
            model=FakeModel(),
            support_model=FakeModel(answer="снова ничего"),
            store=InMemoryStore(),
            search=NeverHit(),
            max_rewrites=2,
        )
        state: AgentState = {"messages": [{"role": "human", "content": "СОУТ"}]}

        await app.ainvoke(state)

        # 1 исходный поиск + 2 переписывания = ровно 3 обращения к поиску
        assert len(seen) == 3
