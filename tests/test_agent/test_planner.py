"""Тесты планировщика поиска.

Планировщик — первая точка, где слабая модель может испортить поиск:
неверный mode="chat" оставит пользователя без ответа, а раздутые subqueries
съедят бюджет. Поэтому тестируются прежде всего фолбэки и ограничения.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agent.planner import max_subqueries, plan_search
from agent.prompts import PLAN_PROMPT
from agent.state import SearchPlan


class _Structured:
    """Модель со структурированным выводом, отдающая заданный результат."""

    def __init__(self, result):
        self._result = result
        self.calls = 0
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.messages = messages
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Model:
    """Клиент модели с доступом к `raw`."""

    def __init__(self, result):
        self.raw = _Structured(result)


class TestMaxSubqueries:
    def test_divides_budget_by_branch_calls(self):
        assert max_subqueries(10, 1) == 5

    def test_no_retries_means_full_budget(self):
        assert max_subqueries(10, 0) == 10

    def test_budget_smaller_than_retry_cost(self):
        assert max_subqueries(1, 1) == 1

    def test_budget_not_divisible(self):
        assert max_subqueries(7, 2) == 2


class TestPlanSearch:
    @pytest.mark.asyncio
    async def test_kb_plan_passes_through(self):
        model = _Model(SearchPlan(mode="kb", subqueries=["СОУТ", "ГОСТ 4.512"]))
        plan = await plan_search(model, "Что такое СОУТ?")
        assert plan == SearchPlan(mode="kb", subqueries=["СОУТ", "ГОСТ 4.512"])

    @pytest.mark.asyncio
    async def test_chat_plan_passes_through(self):
        model = _Model(SearchPlan(mode="chat"))
        plan = await plan_search(model, "Привет!")
        assert plan == SearchPlan(mode="chat")

    @pytest.mark.asyncio
    async def test_schema_is_search_plan(self):
        model = _Model(SearchPlan(mode="kb"))
        await plan_search(model, "вопрос")
        assert model.raw.schema is SearchPlan

    @pytest.mark.asyncio
    async def test_prompt_carries_budget(self):
        model = _Model(SearchPlan(mode="kb"))
        await plan_search(model, "вопрос", max_requests=10, retries=1)
        system = model.raw.messages[0]
        assert isinstance(system, SystemMessage)
        assert "5" in system.content
        assert isinstance(model.raw.messages[1], HumanMessage)

    @pytest.mark.asyncio
    async def test_model_failure_falls_back_to_full_question(self):
        model = _Model(RuntimeError("модель упала"))
        plan = await plan_search(model, "Как часто проводится СОУТ?")
        assert plan == SearchPlan(
            mode="kb", subqueries=["Как часто проводится СОУТ?"]
        )

    @pytest.mark.asyncio
    async def test_none_result_falls_back(self):
        model = _Model(None)
        plan = await plan_search(model, "вопрос")
        assert plan == SearchPlan(mode="kb", subqueries=["вопрос"])

    @pytest.mark.asyncio
    async def test_empty_subqueries_falls_back_to_question(self):
        model = _Model(SearchPlan(mode="kb", subqueries=[]))
        plan = await plan_search(model, "вопрос")
        assert plan == SearchPlan(mode="kb", subqueries=["вопрос"])

    @pytest.mark.asyncio
    async def test_blank_subqueries_dropped(self):
        model = _Model(SearchPlan(mode="kb", subqueries=["  ", "нормы", ""]))
        plan = await plan_search(model, "вопрос")
        assert plan == SearchPlan(mode="kb", subqueries=["нормы"])

    @pytest.mark.asyncio
    async def test_all_blank_subqueries_fall_back(self):
        model = _Model(SearchPlan(mode="kb", subqueries=["  "]))
        plan = await plan_search(model, "вопрос")
        assert plan == SearchPlan(mode="kb", subqueries=["вопрос"])

    @pytest.mark.asyncio
    async def test_subqueries_capped_by_budget(self):
        model = _Model(
            SearchPlan(mode="kb", subqueries=[f"q{i}" for i in range(7)])
        )
        plan = await plan_search(model, "вопрос", max_requests=10, retries=1)
        assert len(plan.subqueries) == 5

    @pytest.mark.asyncio
    async def test_always_answer_with_db_skips_model(self):
        model = _Model(SearchPlan(mode="chat"))
        plan = await plan_search(
            model, "Привет!", always_answer_with_db=True
        )
        assert plan == SearchPlan(mode="kb", subqueries=["Привет!"])
        assert model.raw.calls == 0

    @pytest.mark.asyncio
    async def test_always_answer_with_db_ignores_model_chat(self):
        model = _Model(SearchPlan(mode="chat"))
        plan = await plan_search(
            model, "Привет!", always_answer_with_db=True
        )
        assert plan.mode == "kb"

    def test_prompt_mentions_budget_placeholder(self):
        assert "{max_subqueries}" in PLAN_PROMPT