"""Тесты модели состояния графа."""

import pytest
from pydantic import ValidationError

from agent.state import SearchPlan


class TestSearchPlan:
    def test_chat_mode_defaults_to_empty_subqueries(self):
        plan = SearchPlan(mode="chat")
        assert plan.mode == "chat"
        assert plan.subqueries == []

    def test_kb_mode_keeps_subqueries(self):
        plan = SearchPlan(mode="kb", subqueries=["СОУТ", "ГОСТ 4.512"])
        assert plan.mode == "kb"
        assert plan.subqueries == ["СОУТ", "ГОСТ 4.512"]

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            SearchPlan(mode="other")

    def test_round_trip_through_model_dump(self):
        plan = SearchPlan(mode="kb", subqueries=["а", "б"])
        assert SearchPlan(**plan.model_dump()) == plan

    def test_equality_is_value_based(self):
        assert SearchPlan(mode="chat") == SearchPlan(mode="chat")
        assert SearchPlan(mode="kb", subqueries=["x"]) != SearchPlan(
            mode="kb", subqueries=["y"]
        )