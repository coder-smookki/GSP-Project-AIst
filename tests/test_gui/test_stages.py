"""Отображение этапов графа: разбор события и стойкость отрисовки."""

import pytest

from gui.contract import parse_stage
from gui.stages import INTERRUPTED_DETAIL, StageView


class FakeStep:
    """Двойник ``cl.Step``: запоминает вызовы вместо отправки в интерфейс."""

    def __init__(self, title: str) -> None:
        """Создаёт двойник шага.

        Args:
            title: Заголовок шага.
        """
        self.name = title
        self.output = ""
        self.start: str | None = None
        self.end: str | None = None
        self.is_error = False
        self.sent = 0
        self.updated = 0

    async def send(self) -> None:
        """Считает отправки шага.

        Returns:
            None
        """
        self.sent += 1

    async def update(self) -> None:
        """Считает обновления шага.

        Returns:
            None
        """
        self.updated += 1


class StepRecorder:
    """Фабрика двойников шага с доступом к созданным экземплярам."""

    def __init__(self) -> None:
        """Создаёт пустой список созданных шагов."""
        self.steps: list[FakeStep] = []

    def __call__(self, title: str) -> FakeStep:
        """Создаёт двойник шага и запоминает его.

        Args:
            title: Заголовок шага.

        Returns:
            FakeStep: Созданный двойник.
        """
        step = FakeStep(title)
        self.steps.append(step)
        return step


class TestParseStage:
    def test_start_marks_stage_running(self):
        event = parse_stage({"stage": "plan", "status": "start"})
        assert event is not None
        assert event.running is True
        assert event.failed is False
        assert event.detail == ""
        assert event.title == "Планирование поиска"

    def test_plan_done_counts_subqueries(self):
        event = parse_stage(
            {"stage": "plan", "status": "done", "mode": "kb", "subqueries": 3}
        )
        assert event is not None
        assert event.running is False
        assert event.detail == "Подзапросов: 3"

    def test_plan_chat_mode_says_search_not_needed(self):
        event = parse_stage({"stage": "plan", "status": "done", "mode": "chat"})
        assert event is not None
        assert event.detail == "Поиск по базе не требуется"

    def test_search_branches_get_separate_keys(self):
        first = parse_stage(
            {"stage": "search_agent", "status": "start", "query": "наряд-допуск"}
        )
        second = parse_stage(
            {"stage": "search_agent", "status": "start", "query": "средства защиты"}
        )
        assert first is not None
        assert second is not None
        assert first.key != second.key
        assert "наряд-допуск" in first.title

    def test_search_start_and_done_share_key(self):
        start = parse_stage(
            {"stage": "search_agent", "status": "start", "query": "пожар"}
        )
        done = parse_stage(
            {
                "stage": "search_agent",
                "status": "done",
                "query": "пожар",
                "retrieved": 4,
                "failed": False,
            }
        )
        assert start is not None
        assert done is not None
        assert start.key == done.key
        assert done.detail == "Найдено фрагментов: 4"

    def test_search_failure_marks_error(self):
        event = parse_stage(
            {
                "stage": "search_agent",
                "status": "done",
                "query": "пожар",
                "retrieved": 0,
                "failed": True,
            }
        )
        assert event is not None
        assert event.failed is True
        assert event.detail == "Поиск недоступен"

    def test_search_with_hits_is_not_error_even_after_retry_failure(self):
        event = parse_stage(
            {
                "stage": "search_agent",
                "status": "done",
                "query": "пожар",
                "retrieved": 2,
                "failed": True,
            }
        )
        assert event is not None
        assert event.failed is False

    def test_long_query_is_shortened_in_title(self):
        event = parse_stage(
            {"stage": "search_agent", "status": "start", "query": "слово " * 40}
        )
        assert event is not None
        assert len(event.title) < 100

    def test_finalize_reports_sources(self):
        event = parse_stage({"stage": "finalize", "status": "done", "retrieved": 5})
        assert event is not None
        assert event.detail == "Источников: 5"
        assert event.failed is False

    def test_finalize_retrieval_failed_marks_error(self):
        event = parse_stage(
            {
                "stage": "finalize",
                "status": "done",
                "retrieved": 0,
                "retrieval_failed": True,
            }
        )
        assert event is not None
        assert event.failed is True

    def test_model_unavailable_marks_error(self):
        event = parse_stage({"stage": "assistant", "status": "model_unavailable"})
        assert event is not None
        assert event.failed is True
        assert event.detail == "Модель недоступна"

    def test_condense_distinguishes_empty_run(self):
        done = parse_stage({"stage": "condense", "status": "done", "condensed": 1})
        empty = parse_stage({"stage": "condense", "status": "done", "condensed": 0})
        assert done is not None
        assert empty is not None
        assert done.detail == "История свёрнута"
        assert empty.detail == "История не менялась"

    def test_event_without_status_is_treated_as_done(self):
        event = parse_stage({"stage": "finalize", "retrieved": 1})
        assert event is not None
        assert event.running is False

    def test_unknown_stage_is_ignored(self):
        assert parse_stage({"stage": "zzz", "status": "start"}) is None

    def test_empty_payload_is_ignored(self):
        assert parse_stage({}) is None

    def test_non_dict_payload_is_ignored(self):
        assert parse_stage("не словарь") is None  # type: ignore[arg-type]

    def test_no_em_dash_in_visible_text(self):
        payloads = [
            {"stage": "plan", "status": "done", "mode": "kb", "subqueries": 1},
            {"stage": "plan", "status": "done", "mode": "chat"},
            {"stage": "condense", "status": "done", "condensed": 1},
            {"stage": "search_agent", "status": "done", "retrieved": 0, "failed": True},
            {"stage": "finalize", "status": "done", "retrieval_failed": True},
            {"stage": "assistant", "status": "model_unavailable"},
        ]
        for payload in payloads:
            event = parse_stage(payload)
            assert event is not None
            assert "—" not in event.title
            assert "—" not in event.detail


class TestStageViewRendering:
    async def test_start_sends_running_step(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle({"stage": "plan", "status": "start"})

        assert len(recorder.steps) == 1
        step = recorder.steps[0]
        assert step.sent == 1
        assert step.start is not None
        assert step.end is None

    async def test_done_closes_the_same_step(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle({"stage": "plan", "status": "start"})
        await view.handle(
            {"stage": "plan", "status": "done", "mode": "kb", "subqueries": 2}
        )

        assert len(recorder.steps) == 1
        step = recorder.steps[0]
        assert step.updated == 1
        assert step.end is not None
        assert step.output == "Подзапросов: 2"
        assert step.is_error is False

    async def test_parallel_search_branches_get_own_steps(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle(
            {"stage": "search_agent", "status": "start", "query": "первый"}
        )
        await view.handle(
            {"stage": "search_agent", "status": "start", "query": "второй"}
        )
        await view.handle(
            {
                "stage": "search_agent",
                "status": "done",
                "query": "второй",
                "retrieved": 3,
            }
        )

        assert len(recorder.steps) == 2
        assert recorder.steps[1].end is not None
        assert recorder.steps[0].end is None

    async def test_repeated_start_does_not_duplicate_step(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle({"stage": "plan", "status": "start"})
        await view.handle({"stage": "plan", "status": "start"})

        assert len(recorder.steps) == 1

    async def test_done_without_start_creates_closed_step(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle({"stage": "finalize", "status": "done", "retrieved": 2})

        assert len(recorder.steps) == 1
        step = recorder.steps[0]
        assert step.sent == 1
        assert step.updated == 0
        assert step.end is not None
        assert step.output == "Источников: 2"

    async def test_close_finishes_hanging_steps(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle({"stage": "plan", "status": "start"})
        await view.close()

        step = recorder.steps[0]
        assert step.end is not None
        assert step.output == INTERRUPTED_DETAIL

    async def test_close_is_idempotent(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle({"stage": "plan", "status": "start"})
        await view.close()
        await view.close()

        assert recorder.steps[0].updated == 1

    async def test_unknown_stage_creates_no_step(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle({"stage": "zzz", "status": "start"})

        assert recorder.steps == []


class TestStageViewNeverRaises:
    async def test_factory_failure_is_swallowed(self):
        def broken_factory(title: str):
            raise RuntimeError("интерфейс недоступен")

        view = StageView(step_factory=broken_factory)
        await view.handle({"stage": "plan", "status": "start"})
        await view.handle({"stage": "plan", "status": "done"})

    async def test_send_failure_is_swallowed(self):
        class BrokenStep(FakeStep):
            async def send(self) -> None:
                """Имитирует сбой отправки шага.

                Returns:
                    None
                """
                raise RuntimeError("сокет закрыт")

        view = StageView(step_factory=BrokenStep)
        await view.handle({"stage": "plan", "status": "start"})

    async def test_update_failure_is_swallowed_on_close(self):
        class BrokenStep(FakeStep):
            async def update(self) -> None:
                """Имитирует сбой обновления шага.

                Returns:
                    None
                """
                raise RuntimeError("сокет закрыт")

        view = StageView(step_factory=BrokenStep)
        await view.handle({"stage": "plan", "status": "start"})
        await view.handle({"stage": "plan", "status": "done"})
        await view.close()

    async def test_malformed_payload_is_swallowed(self):
        recorder = StepRecorder()
        view = StageView(step_factory=recorder)

        await view.handle({"stage": None})
        await view.handle({})

        assert recorder.steps == []

    async def test_broken_stage_does_not_stop_the_next_one(self):
        calls: list[str] = []

        class HalfBrokenFactory:
            def __call__(self, title: str):
                calls.append(title)
                if len(calls) == 1:
                    raise RuntimeError("первый шаг не создался")
                return FakeStep(title)

        view = StageView(step_factory=HalfBrokenFactory())
        await view.handle({"stage": "plan", "status": "start"})
        await view.handle({"stage": "finalize", "status": "done", "retrieved": 1})

        assert len(calls) == 2


@pytest.mark.parametrize(
    "stage",
    ["condense", "plan", "search_agent", "finalize", "assistant"],
)
def test_every_graph_stage_has_a_title(stage: str):
    event = parse_stage({"stage": stage, "status": "start"})
    assert event is not None
    assert event.title
