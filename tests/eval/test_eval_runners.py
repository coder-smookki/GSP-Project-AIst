"""Тесты раннеров бенчмаркинга.

Раннеры проверяются на фейках поиска и агента: сами они не должны знать про
инфраструктуру (DIP), значит весь прогон обязан работать на синтетических
зависимостях с известными ответами.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval.config import Environment, RunConfig, Variant
from eval.golden import DEFAULT_GOLDEN
from eval.metrics import default_metrics
from eval.result import RunResult
from eval.runners import BuildResult, GoldenSetRunner, OperationalRunner

ENV = Environment(
    platform="test",
    python="3.14",
    llm_model="m",
    llm_support_model="s",
    embedding_model="e",
    retrieval_top_k=5,
    rerank_enabled=False,
    rerank_model="",
    grading_enabled=False,
    verify_numbers=True,
)


class _Chunk:
    """Фрагмент с кодом документа и текстом."""

    def __init__(self, doc_code: str, text: str) -> None:
        self.doc_code = doc_code
        self.text = text


class _FakeSearch:
    """Поиск по заранее заданным выдачам."""

    def __init__(self, hits_by_question: dict[str, list[tuple[_Chunk, float]]]) -> None:
        self._hits = hits_by_question

    async def query(self, question: str):
        return self._hits.get(question, [])


class _FakeAgent:
    """Агент, отвечающий фиксированным текстом."""

    def __init__(self, text: str = "Ответ") -> None:
        self._text = text

    async def ainvoke(self, state, config=None):
        message = SimpleNamespace(content=self._text)
        return {"messages": [message]}


def _write_golden(tmp_path, answerable, unanswerable) -> str:
    """Пишет золотой набор во временный файл."""
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {"answerable": answerable, "unanswerable": unanswerable}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return str(path)


def _answerable(question: str, quote: str) -> dict:
    return {"question": question, "quote": quote, "kind": "фактический"}


def _unanswerable(question: str) -> dict:
    return {"question": question, "kind": "вне области"}


class TestGoldenSetRunner:
    """Прогон на золотом наборе с фейковым поиском."""

    def _run(self, tmp_path, search, config=None):
        golden_path = _write_golden(
            tmp_path,
            [
                _answerable("вопрос один", "цитата один"),
                _answerable("вопрос два", "цитата два"),
            ],
            [_unanswerable("вопрос без ответа")],
        )

        def build(_v, _c):
            return BuildResult(search=search, environment=ENV)

        runner = GoldenSetRunner(build=build, metrics=default_metrics(k_values=(1,)))
        variant = Variant(name="base")
        config = config or RunConfig(golden=golden_path)
        return runner.run(variant, config)

    async def test_metrics_on_known_hits(self, tmp_path) -> None:
        """Первый вопрос на позиции 1, второй не найден: hit@1 = 0.5."""
        hits = {
            "вопрос один": [(_Chunk("doc", "текст цитата один текст"), 0.9)],
        }
        result = await self._run(tmp_path, _FakeSearch(hits))
        assert result.variant == "base"
        assert result.environment == ENV.fingerprint()
        assert result.metrics["hit@1"] == 0.5
        assert result.metrics["mrr"] == 0.5
        assert result.metrics["not_found"] == 0.5

    async def test_outcomes_are_tagged(self, tmp_path) -> None:
        """Вопросы размечены флагом answerable."""
        result = await self._run(tmp_path, _FakeSearch({}))
        flags = [o.answerable for o in result.outcomes]
        assert flags == [True, True, False]

    async def test_with_answers_checks_refusal(self, tmp_path) -> None:
        """С агентом проверяется честный отказ по тексту ответа."""
        agent = _FakeAgent(text="В базе знаний информации нет")
        hits = {"вопрос без ответа": [(_Chunk("doc", "прочий текст"), 0.6)]}
        search = _FakeSearch(hits)
        config = RunConfig(
            golden=_write_golden(tmp_path, [], [_unanswerable("вопрос без ответа")]),
            with_answers=True,
        )

        def build(_v, _c):
            return BuildResult(search=search, agent=agent, environment=ENV)

        runner = GoldenSetRunner(build=build)
        result = await runner.run(Variant(name="base"), config)
        assert result.metrics["refusal_rate"] == 1.0

    async def test_requires_search(self, tmp_path) -> None:
        """Без поиска в BuildResult раннер падает с понятной ошибкой."""

        def build(_v, _c):
            return BuildResult(environment=ENV)

        runner = GoldenSetRunner(build=build)
        config = RunConfig(golden=DEFAULT_GOLDEN)
        with pytest.raises(ValueError, match="поиск"):
            await runner.run(Variant(name="base"), config)

    async def test_meta_describes_run(self, tmp_path) -> None:
        """В метаданных лежит описание прогона."""
        result = await self._run(tmp_path, _FakeSearch({}))
        assert result.meta["with_answers"] is False
        assert "golden" in result.meta


class TestOperationalRunner:
    """Операционный замер с фейковым агентом."""

    async def test_metrics_and_meta(self) -> None:
        """Считаются латенси и пропускная способность, замеры в meta."""

        def build(_v, _c):
            return BuildResult(
                agent=_FakeAgent(text="длинный ответ" * 5), environment=ENV
            )

        runner = OperationalRunner(build=build, questions=(("тест", "Вопрос?"),))
        result = await runner.run(
            Variant(name="base"), RunConfig(repeats=2, concurrency=(1,))
        )
        assert "latency_median" in result.metrics
        assert result.metrics["answers_per_minute"] > 0
        assert len(result.meta["sequential"]) == 1
        assert len(result.meta["concurrency"]) == 1
        assert result.meta["concurrency"][0]["одновременно"] == 1

    async def test_empty_concurrency(self) -> None:
        """Без concurrency блок одновременных обращений пуст."""

        def build(_v, _c):
            return BuildResult(agent=_FakeAgent(), environment=ENV)

        runner = OperationalRunner(build=build, questions=(("тест", "Вопрос?"),))
        result = await runner.run(
            Variant(name="base"), RunConfig(repeats=1, concurrency=())
        )
        assert result.meta["concurrency"] == []
        assert len(result.outcomes) == 1

    async def test_requires_agent(self) -> None:
        """Без агента в BuildResult раннер падает с понятной ошибкой."""

        def build(_v, _c):
            return BuildResult(environment=ENV)

        runner = OperationalRunner(build=build)
        with pytest.raises(ValueError, match="агент"):
            await runner.run(Variant(name="base"), RunConfig(repeats=1))


class TestResultRoundTrip:
    """Прогон раннера совместим с сериализацией RunResult."""

    async def test_runner_result_round_trips(self, tmp_path) -> None:
        """Результат раннера сохраняется и восстанавливается."""
        golden_path = _write_golden(
            tmp_path,
            [_answerable("вопрос один", "цитата один")],
            [],
        )
        hits = {"вопрос один": [(_Chunk("doc", "текст цитата один"), 0.9)]}

        def build(_v, _c):
            return BuildResult(search=_FakeSearch(hits), environment=ENV)

        runner = GoldenSetRunner(build=build)
        result = await runner.run(Variant(name="base"), RunConfig(golden=golden_path))
        restored = RunResult.from_dict(result.to_dict())
        assert restored.variant == result.variant
        assert restored.metrics == result.metrics
        assert len(restored.outcomes) == 1
