"""Тесты метрик бенчмаркинга.

Метрики — то, что показывается в отчёте и сравнивается в A/B, поэтому они
проверяются на синтетических итогах с заранее известными значениями: ошибка
в арифметике метрики тихо превращается в неверный вывод о качестве.
"""

from __future__ import annotations

from eval.metrics import (
    MRR,
    AnswerChars,
    HitAtK,
    LatencyMedian,
    LatencyP95,
    NotFoundRate,
    RefusalRate,
    compute_metrics,
    default_metrics,
)
from eval.result import Outcome


def _outcome(
    rank=None,
    seconds: float = 0.1,
    refused: bool = False,
    answer: str | None = None,
    answerable: bool = True,
) -> Outcome:
    """Синтетический итог по вопросу."""
    return Outcome(
        question="вопрос",
        answerable=answerable,
        rank=rank,
        seconds=seconds,
        refused=refused,
        answer=answer,
    )


class TestHitAtK:
    """Доля ответов в первых k фрагментах."""

    def test_counts_only_top_positions(self) -> None:
        """Ранки 1, 2, 6, None: hit@1 = 1/4, hit@3 = 2/4."""
        outcomes = [_outcome(1), _outcome(2), _outcome(6), _outcome(None)]
        assert HitAtK(1).evaluate(outcomes) == 0.25
        assert HitAtK(3).evaluate(outcomes) == 0.5
        assert HitAtK(5).evaluate(outcomes) == 0.5

    def test_ignores_unanswerable(self) -> None:
        """Вопросы без ответа в подсчёт hit@k не входят."""
        outcomes = [_outcome(1), _outcome(None, answerable=False)]
        assert HitAtK(1).evaluate(outcomes) == 1.0

    def test_empty_returns_zero(self) -> None:
        """Пустой набор не роняет метрику."""
        assert HitAtK(3).evaluate([]) == 0.0


class TestMRR:
    """Средняя обратная позиция."""

    def test_known_value(self) -> None:
        """Ранки 1, 2, None, 5: (1 + 1/2 + 0 + 1/5) / 4 = 0.425."""
        outcomes = [_outcome(1), _outcome(2), _outcome(None), _outcome(5)]
        assert MRR().evaluate(outcomes) == 0.425

    def test_empty_returns_zero(self) -> None:
        assert MRR().evaluate([]) == 0.0


class TestNotFoundRate:
    """Доля ответов, не дошедших до контекста."""

    def test_known_value(self) -> None:
        outcomes = [_outcome(1), _outcome(2), _outcome(None), _outcome(None)]
        assert NotFoundRate().evaluate(outcomes) == 0.5

    def test_ignores_unanswerable(self) -> None:
        """Не-найдено считается только по отвечаемым."""
        outcomes = [_outcome(None), _outcome(None, answerable=False)]
        assert NotFoundRate().evaluate(outcomes) == 1.0


class TestRefusalRate:
    """Доля честных отказов на вопросах без ответа."""

    def test_known_value(self) -> None:
        outcomes = [
            _outcome(refused=True, answerable=False),
            _outcome(refused=True, answerable=False),
            _outcome(refused=False, answerable=False),
        ]
        assert RefusalRate().evaluate(outcomes) == 2 / 3

    def test_empty_returns_zero(self) -> None:
        assert RefusalRate().evaluate([]) == 0.0


class TestLatency:
    """Квантили времени поиска."""

    def test_median(self) -> None:
        outcomes = [_outcome(seconds=s) for s in (1.0, 2.0, 3.0, 4.0, 5.0)]
        assert LatencyMedian().evaluate(outcomes) == 3.0

    def test_p95(self) -> None:
        outcomes = [_outcome(seconds=s) for s in (1.0, 2.0, 3.0, 4.0, 5.0)]
        assert LatencyP95().evaluate(outcomes) == 5.0


class TestAnswerChars:
    """Средняя длина ответа."""

    def test_known_value(self) -> None:
        outcomes = [
            _outcome(answer="0123456789"),
            _outcome(answer="abcde"),
        ]
        assert AnswerChars().evaluate(outcomes) == 7.5

    def test_skips_missing_answers(self) -> None:
        outcomes = [_outcome(answer=None), _outcome(answer="0123456789")]
        assert AnswerChars().evaluate(outcomes) == 10.0

    def test_empty_returns_zero(self) -> None:
        assert AnswerChars().evaluate([]) == 0.0


class TestRegistry:
    """Реестр метрик."""

    def test_default_set_is_complete(self) -> None:
        """Стандартный набор содержит все ожидаемые метрики."""
        keys = {m.key for m in default_metrics()}
        assert keys == {
            "hit@1",
            "hit@3",
            "hit@5",
            "mrr",
            "not_found",
            "refusal_rate",
            "latency_median",
            "latency_p95",
            "answer_chars",
        }

    def test_custom_k_values(self) -> None:
        """Набор hit@k настраивается."""
        keys = {m.key for m in default_metrics(k_values=(1, 10))}
        assert "hit@1" in keys
        assert "hit@10" in keys
        assert "hit@5" not in keys

    def test_compute_metrics(self) -> None:
        """compute_metrics возвращает значения по ключам."""
        outcomes = [
            _outcome(1, seconds=0.1),
            _outcome(2, seconds=0.2),
            _outcome(None, seconds=0.3),
        ]
        values = compute_metrics(default_metrics(k_values=(1,)), outcomes)
        assert values["hit@1"] == 1 / 3
        assert values["mrr"] == 0.5
        assert values["latency_median"] == 0.2

    def test_metric_kinds(self) -> None:
        """Категориальные и непрерывные помечены для выбора A/B-теста."""
        by_key = {m.key: m for m in default_metrics()}
        assert by_key["hit@1"].kind == "categorical"
        assert by_key["mrr"].kind == "continuous"
