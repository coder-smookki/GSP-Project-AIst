"""Тесты A/B-статистики: парный пермутационный тест.

Ключевые свойства: детерминизм, направление вердикта относительно
``higher_is_better``, корректность при сильном разделении (p-value должен
быть мал, а не 1), отказ при разном числе вопросов.
"""

from __future__ import annotations

from eval.metrics import HitAtK, LatencyMedian
from eval.result import Outcome, RunResult
from eval.stats import SIGNIFICANCE, compare_all, compare_metric


def make_result(
    ranks: list[int | None], seconds: list[float] | float = 0.1
) -> RunResult:
    """Собирает результат прогона по рангам ответов.

    Args:
        ranks: Ранг найденного ответа по вопросу; ``None`` — не найдено.
        seconds: Время поиска на каждый вопрос; число повторяется на все.

    Returns:
        RunResult: Результат с одним вопросом на ранг.
    """
    if isinstance(seconds, (int, float)):
        seconds = [float(seconds)] * len(ranks)
    outcomes = [
        Outcome(question=f"вопрос {i}", answerable=True, rank=rank, seconds=s)
        for i, (rank, s) in enumerate(zip(ranks, seconds, strict=True))
    ]
    return RunResult(
        variant="test", environment="env", metrics={}, outcomes=outcomes
    )


class TestCompareMetric:
    """Парный пермутационный тест одной метрики."""

    def test_identical_variants_no_diff(self) -> None:
        """Одинаковые результаты — без разницы и большой p-value."""
        base = make_result([1, 2, 1, 3])
        variant = make_result([1, 2, 1, 3])
        verdict = compare_metric(HitAtK(1), base, variant)
        assert verdict.direction == "no_diff"
        assert verdict.p_value == 1.0

    def test_better_variant_better_direction(self) -> None:
        """Заметно лучший вариант получает направление better."""
        base = make_result([5, 5, 5, 5, 5, 5])
        variant = make_result([1, 1, 1, 1, 1, 1])
        verdict = compare_metric(HitAtK(1), base, variant)
        assert verdict.direction == "better"
        assert verdict.p_value < SIGNIFICANCE

    def test_worse_variant_worse_direction(self) -> None:
        """Заметно худший вариант получает направление worse."""
        base = make_result([1, 1, 1, 1, 1, 1])
        variant = make_result([5, 5, 5, 5, 5, 5])
        verdict = compare_metric(HitAtK(1), base, variant)
        assert verdict.direction == "worse"
        assert verdict.p_value < SIGNIFICANCE

    def test_lower_latency_is_better(self) -> None:
        """Меньшее время — улучшение (higher_is_better=False)."""
        base = make_result([1] * 10, seconds=[9.0 + 0.1 * i for i in range(10)])
        variant = make_result([1] * 10, seconds=[1.0 + 0.1 * i for i in range(10)])
        verdict = compare_metric(LatencyMedian(), base, variant)
        assert verdict.direction == "better"
        assert verdict.p_value < SIGNIFICANCE

    def test_unequal_question_counts_raise(self) -> None:
        """Разное число вопросов — парный тест невозможен."""
        base = make_result([1, 2, 3])
        variant = make_result([1, 2])
        try:
            compare_metric(HitAtK(1), base, variant)
        except ValueError:
            return
        raise AssertionError("ожидалось ValueError")

    def test_deterministic(self) -> None:
        """Один и тот же вызов даёт одинаковый p-value."""
        base = make_result([3, 4, 5, 3, 2, 5])
        variant = make_result([1, 2, 1, 2, 1, 1])
        first = compare_metric(HitAtK(1), base, variant)
        second = compare_metric(HitAtK(1), base, variant)
        assert first.p_value == second.p_value
        assert first.direction == second.direction


class TestCompareAll:
    """Сравнение по списку метрик."""

    def test_verdicts_in_metric_order(self) -> None:
        """Вердикты возвращаются в порядке метрик."""
        metrics = [HitAtK(1), HitAtK(3)]
        verdicts = compare_all(metrics, make_result([5, 5]), make_result([1, 1]))
        assert [v.metric for v in verdicts] == ["hit@1", "hit@3"]

    def test_values_reported(self) -> None:
        """В вердикте лежат значения метрик обоих вариантов."""
        base = make_result([5, 5])
        variant = make_result([1, 1])
        verdict = compare_all([HitAtK(1)], base, variant)[0]
        assert verdict.base_value == 0.0
        assert verdict.variant_value == 1.0
