"""Тесты A/B-статистики: тесты по типу метрики.

Ключевые свойства: детерминизм, направление вердикта относительно
``higher_is_better``, корректность при сильном разделении (p-value должен
быть мал, а не 1), отказ при разном числе вопросов. Точные тесты
(МакНемар, Уилкоксон) сверяются с известными значениями scipy.
"""

from __future__ import annotations

import pytest

try:
    from scipy.stats import binomtest

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - без scipy тесты с пропуском
    HAS_SCIPY = False
    binomtest = None

from eval.metrics import HitAtK, LatencyMedian, RefusalRate
from eval.result import Outcome, RunResult
from eval.stats import (
    SIGNIFICANCE,
    McNemarTest,
    PairedPermutationTest,
    WilcoxonTest,
    compare_all,
    compare_bootstrapped,
    compare_metric,
    default_test,
)

NEEDS_SCIPY = pytest.mark.skipif(
    not HAS_SCIPY,
    reason="нужен scipy для точных тестов: uv sync --extra benchmark",
)


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
    return RunResult(variant="test", environment="env", metrics={}, outcomes=outcomes)


def make_unanswerable(refused: list[bool]) -> RunResult:
    """Собирает результат по вопросам без ответа в базе.

    Args:
        refused: Отказалась ли система по каждому вопросу.

    Returns:
        RunResult: Результат с вопросами без ответа.
    """
    outcomes = [
        Outcome(question=f"отказ {i}", answerable=False, refused=value)
        for i, value in enumerate(refused)
    ]
    return RunResult(variant="test", environment="env", metrics={}, outcomes=outcomes)


class TestCompareMetric:
    """Сравнение метрики тестом по умолчанию (по типу метрики)."""

    def test_identical_variants_no_diff(self) -> None:
        """Одинаковые результаты — без разницы и большой p-value."""
        base = make_result([1, 2, 1, 3])
        variant = make_result([1, 2, 1, 3])
        verdict = compare_metric(HitAtK(1), base, variant)
        assert verdict.direction == "no_diff"
        assert verdict.p_value == 1.0
        if HAS_SCIPY:
            assert verdict.test == "mcnemar"

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


class TestMcNemar:
    """Точный тест МакНемара на известных биномиальных кейсах."""

    @NEEDS_SCIPY
    def test_known_discordant_counts(self) -> None:
        """9 против 1 расходящейся пары — p-value из биномиального теста."""
        base = make_result([None] * 9 + [1])
        variant = make_result([1] * 9 + [None])
        verdict = McNemarTest().compare(HitAtK(1), base, variant)
        expected = binomtest(1, 10, 0.5).pvalue
        assert verdict.p_value == expected
        assert verdict.direction == "better"
        assert verdict.p_value < SIGNIFICANCE

    def test_no_discordant_pairs(self) -> None:
        """Нет расходящихся пар — нет разницы."""
        base = make_result([1, 2, None])
        variant = make_result([1, 2, None])
        verdict = McNemarTest().compare(HitAtK(1), base, variant)
        assert verdict.p_value == 1.0
        assert verdict.direction == "no_diff"

    @NEEDS_SCIPY
    def test_refusal_improvement(self) -> None:
        """Вариант отказывает там, где base отвечал, — улучшение."""
        base = make_unanswerable([False] * 7 + [True] * 3)
        variant = make_unanswerable([True] * 10)
        verdict = McNemarTest().compare(RefusalRate(), base, variant)
        assert verdict.direction == "better"
        assert verdict.p_value == binomtest(0, 7, 0.5).pvalue
        assert verdict.p_value < SIGNIFICANCE


class TestWilcoxon:
    """Знаково-ранговый критерий Уилкоксона."""

    @NEEDS_SCIPY
    def test_lower_latency_better(self) -> None:
        """Заметно меньшее время — значимое улучшение."""
        base = make_result([1] * 10, seconds=[10.0] * 10)
        variant = make_result([1] * 10, seconds=[5.0] * 10)
        verdict = WilcoxonTest().compare(LatencyMedian(), base, variant)
        assert verdict.direction == "better"
        assert verdict.p_value < SIGNIFICANCE

    @NEEDS_SCIPY
    def test_identical_returns_no_diff(self) -> None:
        """Одинаковые измерения — без разницы."""
        base = make_result([1, 2, 3], seconds=[1.0, 2.0, 3.0])
        variant = make_result([1, 2, 3], seconds=[1.0, 2.0, 3.0])
        verdict = WilcoxonTest().compare(LatencyMedian(), base, variant)
        assert verdict.p_value == 1.0
        assert verdict.direction == "no_diff"


class TestDefaultTest:
    """Выбор теста по типу метрики."""

    @NEEDS_SCIPY
    def test_categorical_uses_mcnemar(self) -> None:
        """Категориальные метрики — МакНемар."""
        assert isinstance(default_test(HitAtK(1)), McNemarTest)

    @NEEDS_SCIPY
    def test_continuous_uses_wilcoxon(self) -> None:
        """Непрерывные метрики — Уилкоксон."""
        assert isinstance(default_test(LatencyMedian()), WilcoxonTest)

    def test_permutation_explicit(self) -> None:
        """Пермутационный тест доступен явно."""
        base = make_result([5] * 8)
        variant = make_result([1] * 8)
        verdict = compare_metric(HitAtK(1), base, variant, test=PairedPermutationTest())
        assert verdict.test == "permutation (paired)"
        assert verdict.direction == "better"


class TestBootstrapComparison:
    """Сравнение по бутстрап-выборкам: пары — выборки, а не вопросы."""

    @staticmethod
    def _result(samples: list[dict[str, float]]) -> RunResult:
        """Собирает агрегированный результат с выборками в ``meta``."""
        return RunResult(
            variant="test",
            environment="env",
            metrics=samples[-1],
            outcomes=[],
            meta={"bootstrap": {"n": len(samples), "samples": samples, "std": {}}},
        )

    def test_consistent_improvement(self) -> None:
        """Вариант выше на каждой выборке — значимое улучшение."""
        base = self._result([{"hit@1": 0.6} for _ in range(9)])
        variant = self._result([{"hit@1": 0.9} for _ in range(9)])
        verdict = compare_bootstrapped([HitAtK(1)], base, variant)[0]
        assert verdict.test == "bootstrap"
        assert verdict.direction == "better"
        assert verdict.p_value < SIGNIFICANCE
        assert verdict.base_value == pytest.approx(0.6)
        assert verdict.variant_value == pytest.approx(0.9)

    def test_identical_no_diff(self) -> None:
        """Одинаковые значения на каждой выборке — без разницы."""
        base = self._result([{"hit@1": 0.7} for _ in range(5)])
        variant = self._result([{"hit@1": 0.7} for _ in range(5)])
        verdict = compare_bootstrapped([HitAtK(1)], base, variant)[0]
        assert verdict.direction == "no_diff"
        assert verdict.p_value == 1.0

    def test_latency_lower_better(self) -> None:
        """Для непрерывной метрики меньшее время — улучшение."""
        base = self._result([{"latency_median": 1.0} for _ in range(9)])
        variant = self._result([{"latency_median": 0.5} for _ in range(9)])
        verdict = compare_bootstrapped([LatencyMedian()], base, variant)[0]
        assert verdict.direction == "better"

    def test_falls_back_without_bootstrap_meta(self) -> None:
        """Без метаданных выборок — обычное сравнение по вопросам."""
        base = make_result([None] * 8)
        variant = make_result([1] * 8)
        verdict = compare_bootstrapped([HitAtK(1)], base, variant)[0]
        assert verdict.test != "bootstrap"
        assert verdict.direction == "better"

    def test_mismatched_sample_counts(self) -> None:
        """Разное число выборок у вариантов — ошибка."""
        base = self._result([{"hit@1": 0.6} for _ in range(3)])
        variant = self._result([{"hit@1": 0.8} for _ in range(4)])
        with pytest.raises(ValueError, match="выборок"):
            compare_bootstrapped([HitAtK(1)], base, variant)
