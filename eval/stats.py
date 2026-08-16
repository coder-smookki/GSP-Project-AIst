"""Статистическое сравнение вариантов: тесты по типу метрики.

Сравнение двух вариантов, прогнанных на одном золотом наборе, — парное
(вопросы совпадают и идут в одном порядке), поэтому тесты парные:

* **категориальные** метрики (hit@k, честный отказ) — точный тест МакНемара:
  значение имеют только расходящиеся пары (в base ответ нашёлся, в варианте
  нет — и наоборот), значимость — биномиальный тест доли 1/2;
* **непрерывные** метрики (MRR, латенси) — знаково-ранговый критерий
  Уилкоксона для связных выборок;
* **без scipy** — парный пермутационный тест, одинаково честный для обоих
  типов.

Каждый тест реализует протокол ``StatTest`` (OCP): метрика выбирает тест по
``kind``, новый тест добавляется классом без правки существующих.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from eval.metrics import BaseMetric
from eval.result import RunResult, Verdict

#: Порог значимости: ``p < 0.05`` даёт направленный вердикт.
SIGNIFICANCE = 0.05

#: Перестановок для фолбэк-теста без scipy.
PERMUTATIONS = 5000


@runtime_checkable
class StatTest(Protocol):
    """Парный тест сравнения двух вариантов по одной метрике.

    Attributes:
        name: Короткое имя теста для отчёта.
    """

    name: str

    def compare(
        self,
        metric: BaseMetric,
        base: RunResult,
        variant: RunResult,
        seed: int = 42,
    ) -> Verdict:
        """Сравнивает метрику двух вариантов.

        Args:
            metric: Метрика для сравнения.
            base: Результат базового варианта.
            variant: Результат нового варианта.
            seed: Зерно ГПСЧ для тестов с перестановками.

        Returns:
            Verdict: Значения метрики, p-value и направление.
        """
        raise NotImplementedError


def _seed_for(seed: int, metric_key: str) -> int:
    """Детерминированный под-сид для метрики.

    Args:
        seed: Базовое зерно.
        metric_key: Ключ метрики.

    Returns:
        int: Зерно, не зависящее от порядка вызова метрик.
    """
    return seed + sum(map(ord, metric_key))


def _direction(metric: BaseMetric, observed: float, p_value: float) -> str:
    """Определяет направление изменения относительно base.

    Args:
        metric: Метрика (направление улучшения задаёт её ``higher_is_better``).
        observed: Наблюдаемая разность ``variant - base``.
        p_value: Достигнутый уровень значимости.

    Returns:
        str: ``better``, ``worse`` или ``no_diff``.
    """
    if observed == 0 or p_value >= SIGNIFICANCE or metric.higher_is_better is None:
        return "no_diff"
    better = (observed > 0) == metric.higher_is_better
    return "better" if better else "worse"


def _paired_values(
    metric: BaseMetric, base: RunResult, variant: RunResult
) -> tuple[list[float], list[float]]:
    """Значения метрики по вопросам обоих вариантов (парно).

    Args:
        metric: Метрика.
        base: Результат базового варианта.
        variant: Результат нового варианта.

    Returns:
        tuple[list[float], list[float]]: Значения base и варианта по одному
        вопросу, в одном порядке.

    Raises:
        ValueError: Варианты прогнаны на разном числе вопросов.
    """
    if len(base.outcomes) != len(variant.outcomes):
        raise ValueError("варианты прогнаны на разном числе вопросов")
    base_rel = metric.relevant(base.outcomes)
    variant_rel = metric.relevant(variant.outcomes)
    if len(base_rel) != len(variant_rel):
        raise ValueError("варианты прогнаны на разном числе вопросов")
    return (
        metric.per_question(base_rel),
        metric.per_question(variant_rel),
    )


class McNemarTest:
    """Точный тест МакНемара для категориальных метрик.

    Считаются только расходящиеся пары: в base вопрос не отвечен, в варианте
    отвечен (``n01``) и наоборот (``n10``). При отсутствии различий доли этих
    пар должны быть равны, то есть ``n01`` подчиняется биномиальному
    распределению с параметром 1/2 — p-value даёт ``scipy.stats.binomtest``.
    """

    name = "mcnemar"

    def compare(
        self,
        metric: BaseMetric,
        base: RunResult,
        variant: RunResult,
        seed: int = 42,
    ) -> Verdict:
        base_values, variant_values = _paired_values(metric, base, variant)

        n01 = n10 = 0
        for base_value, variant_value in zip(base_values, variant_values, strict=True):
            if variant_value and not base_value:
                n01 += 1
            elif base_value and not variant_value:
                n10 += 1

        total = n01 + n10
        if total == 0:
            p_value = 1.0
        else:
            from scipy.stats import binomtest

            p_value = binomtest(min(n01, n10), total, 0.5).pvalue

        observed = metric.evaluate(variant.outcomes) - metric.evaluate(base.outcomes)
        return Verdict(
            metric=metric.key,
            test=self.name,
            p_value=p_value,
            base_value=metric.evaluate(base.outcomes),
            variant_value=metric.evaluate(variant.outcomes),
            direction=_direction(metric, observed, p_value),
        )


class WilcoxonTest:
    """Знаково-ранговый критерий Уилкоксона для непрерывных метрик.

    Связные выборки: ранжируются модули разностей по вопросам, статистика —
    сумма рангов с положительным знаком. Требует scipy; не все разности равны
    нулю, иначе критерий не определён, и возвращается ``p = 1``.
    """

    name = "wilcoxon"

    def compare(
        self,
        metric: BaseMetric,
        base: RunResult,
        variant: RunResult,
        seed: int = 42,
    ) -> Verdict:
        base_values, variant_values = _paired_values(metric, base, variant)

        if not any(
            variant_values[i] != base_values[i] for i in range(len(base_values))
        ):
            p_value = 1.0
        else:
            from scipy.stats import wilcoxon

            result = wilcoxon(variant_values, base_values)
            p_value = float(getattr(result, "pvalue", 1.0))
            if math.isnan(p_value):
                p_value = 1.0

        observed = metric.evaluate(variant.outcomes) - metric.evaluate(base.outcomes)
        return Verdict(
            metric=metric.key,
            test=self.name,
            p_value=p_value,
            base_value=metric.evaluate(base.outcomes),
            variant_value=metric.evaluate(variant.outcomes),
            direction=_direction(metric, observed, p_value),
        )


class PairedPermutationTest:
    """Парный пермутационный тест — фолбэк без scipy.

    Для каждой пары итогов решение «поменять местами» принимается честной
    монетой, разность метрики пересчитывается на переставленных группах;
    p-value — доля перестановок, где разность не слабее наблюдаемой. Одинаково
    работает для категориальных и непрерывных метрик, точен на малых наборах.
    """

    name = "permutation (paired)"

    def compare(
        self,
        metric: BaseMetric,
        base: RunResult,
        variant: RunResult,
        seed: int = 42,
    ) -> Verdict:
        if len(base.outcomes) != len(variant.outcomes):
            raise ValueError("варианты прогнаны на разном числе вопросов")
        base_outcomes = base.outcomes
        variant_outcomes = variant.outcomes

        observed = metric.evaluate(variant_outcomes) - metric.evaluate(base_outcomes)
        rng = random.Random(_seed_for(seed, metric.key))
        n = len(base_outcomes)
        extremes = 0
        for _ in range(PERMUTATIONS):
            swapped_base: list = []
            swapped_variant: list = []
            for i in range(n):
                if rng.random() < 0.5:
                    swapped_base.append(variant_outcomes[i])
                    swapped_variant.append(base_outcomes[i])
                else:
                    swapped_base.append(base_outcomes[i])
                    swapped_variant.append(variant_outcomes[i])
            diff = metric.evaluate(swapped_variant) - metric.evaluate(swapped_base)
            if abs(diff) >= abs(observed):
                extremes += 1

        p_value = (extremes + 1) / (PERMUTATIONS + 1)
        return Verdict(
            metric=metric.key,
            test=self.name,
            p_value=p_value,
            base_value=metric.evaluate(base_outcomes),
            variant_value=metric.evaluate(variant_outcomes),
            direction=_direction(metric, observed, p_value),
        )


def default_test(metric: BaseMetric) -> StatTest:
    """Выбирает тест по типу метрики.

    Args:
        metric: Метрика.

    Returns:
        StatTest: МакНемар для категориальных, Уилкоксон для непрерывных;
        при отсутствии scipy — пермутационный тест.
    """
    try:
        import scipy  # noqa: F401
    except ImportError:
        return PairedPermutationTest()
    if metric.kind == "categorical":
        return McNemarTest()
    return WilcoxonTest()


def compare_metric(
    metric: BaseMetric,
    base: RunResult,
    variant: RunResult,
    test: StatTest | None = None,
    seed: int = 42,
) -> Verdict:
    """Сравнивает одну метрику двух вариантов.

    Args:
        metric: Метрика для сравнения.
        base: Результат базового варианта.
        variant: Результат нового варианта.
        test: Тест; по умолчанию выбирается по типу метрики.
        seed: Зерно ГПСЧ для пермутационного теста.

    Returns:
        Verdict: Значения метрики, p-value и направление.
    """
    if test is None:
        test = default_test(metric)
    return test.compare(metric, base, variant, seed=seed)


def compare_all(
    metrics: Iterable[BaseMetric],
    base: RunResult,
    variant: RunResult,
    seed: int = 42,
) -> list[Verdict]:
    """Сравнивает все метрики двух вариантов.

    Args:
        metrics: Метрики для сравнения.
        base: Результат базового варианта.
        variant: Результат нового варианта.
        seed: Зерно ГПСЧ.

    Returns:
        list[Verdict]: Вердикты по каждой метрике в порядке передачи.
    """
    return [compare_metric(m, base, variant, seed=seed) for m in metrics]


def _paired_sample_p_value(diffs: list[float], seed: int) -> float:
    """p-value для разностей метрик по бутстрап-выборкам.

    Разности берутся по одной на выборку: ``d_k = variant_k - base_k``. При
    ``scipy`` — знаково-ранговый критерий Уилкоксона (если разности не все
    нулевые и их не слишком мало), иначе/в крайнем случае — парный
    пермутационный тест: знаки разностей честной монетой, статистика — сумма
    разностей.

    Args:
        diffs: Разности метрики по выборкам.
        seed: Зерно ГПСЧ для пермутаций.

    Returns:
        float: p-value в диапазоне [0, 1].
    """
    if not any(d != 0 for d in diffs):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        if len(diffs) >= 5:
            result = wilcoxon(diffs)
            p_value = float(getattr(result, "pvalue", 1.0))
            if not math.isnan(p_value):
                return p_value
    except (ImportError, ValueError):
        pass

    observed = abs(sum(diffs))
    rng = random.Random(_seed_for(seed, "bootstrap"))
    extremes = 0
    for _ in range(PERMUTATIONS):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(sum(flipped)) >= observed:
            extremes += 1
    return (extremes + 1) / (PERMUTATIONS + 1)


def compare_bootstrapped(
    metrics: Iterable[BaseMetric],
    base: RunResult,
    variant: RunResult,
    seed: int = 42,
) -> list[Verdict]:
    """Сравнивает метрики двух вариантов по бутстрап-выборкам.

    Классическое А/Б-сравнение парное по вопросам: оба варианта прогоняются
    на одном наборе, и статистика снимается по вопросам. При бутстрепе вместо
    этого каждый вариант прогнан ``N`` раз, по разу на каждый ресемплированный
    пул; пары теперь — выборки, а не вопросы: ``d_k = variant_k - base_k`` на
    одной и той же выборке (пулы детерминированы общим зерном).

    Args:
        metrics: Метрики для сравнения.
        base: Агрегированный результат базового варианта.
        variant: Агрегированный результат нового варианта.
        seed: Зерно ГПСЧ для пермутаций.

    Returns:
        list[Verdict]: Вердикты по каждой метрике в порядке передачи.

    Raises:
        ValueError: Число выборок у вариантов не совпадает.
    """
    base_samples = base.meta.get("bootstrap", {}).get("samples")
    variant_samples = variant.meta.get("bootstrap", {}).get("samples")
    if base_samples is None or variant_samples is None:
        return compare_all(metrics, base, variant, seed=seed)
    if len(base_samples) != len(variant_samples):
        raise ValueError("число бутстрап-выборок у вариантов разное")

    verdicts: list[Verdict] = []
    for metric in metrics:
        base_values = [s.get(metric.key) for s in base_samples]
        variant_values = [s.get(metric.key) for s in variant_samples]
        pairs = [
            (b, v)
            for b, v in zip(base_values, variant_values, strict=True)
            if b is not None and v is not None
        ]
        if not pairs:
            verdicts.append(
                Verdict(
                    metric=metric.key,
                    test="bootstrap",
                    p_value=1.0,
                    base_value=0.0,
                    variant_value=0.0,
                    direction="no_diff",
                )
            )
            continue
        diffs = [v - b for b, v in pairs]
        base_value = sum(b for b, _ in pairs) / len(pairs)
        variant_value = sum(v for _, v in pairs) / len(pairs)
        observed = variant_value - base_value
        p_value = _paired_sample_p_value(diffs, seed)
        verdicts.append(
            Verdict(
                metric=metric.key,
                test="bootstrap",
                p_value=p_value,
                base_value=base_value,
                variant_value=variant_value,
                direction=_direction(metric, observed, p_value),
            )
        )
    return verdicts
