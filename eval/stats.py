"""Статистическое сравнение вариантов: парный пермутационный тест.

Метрики — функции от итогов по вопросам (:class:`eval.result.Outcome`),
поэтому сравнение двух вариантов, прогнанных на одном золотом наборе, — это
сравнение двух эмпирических распределений. Вопросы совпадают и идут в одном
порядке, поэтому применяется парный тест: для каждой пары итогов решение
«поменять местами» принимается честной монетой, и разность метрики
пересчитывается на переставленных группах. p-value — доля перестановок, где
разность не слабее наблюдаемой.

Классические тесты (Манна-Уитни, хи-квадрат) потянули бы scipy; перестановочный
тест на ``random`` обходится без тяжёлых зависимостей и точен на небольших
наборах, где асимптотические приближения врут.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from eval.metrics import BaseMetric
from eval.result import RunResult, Verdict

#: Сколько перестановок сделать.
PERMUTATIONS = 5000

#: Порог значимости: ``p < 0.05`` даёт направленный вердикт.
SIGNIFICANCE = 0.05


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


def compare_metric(
    metric: BaseMetric,
    base: RunResult,
    variant: RunResult,
    seed: int = 42,
    permutations: int = PERMUTATIONS,
) -> Verdict:
    """Сравнивает одну метрику двух вариантов.

    Args:
        metric: Метрика для сравнения.
        base: Результат базового варианта.
        variant: Результат нового варианта.
        seed: Зерно ГПСЧ — вывод детерминирован.
        permutations: Сколько перестановок сделать.

    Returns:
        Verdict: Значения метрики, p-value и направление.

    Raises:
        ValueError: Итоги вариантов различаются по числу вопросов — парный
            тест требует, чтобы варианты были прогнаны на одном наборе.
    """
    base_outcomes = base.outcomes
    variant_outcomes = variant.outcomes
    if len(base_outcomes) != len(variant_outcomes):
        raise ValueError("варианты прогнаны на разном числе вопросов")

    observed = metric.evaluate(variant_outcomes) - metric.evaluate(base_outcomes)

    rng = random.Random(_seed_for(seed, metric.key))
    n = len(base_outcomes)
    extremes = 0
    for _ in range(permutations):
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

    # +1: наблюдаемая разность — одна из возможных перестановок, её учитываем,
    # иначе при сильном разделении p-value обнулился бы.
    p_value = (extremes + 1) / (permutations + 1)

    return Verdict(
        metric=metric.key,
        test="permutation (paired)",
        p_value=p_value,
        base_value=metric.evaluate(base_outcomes),
        variant_value=metric.evaluate(variant_outcomes),
        direction=_direction(metric, observed, p_value),
    )


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
