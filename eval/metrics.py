"""Метрики замера качества: сводные числа по выборке замеров.

Пока здесь только квантили времени; по мере расширения бенчмаркинга сюда
придут остальные метрики (hit@k, MRR, доля отказов) как единый реестр.
"""

from __future__ import annotations


def percentile(values: list[float], share: float) -> float:
    """Возвращает значение квантиля.

    Args:
        values: Измерения.
        share: Доля от 0 до 1.

    Returns:
        float: Значение квантиля; ``0.0`` для пустого списка.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(share * (len(ordered) - 1)))
    return ordered[index]
