"""Метрики бенчмаркинга: сводные числа по итогам вопросов.

Каждая метрика — самостоятельный класс (SRP), реестр расширяется добавлением
нового класса без правки существующих (OCP). Метрики делятся по типу значения:
категориальные (доля 0/1) и непрерывные; тип выбирает статистический тест в
A/B-сравнении (см. :mod:`eval.stats`).

Метрики считаются по разным подмножествам итогов:
* hit@k, MRR, не-найдено — только по отвечаемым вопросам (``answerable``);
* честный отказ — только по вопросам без ответа в базе;
* время ответа и длина — по всем вопросам.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eval.result import Outcome

MetricKind = Literal["categorical", "continuous"]


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


def answerable_outcomes(outcomes: list[Outcome]) -> list[Outcome]:
    """Отвечаемые вопросы (есть эталонная цитата).

    Args:
        outcomes: Итоги по вопросам.

    Returns:
        list[Outcome]: Итоги только по отвечаемым вопросам.
    """
    return [o for o in outcomes if o.answerable]


def unanswerable_outcomes(outcomes: list[Outcome]) -> list[Outcome]:
    """Вопросы, ответа на которые в базе нет.

    Args:
        outcomes: Итоги по вопросам.

    Returns:
        list[Outcome]: Итоги только по вопросам без ответа.
    """
    return [o for o in outcomes if not o.answerable]


class BaseMetric:
    """Общий каркас метрики: вычисление оставляет наследникам.

    Attributes:
        higher_is_better: В какую сторону улучшение: ``True`` — больше лучше,
            ``False`` — меньше лучше, ``None`` — направления нет (A/B не
            выносит вердикт).
    """

    kind: MetricKind
    higher_is_better: bool | None = True

    @property
    def key(self) -> str:
        """Ключ метрики для словаря результатов."""
        raise NotImplementedError

    @property
    def display_name(self) -> str:
        """Отображаемое название в отчёте."""
        raise NotImplementedError

    def evaluate(self, outcomes: list[Outcome]) -> float:
        """Считает значение метрики по итогам вопросов.

        Returns:
            float: Значение метрики.
        """
        raise NotImplementedError

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        """Вопросы, по которым метрика определена.

        Статистические тесты сравнивают метрику только по этим вопросам
        (например, отказ — только по вопросам без ответа).

        Args:
            outcomes: Итоги по вопросам.

        Returns:
            list[Outcome]: Подмножество итогов, которое считает метрика.
        """
        return outcomes

    def per_question(self, outcomes: list[Outcome]) -> list[float]:
        """Значение метрики по каждому вопросу.

        Args:
            outcomes: Итоги по вопросам.

        Returns:
            list[float]: Значение метрики на каждом вопросе по отдельности.
        """
        return [self.evaluate([outcome]) for outcome in outcomes]


@dataclass(frozen=True)
class HitAtK(BaseMetric):
    """Доля отвечаемых вопросов, где ответ в первых ``k`` фрагментах."""

    k: int
    kind: MetricKind = "categorical"

    @property
    def key(self) -> str:
        return f"hit@{self.k}"

    @property
    def display_name(self) -> str:
        return f"Hit@{self.k}"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        relevant = answerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        hits = sum(1 for o in relevant if o.rank is not None and o.rank <= self.k)
        return hits / len(relevant)

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        return answerable_outcomes(outcomes)


@dataclass(frozen=True)
class MRR(BaseMetric):
    """Средняя обратная позиция ответа (по отвечаемым вопросам)."""

    kind: MetricKind = "continuous"

    @property
    def key(self) -> str:
        return "mrr"

    @property
    def display_name(self) -> str:
        return "MRR"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        relevant = answerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        return sum(1.0 / o.rank for o in relevant if o.rank) / len(relevant)

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        return answerable_outcomes(outcomes)


@dataclass(frozen=True)
class NotFoundRate(BaseMetric):
    """Доля отвечаемых вопросов, где ответ не дошёл до контекста вовсе."""

    kind: MetricKind = "categorical"
    higher_is_better: bool | None = False

    @property
    def key(self) -> str:
        return "not_found"

    @property
    def display_name(self) -> str:
        return "Не найдено"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        relevant = answerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        return sum(1 for o in relevant if o.rank is None) / len(relevant)

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        return answerable_outcomes(outcomes)


@dataclass(frozen=True)
class RefusalRate(BaseMetric):
    """Доля вопросов без ответа, на которые система честно отказала."""

    kind: MetricKind = "categorical"

    @property
    def key(self) -> str:
        return "refusal_rate"

    @property
    def display_name(self) -> str:
        return "Честный отказ"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        relevant = unanswerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        return sum(1 for o in relevant if o.refused) / len(relevant)

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        return unanswerable_outcomes(outcomes)


@dataclass(frozen=True)
class LatencyMedian(BaseMetric):
    """Медианное время поиска по всем вопросам."""

    kind: MetricKind = "continuous"
    higher_is_better: bool | None = False

    @property
    def key(self) -> str:
        return "latency_median"

    @property
    def display_name(self) -> str:
        return "Время, медиана"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        return percentile([o.seconds for o in outcomes], 0.5)


@dataclass(frozen=True)
class LatencyP95(BaseMetric):
    """95-й процентиль времени поиска по всем вопросам."""

    kind: MetricKind = "continuous"
    higher_is_better: bool | None = False

    @property
    def key(self) -> str:
        return "latency_p95"

    @property
    def display_name(self) -> str:
        return "Время, p95"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        return percentile([o.seconds for o in outcomes], 0.95)


@dataclass(frozen=True)
class AnswerChars(BaseMetric):
    """Средняя длина ответа модели (по вопросам с ответом)."""

    kind: MetricKind = "continuous"
    higher_is_better: bool | None = None

    @property
    def key(self) -> str:
        return "answer_chars"

    @property
    def display_name(self) -> str:
        return "Ответ, символов"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        lengths = [len(o.answer) for o in outcomes if o.answer is not None]
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)


def default_metrics(k_values: tuple[int, ...] = (1, 3, 5)) -> list[BaseMetric]:
    """Стандартный набор метрик замера качества.

    Args:
        k_values: Какие ``hit@k`` включить.

    Returns:
        list[BaseMetric]: Метрики в порядке вывода в отчёте.
    """
    return [
        *[HitAtK(k) for k in k_values],
        MRR(),
        NotFoundRate(),
        RefusalRate(),
        LatencyMedian(),
        LatencyP95(),
        AnswerChars(),
    ]


def compute_metrics(
    metrics: list[BaseMetric], outcomes: list[Outcome]
) -> dict[str, float]:
    """Считает все метрики по итогам вопросов.

    Args:
        metrics: Метрики для расчёта.
        outcomes: Итоги по вопросам.

    Returns:
        dict[str, float]: Значение метрики по её ключу.
    """
    return {metric.key: metric.evaluate(outcomes) for metric in metrics}


@dataclass(frozen=True)
class GraphHitAtK(BaseMetric):
    """Доля отвечаемых вопросов, где ответ в первых ``k`` финальных фрагментах графа."""

    k: int
    kind: MetricKind = "categorical"

    @property
    def key(self) -> str:
        return f"graph_hit@{self.k}"

    @property
    def display_name(self) -> str:
        return f"Graph Hit@{self.k}"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        relevant = answerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        hits = sum(
            1 for o in relevant if o.graph_rank is not None and o.graph_rank <= self.k
        )
        return hits / len(relevant)

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        return answerable_outcomes(outcomes)


@dataclass(frozen=True)
class GraphMRR(BaseMetric):
    """Средняя обратная позиция ответа в финальных фрагментах графа."""

    kind: MetricKind = "continuous"

    @property
    def key(self) -> str:
        return "graph_mrr"

    @property
    def display_name(self) -> str:
        return "Graph MRR"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        relevant = answerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        return sum(1.0 / o.graph_rank for o in relevant if o.graph_rank) / len(relevant)

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        return answerable_outcomes(outcomes)


@dataclass(frozen=True)
class GraphNotFoundRate(BaseMetric):
    """Доля отвечаемых вопросов, где ответ не дошёл до финального контекста графа."""

    kind: MetricKind = "categorical"
    higher_is_better: bool | None = False

    @property
    def key(self) -> str:
        return "graph_not_found"

    @property
    def display_name(self) -> str:
        return "Graph Не найдено"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        relevant = answerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        return sum(1 for o in relevant if o.graph_rank is None) / len(relevant)

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        return answerable_outcomes(outcomes)


@dataclass(frozen=True)
class GraphRefusalRate(BaseMetric):
    """Доля вопросов без ответа, на которые агент честно отказал."""

    kind: MetricKind = "categorical"

    @property
    def key(self) -> str:
        return "graph_refusal_rate"

    @property
    def display_name(self) -> str:
        return "Graph Честный отказ"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        relevant = unanswerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        return sum(1 for o in relevant if o.refused) / len(relevant)

    def relevant(self, outcomes: list[Outcome]) -> list[Outcome]:
        return unanswerable_outcomes(outcomes)


@dataclass(frozen=True)
class AvgSearchCalls(BaseMetric):
    """Среднее число обращений к поиску на один запрос (по trace/state)."""

    kind: MetricKind = "continuous"
    higher_is_better: bool | None = False

    @property
    def key(self) -> str:
        return "avg_search_calls"

    @property
    def display_name(self) -> str:
        return "Поиск. вызовов/запрос"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        counts = [o.search_calls for o in outcomes if o.search_calls is not None]
        if not counts:
            return 0.0
        return sum(counts) / len(counts)


@dataclass(frozen=True)
class GraphLatencyMedian(BaseMetric):
    """Медианное время работы графа целиком."""

    kind: MetricKind = "continuous"
    higher_is_better: bool | None = False

    @property
    def key(self) -> str:
        return "graph_latency_median"

    @property
    def display_name(self) -> str:
        return "Graph время, медиана"

    def evaluate(self, outcomes: list[Outcome]) -> float:
        return percentile([o.seconds for o in outcomes], 0.5)


def graph_metrics(k_values: tuple[int, ...] = (1, 3, 5)) -> list[BaseMetric]:
    """Набор метрик для графа.
    
    Args:
        k_values: Какие ``hit@k`` включить.
        
    Returns:
        list[BaseMetric]: Метрики для графа.
    """
    return [
        *[GraphHitAtK(k) for k in k_values],
        GraphMRR(),
        GraphNotFoundRate(),
        GraphRefusalRate(),
        AvgSearchCalls(),
        GraphLatencyMedian(),
    ]
