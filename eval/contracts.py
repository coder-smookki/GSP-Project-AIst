"""Контракты бенчмаркинга: интерфейсы, на которые полагаются потребители.

SOLID здесь: модули зависят от этих абстракций (DIP), каждая отвечает за одно
(ISP), новые метрики/тесты/раннеры добавляются без правки существующих (OCP),
а реализации взаимозаменяемы (LSP).
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from eval.config import RunConfig, Variant
from eval.result import Outcome, RunResult, Verdict


@runtime_checkable
class Metric(Protocol):
    """Метрика: сводит итоги по вопросам к одному числу."""

    key: str
    display_name: str
    kind: Literal["categorical", "continuous"]

    def evaluate(self, outcomes: list[Outcome]) -> float:
        """Считает значение метрики по итогам вопросов.

        Args:
            outcomes: Итоги по каждому вопросу.

        Returns:
            float: Значение метрики.
        """
        ...


@runtime_checkable
class Runner(Protocol):
    """Раннер: прогоняет вариант и собирает результат."""

    name: str

    async def run(self, variant: Variant, config: RunConfig) -> RunResult:
        """Прогоняет вариант.

        Args:
            variant: Вариант системы.
            config: Параметры прогона.

        Returns:
            RunResult: Результат прогона.
        """
        ...


@runtime_checkable
class StatTest(Protocol):
    """Статистический тест: сравнивает base и новый вариант."""

    name: str

    def compare(self, base: RunResult, variant: RunResult) -> list[Verdict]:
        """Сравнивает результаты по всем метрикам.

        Args:
            base: Результат base-варианта.
            variant: Результат нового варианта.

        Returns:
            list[Verdict]: Вердикт по каждой метрике.
        """
        ...


@runtime_checkable
class Cache(Protocol):
    """Кэш результатов: переиспользует уже снятые прогоны."""

    def get(self, key: str) -> RunResult | None:
        """Возвращает закэшированный результат или ``None``."""
        ...

    def put(self, key: str, result: RunResult) -> None:
        """Сохраняет результат под ключом."""
        ...


@runtime_checkable
class Reporter(Protocol):
    """Репортёр: превращает результаты в документ (BENCHMARK.md)."""

    def render(self, results: list[RunResult], verdicts: list[Verdict]) -> str:
        """Рендерит документ по результатам и вердиктам.

        Args:
            results: Результаты прогонов.
            verdicts: Вердикты A/B (может быть пусто).

        Returns:
            str: Текст документа.
        """
        ...
