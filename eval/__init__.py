"""Бенчмаркинг системы: качество поиска, операционный замер, A/B-тесты.

Пакет выносит измерение метрик из утилитных скриптов в отдельный модуль:
золотой набор и сопоставление цитат — :mod:`eval.golden`, метрики —
:mod:`eval.metrics`, модель данных — :mod:`eval.config` и :mod:`eval.result`,
контракты — :mod:`eval.contracts`.
"""

from eval.config import Environment, RunConfig, Variant, snapshot_environment
from eval.golden import GoldenSet
from eval.metrics import default_metrics
from eval.result import Outcome, RunResult, Verdict

__all__ = [
    "Environment",
    "GoldenSet",
    "Outcome",
    "RunConfig",
    "RunResult",
    "Variant",
    "Verdict",
    "default_metrics",
    "snapshot_environment",
]
