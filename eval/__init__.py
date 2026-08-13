"""Бенчмаркинг системы: качество поиска, операционный замер, A/B-тесты.

Пакет выносит измерение метрик из утилитных скриптов в отдельный модуль:
золотой набор и сопоставление цитат — :mod:`eval.golden`, метрики —
:mod:`eval.metrics`, модель данных — :mod:`eval.config` и :mod:`eval.result`,
контракты — :mod:`eval.contracts`.
"""

from eval.cache import ResultCache, cache_key
from eval.config import Environment, RunConfig, Variant, snapshot_environment
from eval.golden import GoldenSet
from eval.metrics import default_metrics
from eval.result import Outcome, RunResult, Verdict
from eval.runners import BuildResult, GoldenSetRunner, OperationalRunner
from eval.stats import compare_all, compare_metric
from eval.variants import build_variant

__all__ = [
    "BuildResult",
    "Environment",
    "GoldenSet",
    "GoldenSetRunner",
    "OperationalRunner",
    "Outcome",
    "ResultCache",
    "RunConfig",
    "RunResult",
    "Variant",
    "Verdict",
    "build_variant",
    "cache_key",
    "compare_all",
    "compare_metric",
    "default_metrics",
    "snapshot_environment",
]
