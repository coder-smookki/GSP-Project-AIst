"""A/B-сравнение вариантов на золотом наборе вопросов.

Один вариант — текущее решение с пустыми оверрайдами, другой — с
переопределёнными настройками (модель, реранкер, top_k). Сборка варианта
описана в :mod:`eval.variants`: каждый вариант получает собственные настройки
и строит поиск и агента из них, поэтому цифры сопоставимы — именно на этом
держится честность сравнения.

Файл вариантов — JSON со списком в ключе ``variants``::

    {
      "variants": [
        {"name": "base"},
        {"name": "top6", "description": "top_k=6",
         "settings": {"retrieval_top_k": 6}},
        {"name": "no-rerank", "description": "без реранкера",
         "settings": {"rerank_enabled": false}}
      ]
    }

Первый вариант считается базовым: к нему прибавляются вердикты A/B для
остальных (парный пермутационный тест, :mod:`eval.stats`). Запуск::

    uv run python eval/compare.py --variants variants.json --with-answers
    uv run python eval/compare.py --variants variants.json \
        --limit 10 --output report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# При прямом запуске (`uv run python eval/compare.py`) в sys.path попадает
# каталог eval, а не корень репозитория, и импорт `eval` и `infrastructure`
# падает. Полагаться на редактируемую установку нельзя: она отваливается
# при пересборке окружения.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.cache import ResultCache, build_entry, cache_key, file_hash, ab_cache_key, build_ab_entry
from eval.config import RunConfig, Variant
from eval.golden import DEFAULT_GOLDEN
from eval.metrics import BaseMetric, default_metrics
from eval.result import RunResult, Verdict
from eval.runners import GoldenSetRunner
from eval.stats import compare_bootstrapped
from eval.variants import variant_environment

#: Каталог кэша результатов по умолчанию.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

logger = logging.getLogger("compare")


def load_variants(path: str | Path) -> list[Variant]:
    """Читает файл вариантов.

    Args:
        path: Путь к JSON-файлу со списком вариантов в ключе ``variants``.

    Returns:
        list[Variant]: Варианты в порядке файла.

    Raises:
        ValueError: Файл пуст, имена неуникальны или поля невалидны.
        OSError: Файл не найден или не читается.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("variants", data) if isinstance(data, dict) else data
    variants = [Variant(**item) for item in raw]
    if not variants:
        raise ValueError(f"в файле {path} нет вариантов")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("имена вариантов должны быть уникальны")
    return variants


async def run_matrix(
    variants: list[Variant],
    config: RunConfig,
    metrics: list[BaseMetric],
    cache: ResultCache,
    force: bool = False,
    refresh_base: bool = False,
) -> list[RunResult]:
    """Прогоняет варианты, переиспользуя кэш.

    При ``config.bootstraps > 1`` каждый вариант дополнительно прогоняется на
    ``bootstraps`` ресемплированных пулах вопросов (бутстрап с возвратом,
    :meth:`eval.golden.GoldenSet.resample`). Состав пула детерминирован
    общим зерном и индексом выборки, поэтому base и варианты на одной выборке
    видят одинаковые вопросы — сравнение остаётся парным (по выборкам, а не
    по вопросам). Итоговые метрики варианта — среднее по выборкам, разброс
    складывается в ``meta["bootstrap"]``.

    Args:
        variants: Варианты для прогона.
        config: Параметры прогона.
        metrics: Метрики замера.
        cache: Хранилище результатов.
        force: Игнорировать кэш для всех вариантов.
        refresh_base: Переснять только первый вариант.

    Returns:
        list[RunResult]: Результаты в порядке вариантов (по одному на
        вариант, при бутстрепе — агрегированные по выборкам).
    """
    from eval.variants import build_variant, variant_environment

    runner = GoldenSetRunner(build=build_variant, metrics=metrics)
    dataset = file_hash(config.golden)
    samples = max(1, config.bootstraps)

    results: list[RunResult] = []
    for position, variant in enumerate(variants):
        environment = variant_environment(variant)
        per_sample: list[RunResult] = []
        for sample_index in range(samples):
            sample_config = (
                replace(config, sample_index=sample_index)
                if samples > 1
                else config
            )
            key = cache_key(variant, dataset, metrics, environment, sample_config)
            cached = None
            if not force and not (refresh_base and position == 0):
                cached = cache.load(key)
            if cached is not None:
                logger.info("вариант %s: взято из кэша (%s)", variant.name, key)
                per_sample.append(cached)
                continue
            logger.info(
                "прогон варианта %s (выборка %d/%d)",
                variant.name,
                sample_index + 1,
                samples,
            )
            result = await runner.run(variant, sample_config)
            cache.save(
                build_entry(
                    key, variant, result, environment, dataset, metrics, sample_config
                )
            )
            per_sample.append(result)
        results.append(_aggregate_bootstrap(variant, per_sample))
    return results


def _aggregate_bootstrap(variant: Variant, runs: list[RunResult]) -> RunResult:
    """Агрегирует прогоны варианта по бутстрап-выборкам.

    Args:
        variant: Вариант (для совместимости имени результата).
        runs: Результаты прогонов по одной выборке на каждый.

    Returns:
        RunResult: При одной выборке — исходный результат; иначе метрики
        становятся средним по выборкам, разброс — в ``meta["bootstrap"]``.
    """
    if len(runs) == 1:
        return runs[0]

    keys = {key for run in runs for key in run.metrics}
    metrics: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        values = [run.metrics.get(key) for run in runs]
        values = [value for value in values if value is not None]
        if not values:
            continue
        metrics[key] = statistics.mean(values)
        std[key] = statistics.stdev(values) if len(values) > 1 else 0.0

    last = runs[-1]
    meta = {
        **last.meta,
        "bootstrap": {
            "n": len(runs),
            "seed": last.meta.get("sample_index", 0),
            "samples": [run.metrics for run in runs],
            "std": std,
        },
    }
    return RunResult(
        variant=last.variant,
        environment=last.environment,
        metrics=metrics,
        outcomes=last.outcomes,
        created_at=last.created_at,
        meta=meta,
    )


def print_per_variant(results: list[RunResult], metrics: list[BaseMetric]) -> None:
    """Печатает сводку метрик по каждому варианту.

    Args:
        results: Результаты прогонов (по одному на вариант).
        metrics: Метрики в порядке вывода.
    """
    for result in results:
        print(f"\n{'=' * 72}\n{result.variant}\n{'=' * 72}")
        std = result.meta.get("bootstrap", {}).get("std", {})
        for metric in metrics:
            value = result.metrics.get(metric.key)
            if value is None:
                continue
            spread = std.get(metric.key)
            suffix = f" ± {spread:.3f}" if spread else ""
            print(f"  {metric.display_name:<18} {value:.3f}{suffix}")


def print_table(results: list[RunResult], metrics: list[BaseMetric]) -> None:
    """Печатает сравнительную таблицу: метрики по строкам, варианты по колонкам.

    Лучшее значение по строке помечается звёздочкой. Метрики без направления
    (длина ответа) не помечаются.

    Args:
        results: Результаты прогонов.
        metrics: Метрики в порядке вывода.
    """
    variants = [result.variant for result in results]
    width = max(len(v) for v in variants)
    print("\n" + "=" * 72)
    print("СРАВНЕНИЕ")
    print("=" * 72)
    header = f"{'метрика':<18}" + "".join(f"{v:>{width + 4}}" for v in variants)
    print(header)

    for metric in metrics:
        values = [result.metrics.get(metric.key) for result in results]
        if all(v is None for v in values):
            continue
        best = None
        if metric.higher_is_better is not None:
            comparable = [v for v in values if v is not None]
            if comparable:
                best = max(comparable) if metric.higher_is_better else min(comparable)
        row = f"{metric.display_name:<18}"
        for result in results:
            value = result.metrics.get(metric.key)
            if value is None:
                text = "-"
            else:
                spread = result.meta.get("bootstrap", {}).get("std", {}).get(
                    metric.key
                )
                text = f"{value:.3f}±{spread:.3f}" if spread else f"{value:.3f}"
            if best is not None and value == best:
                text += "*"
            row += f"{text:>{width + 4}}"
        print(row)


def print_verdicts(
    base: RunResult, results: list[RunResult], metrics: list[BaseMetric]
) -> None:
    """Печатает вердикты A/B по каждой метрике.

    Args:
        base: Результат базового варианта.
        results: Результаты остальных вариантов.
        metrics: Метрики для сравнения.
    """
    print("\n" + "=" * 72)
    print("ВЕРДИКТЫ A/B (относительно base)")
    print("=" * 72)
    for result in results:
        print(f"\n{result.variant}:")
        for verdict in compare_bootstrapped(metrics, base, result):
            arrow = {"better": "лучше", "worse": "хуже", "no_diff": "без разницы"}
            print(
                f"  {verdict.metric:<18} {verdict.base_value:.3f} -> "
                f"{verdict.variant_value:.3f}  {arrow[verdict.direction]} "
                f"(p={verdict.p_value:.3f})"
            )


def _build_payload(
    variants: list[Variant],
    results: list[RunResult],
    metrics: list[BaseMetric],
    verdicts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Собирает полный отчёт для вывода в JSON.

    Args:
        variants: Варианты.
        results: Результаты прогонов.
        metrics: Метрики сравнения.
        verdicts: Вердикты в виде словарей.

    Returns:
        dict[str, Any]: Словарь отчёта.
    """
    return {
        "variants": [asdict(variant) for variant in variants],
        "results": [result.to_dict() for result in results],
        "verdicts": verdicts,
        "metrics": [metric.key for metric in metrics],
    }


async def main_async(
    args: argparse.Namespace,
    variants: list[Variant],
    metrics: list[BaseMetric],
) -> int:
    """Выполняет матрицу прогонов и сравнение.

    Args:
        args: Разобранные аргументы командной строки.
        variants: Варианты для прогона.
        metrics: Метрики замера и сравнения.

    Returns:
        int: Код возврата процесса.
    """
    from infrastructure.config.settings import get_settings
    from infrastructure.retrieval.factory import shutdown

    settings = get_settings()
    if settings.resolved_retrieval_provider() == "mock":
        logger.error(
            "замер идёт на заглушке поиска: задайте RETRIEVAL_PROVIDER=qdrant, "
            "иначе цифры ничего не значат"
        )
        return 2

    config = RunConfig(
        golden=args.golden,
        limit=args.limit,
        with_answers=args.with_answers,
        bootstraps=args.bootstraps,
        seed=args.seed,
        graph=args.graph,
    )
    cache = ResultCache(args.cache_dir)

    try:
        results = await run_matrix(
            variants,
            config,
            metrics,
            cache,
            force=args.force,
            refresh_base=args.refresh_base,
        )
    finally:
        await shutdown()

    print_per_variant(results, metrics)
    print_table(results, metrics)
    print_verdicts(results[0], results[1:], metrics)

    # Сохраняем A/B вердикты в кэш
    base_variant = variants[0]
    dataset = file_hash(config.golden)
    for result in results[1:]:
        variant = next(v for v in variants if v.name == result.variant)
        environment = variant_environment(variant)
        verdicts = compare_bootstrapped(metrics, results[0], result)
        ab_key = ab_cache_key(base_variant, variant, dataset, metrics, environment, config)
        cache.save_ab(
            build_ab_entry(ab_key, base_variant, variant, verdicts, environment, dataset, metrics, config)
        )
        logger.info("A/B вердикты сохранены в кэш: %s", ab_key)

    if args.output:
        verdicts: list[dict[str, Any]] = []
        for result in results[1:]:
            for verdict in compare_bootstrapped(metrics, results[0], result):
                verdicts.append({"variant": result.variant, **asdict(verdict)})
        payload = _build_payload(variants, results, metrics, verdicts)
        await asyncio.to_thread(
            Path(args.output).write_text,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("отчёт записан в %s", args.output)

    return 0


def main() -> int:
    """Точка входа.

    Returns:
        int: Код возврата процесса.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        required=True,
        help="путь к JSON-файлу с вариантами",
    )
    parser.add_argument(
        "--golden", default=str(DEFAULT_GOLDEN), help="путь к золотому набору"
    )
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="дополнительно генерировать ответы и проверять отказ по тексту",
    )
    parser.add_argument("--limit", type=int, help="взять только первые N вопросов")
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=1,
        help="сколько независимых прогонов на ресемплированных пулах "
        "вопросов (бутстрап с возвратом)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="зерно ресемплинга пула: один пул для всех вариантов",
    )
    parser.add_argument(
        "--graph",
        default="base",
        help="тег версии графа агента (base, rd-v2, ...): отдельная секция "
        "в отчёте и свой кэш",
    )
    parser.add_argument("--output", help="записать полный отчёт в файл")
    parser.add_argument(
        "--cache-dir",
        default=str(RESULTS_DIR),
        help="каталог кэша результатов (по умолчанию eval/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="переснять все варианты, игнорируя кэш",
    )
    parser.add_argument(
        "--refresh-base",
        action="store_true",
        help="переснять только базовый вариант, остальные брать из кэша",
    )
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        variants = load_variants(args.variants)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 2

    return asyncio.run(main_async(args, variants, default_metrics()))


if __name__ == "__main__":
    sys.exit(main())
