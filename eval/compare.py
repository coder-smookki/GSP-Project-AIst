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
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# При прямом запуске (`uv run python eval/compare.py`) в sys.path попадает
# каталог eval, а не корень репозитория, и импорт `eval` и `infrastructure`
# падает. Полагаться на редактируемую установку нельзя: она отваливается
# при пересборке окружения.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.config import RunConfig, Variant
from eval.golden import DEFAULT_GOLDEN
from eval.metrics import BaseMetric, default_metrics
from eval.result import RunResult
from eval.runners import GoldenSetRunner
from eval.stats import compare_all

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


def print_per_variant(results: list[RunResult], metrics: list[BaseMetric]) -> None:
    """Печатает сводку метрик по каждому варианту.

    Args:
        results: Результаты прогонов (по одному на вариант).
        metrics: Метрики в порядке вывода.
    """
    for result in results:
        print(f"\n{'=' * 72}\n{result.variant}\n{'=' * 72}")
        for metric in metrics:
            value = result.metrics.get(metric.key)
            if value is not None:
                print(f"  {metric.display_name:<18} {value:.3f}")


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
        for value in values:
            text = f"{value:.3f}" if value is not None else "-"
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
        for verdict in compare_all(metrics, base, result):
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
    from eval.variants import build_variant
    from infrastructure.config.settings import get_settings
    from infrastructure.retrieval.factory import shutdown

    settings = get_settings()
    if settings.resolved_retrieval_provider() == "mock":
        logger.error(
            "замер идёт на заглушке поиска: задайте RETRIEVAL_PROVIDER=qdrant, "
            "иначе цифры ничего не значат"
        )
        return 2

    runner = GoldenSetRunner(build=build_variant, metrics=metrics)
    config = RunConfig(
        golden=args.golden, limit=args.limit, with_answers=args.with_answers
    )

    results: list[RunResult] = []
    try:
        for variant in variants:
            logger.info("прогон варианта %s", variant.name)
            results.append(await runner.run(variant, config))
    finally:
        await shutdown()

    print_per_variant(results, metrics)
    print_table(results, metrics)
    print_verdicts(results[0], results[1:], metrics)

    if args.output:
        verdicts: list[dict[str, Any]] = []
        for result in results[1:]:
            for verdict in compare_all(metrics, results[0], result):
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
    parser.add_argument("--output", help="записать полный отчёт в файл")
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
