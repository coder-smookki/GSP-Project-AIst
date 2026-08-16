"""Тюнинг параметров перебором по сетке (grid search).

Запускает вариант на каждом пересечении значений параметров из спеки, собирает
метрики через общий кэш и печатает таблицу, ранжированную по целевой метрике
с указанным направлением. Это «ручной» шаг поиска оптимума перед точечным A/B:
низкая размерность сетки решается перебором, дорогие измерения — кэшем.

    uv run python eval/tune.py --params '{
        "retrieval_top_k": [3, 5, 8],
        "rerank_candidates": [5, 10, 15]
    }' --objective hit@1 --direction max

То же через единый CLI::

    uv run eval tune --params '{"retrieval_top_k": [3, 5, 8]}' --objective mrr

Параметры — реальные поля настроек (``Settings``): модель, ``retrieval_top_k``,
``rerank_candidates``, ``rerank_enabled`` и т.д. Имя варианта собирается из
значений сетки, поэтому повторный запуск (например, с бутстрапом) находит те
же точки в кэше.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

# При прямом запуске (`uv run python eval/tune.py`) в sys.path попадает каталог
# eval, а не корень репозитория — поднимаемся на уровень выше, как делают
# остальные точки входа eval.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.cache import ResultCache
from eval.config import RunConfig, Variant
from eval.metrics import default_metrics
from eval.result import RunResult

logger = logging.getLogger("tune")

#: Максимум точек сетки по умолчанию — защита от случайной сетки 10^6.
DEFAULT_MAX_POINTS = 100


def build_grid(params: dict[str, list]) -> list[Variant]:
    """Строит декартову сетку вариантов из спеки параметров.

    Args:
        params: Имя параметра → список значений (``{"retrieval_top_k": [3,5]}``).

    Returns:
        list[Variant]: Вариант на каждое пересечение; имя уникально и читаемо
        (``retrieval_top_k=5;rerank_candidates=10``).

    Raises:
        ValueError: Спека пуста, значение — не список или пустой список.
    """
    if not params:
        raise ValueError("спека параметров пуста: задайте --params JSON-объектом")
    for name, values in params.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"параметр {name!r}: нужен непустой список значений")

    grid: list[Variant] = []
    for combo in itertools.product(*params.values()):
        settings = dict(zip(params.keys(), combo, strict=True))
        name = ";".join(f"{key}={value}" for key, value in settings.items())
        grid.append(Variant(name=name, settings=settings))
    return grid


def rank_results(
    variants: list[Variant],
    results: list[RunResult],
    objective: str,
    direction: str,
) -> list[tuple[Variant, RunResult, float, float]]:
    """Сортирует результаты по целевой метрике.

    Args:
        variants: Варианты в том же порядке, что и результаты.
        results: Результаты прогонов (по одному на вариант).
        objective: Ключ целевой метрики (``hit@1``, ``mrr``, ...).
        direction: ``max`` или ``min``.

    Returns:
        list[tuple]: ``(вариант, результат, значение, std)`` от лучшего
        к худшему.

    Raises:
        ValueError: Целевая метрика отсутствует в результатах.
    """
    scored: list[tuple[Variant, RunResult, float, float]] = []
    for variant, result in zip(variants, results, strict=True):
        value = result.metrics.get(objective)
        if value is None:
            raise ValueError(
                f"метрика {objective!r} не рассчитана для варианта {variant.name}"
            )
        std = result.meta.get("bootstrap", {}).get("std", {}).get(objective, 0.0)
        scored.append((variant, result, value, std))
    return sorted(scored, key=lambda item: item[2], reverse=direction == "max")


def _print_ranked(
    ranked: list[tuple[Variant, RunResult, float, float]], objective: str
) -> None:
    """Печатает ранжированную таблицу сетки.

    Args:
        ranked: Результаты от лучшего к худшему.
        objective: Ключ целевой метрики.
    """
    width = max(len(variant.name) for variant, _, _, _ in ranked)
    print("\n" + "=" * 72)
    print(f"ТЮНИНГ ПО СЕТКЕ (цель: {objective})")
    print("=" * 72)
    for position, (variant, _, value, std) in enumerate(ranked, start=1):
        spread = f" ± {std:.3f}" if std else ""
        print(f"{position:3d}. {variant.name:<{width}}  {value:.4f}{spread}")


def _write_output(
    args: argparse.Namespace,
    params: dict[str, list],
    ranked: list[tuple[Variant, RunResult, float, float]],
) -> None:
    """Пишет полный отчёт тюнинга в JSON.

    Args:
        args: Разобранные аргументы.
        params: Спека параметров сетки.
        ranked: Ранжированные результаты.
    """
    points = [
        {
            "name": variant.name,
            "settings": variant.settings,
            "metrics": result.metrics,
            "objective": value,
            "std": std,
        }
        for variant, result, value, std in ranked
    ]
    payload: dict[str, Any] = {
        "params": params,
        "objective": args.objective,
        "direction": args.direction,
        "bootstraps": args.bootstraps,
        "seed": args.seed,
        "graph": args.graph,
        "points": points,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("отчёт тюнинга записан в %s", args.output)


async def main_async(args: argparse.Namespace) -> int:
    """Выполняет сеточный перебор.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        int: Код возврата процесса.
    """
    from eval.compare import run_matrix
    from infrastructure.config.settings import get_settings
    from infrastructure.retrieval.factory import shutdown

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        logger.error("--params: неверный JSON: %s", exc)
        return 2
    if not isinstance(params, dict):
        logger.error("--params: ожидается JSON-объект {параметр: [значения]}")
        return 2

    metrics = default_metrics()
    known = {metric.key for metric in metrics}
    if args.objective not in known:
        logger.error(
            "--objective: неизвестная метрика %r; доступные: %s",
            args.objective,
            ", ".join(sorted(known)),
        )
        return 2

    try:
        variants = build_grid(params)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    if len(variants) > args.max_points:
        logger.error(
            "в сетке %d вариантов, а максимум %d: сузьте сетку или поднимите "
            "--max-points",
            len(variants),
            args.max_points,
        )
        return 2
    logger.info("сетка: %d вариантов", len(variants))

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
            variants, config, metrics, cache, force=args.force
        )
    finally:
        await shutdown()

    try:
        ranked = rank_results(variants, results, args.objective, args.direction)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    _print_ranked(ranked, args.objective)

    if args.output:
        _write_output(args, params, ranked)

    return 0


def main() -> int:
    """Точка входа (для прямого ``python eval/tune.py``).

    Returns:
        int: Код возврата процесса.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        required=True,
        help='JSON-спека сетки: \'{"retrieval_top_k": [3, 5, 8]}\'',
    )
    parser.add_argument(
        "--objective",
        default="hit@1",
        help="ключ целевой метрики (по умолчанию hit@1)",
    )
    parser.add_argument(
        "--direction", default="max", choices=("min", "max"), help="направление"
    )
    parser.add_argument(
        "--golden", default="tests/fixtures/golden/questions.json",
        help="путь к золотому набору",
    )
    parser.add_argument("--limit", type=int, help="взять только первые N вопросов")
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="дополнительно генерировать ответы и проверять отказ по тексту",
    )
    parser.add_argument(
        "--bootstraps", type=int, default=1,
        help="прогонов на ресемплированных пулах (бутстрап)",
    )
    parser.add_argument("--seed", type=int, default=42, help="зерно ресемплинга")
    parser.add_argument(
        "--graph",
        default="base",
        help="тег версии графа агента (base, rd-v2, ...): отдельная секция "
        "в отчёте и свой кэш",
    )
    parser.add_argument(
        "--cache-dir", help="каталог кэша результатов (по умолчанию eval/results)"
    )
    parser.add_argument(
        "--force", action="store_true", help="переснять все точки, игнорируя кэш"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="максимум точек сетки (защита от гигантского перебора)",
    )
    parser.add_argument("--output", help="записать полный отчёт в файл")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args()

    if args.cache_dir is None:
        from eval.compare import RESULTS_DIR

        args.cache_dir = str(RESULTS_DIR)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())