"""Единая точка входа в бенчмаркинг: run, ab, bench, cache.

Оборачивает раздельные харнессы в одну команду, чтобы все замеры были
доступны из одного места::

    uv run eval run --with-answers              # качество на золотом наборе
    uv run eval run --variant top6 --settings '{"retrieval_top_k": 6}'
    uv run eval ab --variants variants.json     # A/B-сравнение вариантов
    uv run eval bench --repeats 3               # латенси и нагрузка
    uv run eval cache list | clear              # управление кэшем результатов
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# При запуске как `python eval/cli.py` в sys.path попадает каталог eval,
# а не корень репозитория, и импорт `eval` и `infrastructure` падает.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.cache import DEFAULT_DIR, ResultCache, _print_list, _print_ab_list, _print_list
from eval.config import RunConfig, Variant
from eval.golden import DEFAULT_GOLDEN, GoldenSetError
from eval.result import RunResult

logger = logging.getLogger("eval")

#: Куда report пишет документ по умолчанию.
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "docs" / "BENCHMARK.md"


def _mock_guard() -> bool:
    """Проверяет, что замер не идёт на заглушке поиска.

    Returns:
        bool: ``True``, если можно мерить.
    """
    from infrastructure.config.settings import get_settings

    if get_settings().resolved_retrieval_provider() == "mock":
        logger.error(
            "замер идёт на заглушке поиска: задайте RETRIEVAL_PROVIDER=qdrant, "
            "иначе цифры ничего не значат"
        )
        return False
    return True


def _cmd_run(args: argparse.Namespace) -> int:
    """Замер качества одного варианта на золотом наборе.

    Args:
        args: Разобранные аргументы.

    Returns:
        int: Код возврата процесса.
    """
    from eval.compare import run_matrix
    from eval.evaluate import report
    from eval.metrics import default_metrics
    from infrastructure.retrieval.factory import shutdown

    if not _mock_guard():
        return 2
    try:
        settings = json.loads(args.settings) if args.settings else {}
    except json.JSONDecodeError as exc:
        logger.error("--settings: неверный JSON: %s", exc)
        return 2

    variant = Variant(name=args.variant, settings=settings)
    config = RunConfig(
        golden=args.golden,
        limit=args.limit,
        with_answers=args.with_answers,
        bootstraps=args.bootstraps,
        seed=args.seed,
        graph=args.graph,
    )

    async def run() -> RunResult:
        cache = ResultCache(args.cache_dir)
        try:
            results = await run_matrix(
                [variant], config, default_metrics(), cache, force=args.force
            )
            return results[0]
        finally:
            await shutdown()

    try:
        result = asyncio.run(run())
    except GoldenSetError as exc:
        logger.error("%s", exc)
        return 2

    answerable = [o for o in result.outcomes if o.answerable]
    unanswerable = [o for o in result.outcomes if not o.answerable]
    ok = report(answerable, unanswerable)

    if args.output:
        payload = {
            "variant": result.variant,
            "environment": result.environment,
            "metrics": result.metrics,
            "answerable": [vars(o) for o in answerable],
            "unanswerable": [vars(o) for o in unanswerable],
            "created_at": result.created_at,
        }
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("подробности записаны в %s", args.output)

    return 0 if ok else 1


def _cmd_ab(args: argparse.Namespace) -> int:
    """A/B-сравнение вариантов из файла.

    Args:
        args: Разобранные аргументы.

    Returns:
        int: Код возврата процесса.
    """
    from eval.compare import main_async as ab_main
    from eval.metrics import default_metrics

    if not _mock_guard():
        return 2
    try:
        variants = _load_variants(args.variants)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 2
    return asyncio.run(ab_main(args, variants, default_metrics()))


def _cmd_bench(args: argparse.Namespace) -> int:
    """Операционный замер: латенси и поведение при нагрузке.

    Args:
        args: Разобранные аргументы.

    Returns:
        int: Код возврата процесса.
    """
    from eval.benchmark import main_async as bench_main

    return asyncio.run(bench_main(args))


def _cmd_tune(args: argparse.Namespace) -> int:
    """Сеточный перебор параметров по целевой метрике.

    Args:
        args: Разобранные аргументы.

    Returns:
        int: Код возврата процесса.
    """
    from eval.tune import main_async as tune_main

    if not _mock_guard():
        return 2
    return asyncio.run(tune_main(args))


def _cmd_report(args: argparse.Namespace) -> int:
    """Генерирует docs/BENCHMARK.md из кэша результатов.

    Args:
        args: Разобранные аргументы.

    Returns:
        int: Код возврата процесса.
    """
    from eval.report import generate

    return generate(
        ResultCache(args.cache_dir),
        args.output,
        tune_paths=getattr(args, "tune", None) or [],
    )


def _cmd_cache(args: argparse.Namespace) -> int:
    """Управление кэшем результатов.

    Args:
        args: Разобранные аргументы.

    Returns:
        int: Код возврата процесса.
    """
    cache = ResultCache(args.dir)
    if args.cache_command == "list":
        _print_list(cache.metadata())
    elif args.cache_command == "clear":
        cache.clear()
    elif args.cache_command == "ab-list":
        _print_ab_list(cache.list_ab())
    elif args.cache_command == "ab-clear":
        cache.clear_ab()
    return 0


def _load_variants(path: str):
    """Читает файл вариантов.

    Args:
        path: Путь к JSON-файлу.

    Returns:
        list[Variant]: Варианты.
    """
    from eval.compare import load_variants

    return load_variants(path)


def build_parser() -> argparse.ArgumentParser:
    """Собирает дерево аргументов.

    Returns:
        argparse.ArgumentParser: Парсер с подкомандами.
    """
    parser = argparse.ArgumentParser(prog="eval", description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    sub = parser.add_subparsers(dest="command", required=True, title="команды")

    run_p = sub.add_parser(
        "run", help="качество поиска и ответов на золотом наборе"
    )
    run_p.add_argument(
        "--golden", default=str(DEFAULT_GOLDEN), help="путь к золотому набору"
    )
    run_p.add_argument(
        "--with-answers",
        action="store_true",
        help="дополнительно генерировать ответы и проверять отказ по тексту",
    )
    run_p.add_argument("--limit", type=int, help="взять только первые N вопросов")
    run_p.add_argument(
        "--bootstraps",
        type=int,
        default=1,
        help="сколько независимых прогонов на ресемплированных пулах "
        "вопросов (бутстрап с возвратом)",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="зерно ресемплинга пула",
    )
    run_p.add_argument(
        "--graph",
        default="base",
        help="тег версии графа агента (base, rd-v2, ...): отдельная секция "
        "в отчёте и свой кэш",
    )
    run_p.add_argument(
        "--variant", default="base", help="имя варианта (по умолчанию base)"
    )
    run_p.add_argument(
        "--settings",
        help="JSON-оверрайды настроек, например '{\"retrieval_top_k\": 6}'",
    )
    run_p.add_argument("--output", help="записать подробности замера в файл")
    run_p.add_argument(
        "--cache-dir",
        default=str(DEFAULT_DIR),
        help="каталог кэша результатов",
    )
    run_p.add_argument(
        "--force", action="store_true", help="переснять вариант, игнорируя кэш"
    )

    ab_p = sub.add_parser("ab", help="A/B-сравнение вариантов")
    ab_p.add_argument(
        "--variants", required=True, help="путь к JSON-файлу с вариантами"
    )
    ab_p.add_argument(
        "--golden", default=str(DEFAULT_GOLDEN), help="путь к золотому набору"
    )
    ab_p.add_argument(
        "--with-answers",
        action="store_true",
        help="дополнительно генерировать ответы и проверять отказ по тексту",
    )
    ab_p.add_argument("--limit", type=int, help="взять только первые N вопросов")
    ab_p.add_argument(
        "--bootstraps",
        type=int,
        default=1,
        help="сколько независимых прогонов на ресемплированных пулах "
        "вопросов (бутстрап с возвратом)",
    )
    ab_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="зерно ресемплинга пула: один пул для всех вариантов",
    )
    ab_p.add_argument(
        "--graph",
        default="base",
        help="тег версии графа агента (base, rd-v2, ...): отдельная секция "
        "в отчёте и свой кэш",
    )
    ab_p.add_argument("--output", help="записать полный отчёт в файл")
    ab_p.add_argument(
        "--cache-dir",
        default=str(DEFAULT_DIR),
        help="каталог кэша результатов",
    )
    ab_p.add_argument(
        "--force", action="store_true", help="переснять все варианты, игнорируя кэш"
    )
    ab_p.add_argument(
        "--refresh-base",
        action="store_true",
        help="переснять только базовый вариант",
    )

    bench_p = sub.add_parser("bench", help="операционный замер: латенси и нагрузка")
    bench_p.add_argument(
        "--repeats", type=int, default=2, help="прогонов на каждый вопрос"
    )
    bench_p.add_argument(
        "--concurrency",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="уровни одновременных обращений; пусто — не мерить",
    )
    bench_p.add_argument("--output", help="записать результат в JSON")

    tune_p = sub.add_parser(
        "tune", help="сеточный перебор параметров по целевой метрике"
    )
    tune_p.add_argument(
        "--params",
        required=True,
        help='JSON-спека сетки: \'{"retrieval_top_k": [3, 5, 8]}\'',
    )
    tune_p.add_argument(
        "--objective",
        default="hit@1",
        help="ключ целевой метрики (по умолчанию hit@1)",
    )
    tune_p.add_argument(
        "--direction", default="max", choices=("min", "max"), help="направление"
    )
    tune_p.add_argument(
        "--golden", default=str(DEFAULT_GOLDEN), help="путь к золотому набору"
    )
    tune_p.add_argument("--limit", type=int, help="взять только первые N вопросов")
    tune_p.add_argument(
        "--with-answers",
        action="store_true",
        help="дополнительно генерировать ответы и проверять отказ по тексту",
    )
    tune_p.add_argument(
        "--bootstraps",
        type=int,
        default=1,
        help="прогонов на ресемплированных пулах (бутстрап с возвратом)",
    )
    tune_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="зерно ресемплинга пула",
    )
    tune_p.add_argument(
        "--graph",
        default="base",
        help="тег версии графа агента (base, rd-v2, ...): отдельная секция "
        "в отчёте и свой кэш",
    )
    tune_p.add_argument("--output", help="записать полный отчёт в JSON")
    tune_p.add_argument(
        "--cache-dir",
        default=str(DEFAULT_DIR),
        help="каталог кэша результатов",
    )
    tune_p.add_argument(
        "--force", action="store_true", help="переснять все точки, игнорируя кэш"
    )
    tune_p.add_argument(
        "--max-points",
        type=int,
        default=100,
        help="максимум точек сетки (защита от гигантского перебора)",
    )

    cache_p = sub.add_parser("cache", help="управление кэшем результатов")
    cache_p.add_argument("--dir", default=str(DEFAULT_DIR), help="каталог кэша")
    cache_sub = cache_p.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("list", help="список сохранённых результатов")
    cache_sub.add_parser("clear", help="удалить все сохранённые результаты")
    cache_sub.add_parser("ab-list", help="список A/B вердиктов")
    cache_sub.add_parser("ab-clear", help="удалить A/B вердикты")

    report_p = sub.add_parser(
        "report", help="обновить docs/BENCHMARK.md из кэша результатов"
    )
    report_p.add_argument(
        "--cache-dir",
        default=str(DEFAULT_DIR),
        help="каталог кэша результатов",
    )
    report_p.add_argument(
        "--output",
        default=str(DEFAULT_REPORT),
        help="куда писать отчёт (по умолчанию docs/BENCHMARK.md)",
    )
    report_p.add_argument(
        "--tune",
        action="append",
        default=[],
        help="путь к отчёту `eval tune --output`; включается в документ "
        "разделом «Итоги тюнинга по сетке»",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа.

    Args:
        argv: Аргументы; по умолчанию — ``sys.argv[1:]``.

    Returns:
        int: Код возврата процесса.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    for noisy in ("httpx", "httpcore", "infrastructure", "agent"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "ab":
        return _cmd_ab(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "tune":
        return _cmd_tune(args)
    if args.command == "report":
        return _cmd_report(args)
    return _cmd_cache(args)


if __name__ == "__main__":
    sys.exit(main())
