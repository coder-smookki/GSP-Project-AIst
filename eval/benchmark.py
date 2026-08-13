"""Замер эксплуатационных характеристик на целевой машине.

Собирает то, чего нельзя вывести из качества поиска: сколько времени
занимает полный ответ, сколько запросов машина выдерживает подряд и как
ведёт себя при одновременных обращениях. Эти числа нужны для расчёта
пропускной способности и для дорожной карты, поэтому меряются на том
железе, на котором система будет работать.

Запуск на целевой машине с поднятым стеком и проиндексированным корпусом::

    uv run python eval/benchmark.py
    uv run python eval/benchmark.py --concurrency 1 2 4 --repeats 3

Результат печатается таблицей и, при указании ``--output``, пишется в JSON.
Измерение выполняет :class:`eval.runners.OperationalRunner`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from pathlib import Path

from eval.config import RunConfig, Variant
from eval.metrics import LatencyMedian, LatencyP95
from eval.runners import OperationalRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("benchmark")


async def main_async(args: argparse.Namespace) -> int:
    """Выполняет замер.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        int: Код возврата процесса.
    """
    from eval.variants import build_variant
    from infrastructure.config.settings import get_settings

    settings = get_settings()
    if settings.resolved_retrieval_provider() == "mock":
        logger.error("замер идёт на заглушке поиска: задайте RETRIEVAL_PROVIDER=qdrant")
        return 2

    print("=" * 78)
    print("ОКРУЖЕНИЕ")
    print("=" * 78)
    print(f"  модель ответа:      {settings.llm_model}")
    print(f"  служебная модель:   {settings.llm_support_model}")
    print(f"  эмбеддер:           {settings.embedding_model}")
    print(f"  фрагментов в ответ: {settings.retrieval_top_k}")
    rerank_state = "включена" if settings.rerank_enabled else "выключена"
    print(f"  переоценка:         {rerank_state}")
    think_state = "включены" if settings.llm_reasoning else "выключены"
    print(f"  размышления:        {think_state}")

    runner = OperationalRunner(
        build=build_variant, metrics=[LatencyMedian(), LatencyP95()]
    )
    config = RunConfig(
        runner="operational",
        repeats=args.repeats,
        concurrency=tuple(args.concurrency),
    )
    result = await runner.run(Variant(name="base"), config)

    rows: list[dict] = result.meta["sequential"]
    print("\n" + "=" * 78)
    print(f"ВРЕМЯ ОТВЕТА, ПО ОДНОМУ ЗАПРОСУ ({args.repeats} прогона на вопрос)")
    print("=" * 78)
    print(f"\n{'вид':<18}{'мин':>7}{'медиана':>10}{'макс':>7}{'симв.':>8}")
    for row in rows:
        print(
            f"{row['вид']:<18}{row['мин']:>7.1f}{row['медиана']:>10.1f}"
            f"{row['макс']:>7.1f}{row['символов']:>8}"
        )
    answered = [r for r in rows if not r["отказ"]]
    if answered:
        avg = statistics.median([r["медиана"] for r in answered])
        print(f"\n  медиана по содержательным ответам: {avg:.1f} с")
        print(f"  теоретический предел одной машины: {3600 / avg:.0f} ответов в час")

    concurrency: list[dict] = result.meta["concurrency"]
    if concurrency:
        print("\n" + "=" * 78)
        print("ОДНОВРЕМЕННЫЕ ОБРАЩЕНИЯ")
        print("=" * 78)
        for row in concurrency:
            print(
                f"  {row['одновременно']:>2} запросов: всего {row['всего_с']:>6.1f} с, "
                f"худший ответ {row['худший_ответ_с']:>6.1f} с, "
                f"{row['ответов_в_минуту']:>5.1f} ответов/мин"
            )

    if args.output:
        payload = {
            "окружение": {
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "retrieval_top_k": settings.retrieval_top_k,
                "rerank_enabled": settings.rerank_enabled,
            },
            "последовательно": rows,
            "одновременно": concurrency,
            "метрики": result.metrics,
        }
        await asyncio.to_thread(
            Path(args.output).write_text,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nподробности записаны в {args.output}")

    return 0


def main() -> int:
    """Точка входа.

    Returns:
        int: Код возврата процесса.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=2, help="прогонов на каждый вопрос"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="уровни одновременных обращений; пусто — не мерить",
    )
    parser.add_argument("--output", help="записать результат в JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "infrastructure", "agent"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
