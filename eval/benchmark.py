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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("benchmark")

#: Вопросы прогона.
#:
#: Три вида намеренно: короткий фактический ответ, длинный процедурный и
#: вопрос без ответа в базе. Они дают разное время генерации, и усреднять
#: только по коротким значит завысить пропускную способность.
QUESTIONS = [
    (
        "короткий факт",
        "За сколько дней предупреждают о проверке знаний по охране труда?",
    ),
    ("длинная процедура", "Что должен сделать работник при обнаружении пожара?"),
    ("числовой", "Как часто проводится плановая идентификация опасностей?"),
    (
        "перечень",
        "Какие средства индивидуальной защиты обязательны на объекте строительства?",
    ),
    ("отказ", "Какая ставка налога на прибыль организаций?"),
]


async def one_answer(app, question: str) -> tuple[float, int, bool]:
    """Прогоняет один вопрос через граф целиком.

    Args:
        app: Скомпилированный граф.
        question: Текст вопроса.

    Returns:
        tuple[float, int, bool]: Время в секундах, длина ответа в символах,
        признак отказа «В базе знаний информации нет».
    """
    config = {
        "configurable": {
            "thread_id": f"bench-{uuid.uuid4()}",
            "user_id": "benchmark",
        }
    }
    started = time.perf_counter()
    result = await app.ainvoke(
        {"messages": [{"role": "human", "content": question}]}, config=config
    )
    elapsed = time.perf_counter() - started
    text = str(result["messages"][-1].content)
    refused = "в базе знаний информации нет" in text.lower().replace("ё", "е")
    return elapsed, len(text), refused


async def measure_sequential(app, repeats: int) -> list[dict]:
    """Меряет время ответа по одному запросу за раз.

    Args:
        app: Скомпилированный граф.
        repeats: Сколько раз прогнать каждый вопрос.

    Returns:
        list[dict]: Замеры по каждому вопросу.
    """
    rows: list[dict] = []
    for kind, question in QUESTIONS:
        times: list[float] = []
        chars = 0
        refused = False
        for _ in range(repeats):
            elapsed, length, was_refused = await one_answer(app, question)
            times.append(elapsed)
            chars = length
            refused = was_refused
        rows.append(
            {
                "вид": kind,
                "вопрос": question,
                "мин": round(min(times), 1),
                "медиана": round(statistics.median(times), 1),
                "макс": round(max(times), 1),
                "символов": chars,
                "отказ": refused,
            }
        )
        logger.info(
            "%-18s медиана %5.1f с  (%d симв.%s)",
            kind,
            statistics.median(times),
            chars,
            ", отказ" if refused else "",
        )
    return rows


async def measure_concurrent(app, level: int) -> dict:
    """Меряет поведение при одновременных обращениях.

    Ollama по умолчанию обслуживает запросы к одной модели последовательно,
    поэтому одновременные обращения не ускоряют работу, а выстраиваются в
    очередь. Замер показывает, во что это выливается для пользователя:
    растёт ли время ответа линейно числу одновременных запросов.

    Args:
        app: Скомпилированный граф.
        level: Сколько запросов запускать одновременно.

    Returns:
        dict: Общее время, худшее время одного ответа, пропускная
        способность в ответах в минуту.
    """
    questions = [QUESTIONS[i % len(QUESTIONS)][1] for i in range(level)]
    started = time.perf_counter()
    results = await asyncio.gather(*(one_answer(app, q) for q in questions))
    total = time.perf_counter() - started
    worst = max(r[0] for r in results)
    return {
        "одновременно": level,
        "всего_с": round(total, 1),
        "худший_ответ_с": round(worst, 1),
        "ответов_в_минуту": round(level / total * 60, 1),
    }


async def main_async(args: argparse.Namespace) -> int:
    """Выполняет замер.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        int: Код возврата процесса.
    """
    from infrastructure.config.composition_root import create_application
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

    app = create_application()

    print("\n" + "=" * 78)
    print(f"ВРЕМЯ ОТВЕТА, ПО ОДНОМУ ЗАПРОСУ ({args.repeats} прогона на вопрос)")
    print("=" * 78)
    rows = await measure_sequential(app, args.repeats)
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

    concurrency: list[dict] = []
    if args.concurrency:
        print("\n" + "=" * 78)
        print("ОДНОВРЕМЕННЫЕ ОБРАЩЕНИЯ")
        print("=" * 78)
        for level in args.concurrency:
            row = await measure_concurrent(app, level)
            concurrency.append(row)
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
