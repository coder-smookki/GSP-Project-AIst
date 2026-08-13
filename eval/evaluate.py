"""Замер качества поиска и ответов на золотом наборе вопросов.

Зачем это нужно. До сих пор качество подтверждалось точечными прогонами:
задали вопрос, посмотрели ответ, решили, что хорошо. Такой способ не ловит
ухудшений. Поменяли модель эмбеддингов, границы чанков, число фрагментов в
контексте — и узнать, стало лучше или хуже, можно только замером на
постоянном наборе вопросов с заранее известными ответами.

Набор лежит в `tests/fixtures/golden/questions.json` и делится надвое:

* **отвечаемые** — вопрос, краткий ответ и дословная цитата из документа.
  Цитата и есть эталон: фрагмент считается попаданием, если содержит её
  текст. Проверяется не формулировка ответа модели, а то, попали ли нужные
  сведения в контекст вообще — без этого модель права быть не может.
* **без ответа** — вопросы, на которые правильный ответ системы «В базе
  знаний информации нет». Делятся на явно посторонние и на самые опасные:
  звучащие уместно, с лексикой корпуса, но без ответа в нём. Именно на
  вторых система и выдумывает.

Два режима. Без ключей считается только качество поиска: быстро,
воспроизводимо, без обращения к большой модели. С ключом ``--with-answers``
дополнительно генерируются ответы и проверяется, отказывается ли система
там, где обязана.

Запуск::

    uv run python eval/evaluate.py
    uv run python eval/evaluate.py --with-answers --limit 10
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

from eval.golden import chunk_contains
from eval.metrics import percentile
from eval.result import Outcome

# При прямом запуске (`uv run python eval/evaluate.py`) в sys.path попадает
# каталог eval, а не корень репозитория, и импорт `infrastructure`
# падает. Полагаться на редактируемую установку нельзя: она отваливается
# при пересборке окружения, а команда описана в README как рабочая.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("evaluate")

#: Путь к золотому набору по умолчанию.
DEFAULT_GOLDEN = Path("tests/fixtures/golden/questions.json")

#: Ответ, который система обязана дать при отсутствии сведений.
REFUSAL = "в базе знаний информации нет"


async def run_answerable(search, items: list[dict]) -> list[Outcome]:
    """Замеряет поиск на вопросах с известным ответом.

    Args:
        search: Поиск по базе знаний.
        items: Вопросы золотого набора с цитатами.

    Returns:
        list[Outcome]: Итоги по каждому вопросу.
    """
    results: list[Outcome] = []
    for item in items:
        started = time.perf_counter()
        hits = await search.query(item["question"])
        elapsed = time.perf_counter() - started

        outcome = Outcome(
            question=item["question"],
            returned=len(hits),
            seconds=elapsed,
            doc_codes=[chunk.doc_code for chunk, _ in hits],
        )
        for position, (chunk, _score) in enumerate(hits, 1):
            if chunk_contains(item["quote"], chunk.text):
                outcome.rank = position
                break
        results.append(outcome)
        logger.info(
            "%-60s %s",
            item["question"][:60],
            f"позиция {outcome.rank}" if outcome.rank else "НЕ НАЙДЕНО",
        )
    return results


async def run_unanswerable(search, items: list[dict], agent=None) -> list[Outcome]:
    """Замеряет поведение на вопросах, ответа на которые в базе нет.

    Args:
        search: Поиск по базе знаний.
        items: Вопросы без ответа.
        agent: Скомпилированный граф; если задан, дополнительно проверяется
            текст ответа. Без него отказ определяется только по пустой
            выдаче поиска.

    Returns:
        list[Outcome]: Итоги по каждому вопросу.
    """
    results: list[Outcome] = []
    for item in items:
        started = time.perf_counter()
        hits = await search.query(item["question"])
        elapsed = time.perf_counter() - started

        outcome = Outcome(
            question=item["question"],
            returned=len(hits),
            seconds=elapsed,
            refused=not hits,
        )

        if agent is not None:
            state = {"messages": [{"role": "human", "content": item["question"]}]}
            # Свой thread_id на каждый вопрос: чекпоинтер его требует, а
            # общий на весь замер тянул бы историю предыдущих вопросов в
            # контекст и портил результат.
            config = {
                "configurable": {
                    "thread_id": f"eval-{uuid.uuid4()}",
                    "user_id": "evaluate",
                }
            }
            final = await agent.ainvoke(state, config=config)
            text = str(final["messages"][-1].content)
            outcome.answer = text
            outcome.refused = REFUSAL in text.lower().replace("ё", "е")

        results.append(outcome)
        logger.info(
            "%-60s %s",
            item["question"][:60],
            "отказ" if outcome.refused else "ОТВЕТИЛА",
        )
    return results


def report(answerable: list[Outcome], unanswerable: list[Outcome]) -> bool:
    """Печатает сводку и решает, пройден ли замер.

    Args:
        answerable: Итоги по вопросам с ответом.
        unanswerable: Итоги по вопросам без ответа.

    Returns:
        bool: ``True``, если замер пройден. Провалом считается только
        полное отсутствие результатов: пороги качества сознательно не
        зашиты, чтобы замер показывал цифры, а не выносил приговор.
    """
    print("\n" + "=" * 72)
    print("КАЧЕСТВО ПОИСКА (вопросы с известным ответом)")
    print("=" * 72)

    if answerable:
        found = [o for o in answerable if o.rank]
        ranks = [o.rank for o in found]
        mrr = sum(1 / r for r in ranks) / len(answerable) if ranks else 0.0
        times = [o.seconds for o in answerable]

        print(f"вопросов: {len(answerable)}")
        for k in (1, 3, 5):
            hits = sum(1 for r in ranks if r <= k)
            share = hits / len(answerable)
            print(f"  hit@{k}: {share:.3f}  ({hits} из {len(answerable)})")
        print(f"  MRR:   {mrr:.3f}")
        print(f"  ответ не найден вовсе: {len(answerable) - len(found)}")
        print(
            f"  время поиска: медиана {statistics.median(times) * 1000:.0f} мс, "
            f"p95 {percentile(times, 0.95) * 1000:.0f} мс"
        )

        missed = [o for o in answerable if not o.rank]
        if missed:
            print("\n  не найдено:")
            for outcome in missed:
                print(f"    - {outcome.question[:66]}")
    else:
        print("вопросов нет")

    print("\n" + "=" * 72)
    print("ЧЕСТНЫЙ ОТКАЗ (вопросы, ответа на которые в базе нет)")
    print("=" * 72)

    if unanswerable:
        refused = [o for o in unanswerable if o.refused]
        print(f"вопросов: {len(unanswerable)}")
        print(
            f"  верных отказов: {len(refused)} из {len(unanswerable)} "
            f"({len(refused) / len(unanswerable):.3f})"
        )
        answered = [o for o in unanswerable if not o.refused]
        if answered:
            print("\n  система ответила там, где обязана была отказать:")
            for outcome in answered:
                print(f"    - {outcome.question[:66]}")
                if outcome.answer:
                    print(f"      -> {outcome.answer[:120]}")
    else:
        print("вопросов нет")

    print()
    return bool(answerable or unanswerable)


async def main_async(args: argparse.Namespace, golden: dict) -> int:
    """Выполняет замер.

    Args:
        args: Разобранные аргументы командной строки.
        golden: Загруженный золотой набор.

    Returns:
        int: Код возврата процесса.
    """
    answerable_items = golden.get("answerable", [])
    unanswerable_items = golden.get("unanswerable", [])

    if args.limit:
        answerable_items = answerable_items[: args.limit]
        unanswerable_items = unanswerable_items[: args.limit]

    from infrastructure.config.model_registry import setup_registry
    from infrastructure.config.settings import get_settings
    from infrastructure.retrieval.factory import get_search, shutdown

    settings = get_settings()
    if settings.resolved_retrieval_provider == "mock":
        logger.error(
            "замер идёт на заглушке поиска: задайте RETRIEVAL_PROVIDER=qdrant, "
            "иначе цифры ничего не значат"
        )
        return 2

    setup_registry(settings)
    search = get_search(settings)

    agent = None
    if args.with_answers:
        from infrastructure.config.composition_root import create_application

        agent = create_application()

    try:
        answerable = await run_answerable(search, answerable_items)
        unanswerable = await run_unanswerable(search, unanswerable_items, agent)
    finally:
        await shutdown()

    ok = report(answerable, unanswerable)

    if args.output:
        payload = {
            "answerable": [vars(o) for o in answerable],
            "unanswerable": [vars(o) for o in unanswerable],
        }
        await asyncio.to_thread(
            Path(args.output).write_text,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("подробности записаны в %s", args.output)

    return 0 if ok else 1


def main() -> int:
    """Точка входа.

    Returns:
        int: Код возврата процесса.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden", default=str(DEFAULT_GOLDEN), help="путь к золотому набору"
    )
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="дополнительно генерировать ответы и проверять отказ по тексту",
    )
    parser.add_argument("--limit", type=int, help="взять только первые N вопросов")
    parser.add_argument("--output", help="записать подробности замера в файл")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    golden_path = Path(args.golden)
    if not golden_path.exists():
        logger.error("золотой набор не найден: %s", golden_path)
        return 2
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    return asyncio.run(main_async(args, golden))


if __name__ == "__main__":
    sys.exit(main())
