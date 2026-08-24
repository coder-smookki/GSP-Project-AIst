"""Разбор непопаданий: почему нужный фрагмент не дошёл до модели.

Отвечает на вопрос, который нельзя решить рассуждением: если фрагмент с
ответом лежит на пятидесятой позиции, воронку контекста имеет смысл
расширять; если его нет и в двух сотнях кандидатов, дело не в ширине, а в
самом представлении текста, и расширение только замедлит поиск.

Для каждого вопроса золотого набора ищем позицию эталонного фрагмента при
разной глубине и печатаем, где он на самом деле находится.

Запуск::

    uv run python scripts/diagnose_misses.py
    uv run python scripts/diagnose_misses.py --depth 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("diagnose")

MIN_MATCH_WORDS = 6


def _words(text: str) -> list[str]:
    """Приводит текст к списку слов.

    Args:
        text: Исходный текст.

    Returns:
        list[str]: Слова в нижнем регистре без пунктуации.
    """
    return re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))


def _run(needle: list[str], haystack: list[str]) -> int:
    """Длина самой длинной общей цепочки слов подряд.

    Args:
        needle: Слова эталонной цитаты.
        haystack: Слова фрагмента.

    Returns:
        int: Число совпавших слов подряд.
    """
    if not needle or not haystack:
        return 0
    positions: dict[str, list[int]] = {}
    for index, word in enumerate(haystack):
        positions.setdefault(word, []).append(index)
    best = 0
    previous: dict[int, int] = {}
    for word in needle:
        current: dict[int, int] = {}
        for index in positions.get(word, ()):
            run = previous.get(index - 1, 0) + 1
            current[index] = run
            best = max(best, run)
        previous = current
    return best


def _contains(quote: str, text: str) -> bool:
    """Проверяет, содержит ли фрагмент эталонную цитату.

    Args:
        quote: Дословная цитата из документа.
        text: Текст фрагмента.

    Returns:
        bool: ``True``, если совпало достаточно слов подряд.
    """
    words = _words(quote)
    if not words:
        return False
    return _run(words, _words(text)) >= min(MIN_MATCH_WORDS, len(words))


async def main_async(args: argparse.Namespace) -> int:
    """Выполняет разбор.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        int: Код возврата процесса.
    """
    from infrastructure.config.model_registry import setup_registry
    from infrastructure.config.settings import get_settings
    from infrastructure.retrieval.factory import get_search, shutdown

    settings = get_settings()
    if settings.resolved_retrieval_provider() == "mock":
        logger.error("нужен настоящий индекс: задайте RETRIEVAL_PROVIDER=qdrant")
        return 2
    setup_registry(settings)

    raw = await asyncio.to_thread(
        Path("tests/fixtures/golden/questions.json").read_text, encoding="utf-8"
    )
    items = json.loads(raw)["answerable"]

    print(f"эмбеддер {settings.embedding_model}, глубина разбора {args.depth}")
    print(
        f"рабочие настройки: top_k={settings.retrieval_top_k}, "
        f"prefetch={settings.retrieval_prefetch}, "
        f"порог={settings.retrieval_min_relevance}\n"
    )

    # Широкая выдача без порога и без переоценки: нас интересует, где эталон
    # лежит в принципе, а не что доходит до модели при текущих настройках.
    wide = get_search(
        settings.model_copy(
            update={
                "retrieval_top_k": args.depth,
                "retrieval_prefetch": args.depth,
                "retrieval_min_relevance": 0.0,
                "rerank_enabled": False,
            }
        )
    )

    buckets = {"1": 0, "2-5": 0, "6-20": 0, "21-50": 0, "51+": 0, "нет": 0}
    misses: list[tuple[str, int | None]] = []

    for item in items:
        hits = await wide.query(item["question"])
        position = None
        for index, (chunk, _score) in enumerate(hits, 1):
            if _contains(item["quote"], chunk.text):
                position = index
                break

        if position is None:
            buckets["нет"] += 1
        elif position == 1:
            buckets["1"] += 1
        elif position <= 5:
            buckets["2-5"] += 1
        elif position <= 20:
            buckets["6-20"] += 1
        elif position <= 50:
            buckets["21-50"] += 1
        else:
            buckets["51+"] += 1

        if position is None or position > settings.retrieval_top_k:
            misses.append((item["question"], position))

    print("Где лежит эталонный фрагмент при широком поиске:")
    for name, count in buckets.items():
        share = count / len(items) if items else 0
        bar = "#" * round(share * 40)
        print(f"  позиция {name:<6} {count:>3} из {len(items)}  {bar}")

    print(f"\nНе доходят до модели при top_k={settings.retrieval_top_k}: {len(misses)}")
    for question, position in misses:
        where = f"позиция {position}" if position else "НЕ НАЙДЕН НИ НА КАКОЙ ГЛУБИНЕ"
        print(f"  {where:<38} {question[:60]}")

    unreachable = buckets["нет"]
    deep = buckets["21-50"] + buckets["51+"]
    print("\nВывод:")
    if deep:
        print(
            f"  {deep} фрагментов лежат глубже 20-й позиции — расширение воронки "
            f"их достанет, но ценой времени переоценки."
        )
    if unreachable:
        print(
            f"  {unreachable} фрагментов не находятся вовсе — ширина не поможет, "
            f"дело в представлении текста: формулировка вопроса, нарезка или эмбеддер."
        )
    if not deep and not unreachable:
        print("  все эталоны в первой двадцатке: расширять воронку смысла нет.")

    await shutdown()
    return 0


def main() -> int:
    """Точка входа.

    Returns:
        int: Код возврата процесса.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=200, help="глубина разбора")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
