"""QA-индексация: фрагмент кодируется вместе с вопросами, на которые отвечает.

Зачем. Человек спрашивает «что делать при сигнале "Внимание всем"», а в
регламенте написано «при получении сигнала оповещения работники обязаны...».
Вектор вопроса и вектор ответа лежат в разных местах пространства - формы
речи разные. Если приписать к фрагменту вопросы, на которые он отвечает, его
вектор сдвигается в сторону вопросов, и попадание становится вероятнее.

Что важно: приписка уходит ТОЛЬКО в индекс. Пользователю показывается
исходный текст фрагмента - иначе в панели источников появились бы служебные
строки «Q:», которых в регламенте нет.

Почему обогащаются все фрагменты, а не «трудные». Первая версия отбирала
подозрительные фрагменты (начатые с середины фразы, табличные, из
нераспознанных разделов) - на них уходило бы вчетверо меньше вызовов модели.
Замер это опроверг: среди промахов поиска таких фрагментов оказалось 13%, а
среди найденных с первой позиции - 28%. То есть отбор целил ровно не в те
фрагменты: разрыв между вопросом и текстом создаёт не структура документа, а
словарь, и предсказать его по разметке нельзя. Обогащается весь корпус.

Цена и выигрыш (121 вопрос золотого набора, bge-m3, без переоценки):
hit@3 0,769 -> 0,860 (p=0,027), hit@5 0,876 -> 0,926 (p=0,031, ни одного
ухудшения), не найдено вовсе 15 -> 9. Прогон по 1388 фрагментам занял около
трёх часов на ноутбуке с qwen3:8b; это разовая цена индексации.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from infrastructure.ingest.models import IngestChunk

logger = logging.getLogger(__name__)

#: Сколько вопросов просить на фрагмент. Больше трёх начинают повторять друг
#: друга и разбавляют вектор.
QUESTIONS_PER_CHUNK = 3

#: Сколько текста фрагмента показывать модели. Дальше идёт хвост, который на
#: формулировку вопросов уже не влияет, а время разбора запроса растит.
PROMPT_CHARS = 1200

#: Сколько запросов держать в полёте. Ограничение стоит на нашей стороне,
#: потому что сервер модели свою очередь не разгоняет, а память ест.
DEFAULT_CONCURRENCY = 3

PROMPT = """Ты составляешь вопросы для поиска по нормативной базе предприятия.

Прочитай фрагмент регламента и напиши {count} коротких вопроса, ответ на \
которые в нём есть. Вопросы задаёт мастер участка или инженер по охране \
труда живым языком, а не языком документа.

Правила:
- по одному вопросу в строке, без нумерации и без пояснений;
- вопрос должен отвечаться именно этим фрагментом, а не соседними;
- используй слова, которыми спросил бы человек, а не заголовок раздела.

Фрагмент:
{text}"""


@dataclass(frozen=True, slots=True)
class QaStats:
    """Итог прогона QA-индексации.

    Attributes:
        considered: Сколько фрагментов рассмотрено.
        enriched: Скольким приписаны вопросы.
        failed: Сколько раз модель не дала ничего пригодного.
    """

    considered: int = 0
    enriched: int = 0
    failed: int = 0


def parse_questions(raw: str, limit: int = QUESTIONS_PER_CHUNK) -> list[str]:
    """Разбирает ответ модели в список вопросов.

    Args:
        raw: Текст ответа модели.
        limit: Сколько вопросов оставить.

    Returns:
        list[str]: Вопросы без нумерации и повторов. Пустой список, если
        модель не дала ничего пригодного.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        text = line.strip()
        if not text:
            continue
        # Модель нередко всё же нумерует, несмотря на просьбу.
        text = re.sub(r"^[\-\*\d]+[.)]?\s*", "", text).strip()
        if len(text) < 12 or "?" not in text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def build_indexed_text(chunk: IngestChunk, questions: list[str]) -> str:
    """Собирает текст, который уйдёт в вектор.

    Args:
        chunk: Фрагмент документа.
        questions: Вопросы, на которые он отвечает.

    Returns:
        str: Текст вида ``Q: ... A: <фрагмент>``; без вопросов - исходный
        текст без изменений.
    """
    if not questions:
        return chunk.text
    asked = " ".join(f"Q: {q}" for q in questions)
    return f"{asked}\nA: {chunk.text}"


async def _ask(model, chunk: IngestChunk) -> list[str]:
    """Просит модель придумать вопросы к фрагменту.

    Args:
        model: Модель с методом ``ainvoke``.
        chunk: Фрагмент документа.

    Returns:
        list[str]: Вопросы; пустой список при сбое - индексация идёт
        десятками минут, и терять её из-за одного вызова нельзя.
    """
    from langchain_core.messages import SystemMessage

    prompt = PROMPT.format(count=QUESTIONS_PER_CHUNK, text=chunk.text[:PROMPT_CHARS])
    try:
        answer = await model.ainvoke([SystemMessage(content=prompt)])
    except Exception as exc:
        logger.warning("вопросы не сгенерированы для %s: %s", chunk.chunk_id, exc)
        return []
    return parse_questions(str(getattr(answer, "content", "") or ""))


async def enrich(
    chunks: list[IngestChunk],
    model,
    limit: int = 0,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> QaStats:
    """Дописывает фрагментам вопросы для индексации.

    Изменяется только ``text_indexed``; ``text`` остаётся исходным, потому
    что именно он показывается в панели источников.

    Args:
        chunks: Фрагменты документа; изменяются на месте.
        model: Модель с методом ``ainvoke``.
        limit: Предел числа обогащаемых фрагментов; ``0`` - без предела.
            Каждый фрагмент стоит вызова модели.
        concurrency: Сколько запросов держать в полёте.

    Returns:
        QaStats: Сколько рассмотрено, обогащено и не удалось.
    """
    targets = chunks[:limit] if limit else chunks
    if not targets:
        return QaStats()

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(chunk: IngestChunk) -> bool:
        """Обогащает один фрагмент.

        Args:
            chunk: Фрагмент документа.

        Returns:
            bool: Удалось ли приписать вопросы.
        """
        async with semaphore:
            questions = await _ask(model, chunk)
        if not questions:
            return False
        object.__setattr__(chunk, "text_indexed", build_indexed_text(chunk, questions))
        return True

    results = await asyncio.gather(*(one(chunk) for chunk in targets))
    enriched = sum(1 for ok in results if ok)
    stats = QaStats(
        considered=len(targets), enriched=enriched, failed=len(targets) - enriched
    )
    logger.info(
        "QA-индексация: рассмотрено %d, обогащено %d, не удалось %d",
        stats.considered,
        stats.enriched,
        stats.failed,
    )
    return stats
