"""QA-индексация: разбор ответа модели и сборка текста для вектора."""

from __future__ import annotations

import pytest

from infrastructure.ingest.models import IngestChunk, SectionType
from infrastructure.ingest.qa_index import (
    build_indexed_text,
    enrich,
    parse_questions,
)


def _chunk(text: str, chunk_id: str = "ДОК-1#c0001") -> IngestChunk:
    """Собирает фрагмент для проверки.

    Args:
        text: Текст фрагмента без строки breadcrumb.
        chunk_id: Идентификатор фрагмента.

    Returns:
        IngestChunk: Фрагмент.
    """
    return IngestChunk(
        chunk_id=chunk_id,
        doc_id="1",
        doc_code="ДОК-1",
        doc_title="Регламент",
        section_no="4",
        section_title="Порядок проведения",
        section_type=SectionType.PROCEDURA,
        clause="4.1",
        breadcrumb="ДОК-1 > 4. Порядок проведения",
        text=f"ДОК-1 > 4. Порядок проведения\n{text}",
        text_raw=text,
        order_no=1,
    )


def test_questions_are_parsed_without_numbering() -> None:
    """Нумерация и маркеры срезаются, повторы отбрасываются."""
    raw = (
        "1. Как оформить наряд-допуск?\n"
        "- Как оформить наряд-допуск?\n"
        "2) Кто подписывает наряд?\n"
        "просто строка"
    )

    assert parse_questions(raw) == [
        "Как оформить наряд-допуск?",
        "Кто подписывает наряд?",
    ]


def test_lines_without_a_question_mark_are_dropped() -> None:
    """Строки без знака вопроса не считаются вопросами.

    Модель охотно дописывает пояснения вроде «Вот три вопроса:».
    """
    assert parse_questions("Вот три вопроса:\nЧто делать при пожаре?") == [
        "Что делать при пожаре?"
    ]


def test_more_than_three_questions_are_cut() -> None:
    """Больше трёх вопросов не берётся: они разбавляют вектор."""
    raw = "\n".join(f"Вопрос номер {n} звучит так?" for n in range(6))

    assert len(parse_questions(raw)) == 3


def test_indexed_text_keeps_the_original_body() -> None:
    """Текст для вектора содержит и вопросы, и исходный фрагмент."""
    chunk = _chunk("Работы ведутся по наряду.")
    built = build_indexed_text(chunk, ["Как вести работы?"])

    assert built.startswith("Q: Как вести работы?")
    assert "Работы ведутся по наряду." in built


def test_without_questions_the_text_is_unchanged() -> None:
    """Пустой список вопросов не портит фрагмент."""
    chunk = _chunk("Работы ведутся по наряду.")

    assert build_indexed_text(chunk, []) == chunk.text


class _Model:
    """Модель, отдающая заранее заданные вопросы."""

    def __init__(self, answer: str = "Что делать при пожаре?\nКто отвечает?") -> None:
        """Готовит модель.

        Args:
            answer: Текст, который вернёт модель.
        """
        self.answer = answer
        self.calls = 0

    async def ainvoke(self, messages):
        """Возвращает заданный ответ.

        Args:
            messages: История сообщений.

        Returns:
            object: Объект с полем ``content``.
        """
        self.calls += 1

        class _R:
            content = self.answer

        return _R()


@pytest.mark.asyncio
async def test_every_chunk_is_enriched() -> None:
    """Обогащается весь корпус, а не выборка.

    Отбор «трудных» фрагментов по разметке документа проверялся замером и
    не подтвердился: среди промахов поиска таких оказалось меньше, чем
    среди уверенных попаданий.
    """
    chunks = [_chunk(f"Текст {n}.", chunk_id=f"ДОК-1#c000{n}") for n in range(4)]
    model = _Model()

    stats = await enrich(chunks, model)

    assert model.calls == 4
    assert stats.enriched == 4
    assert all(c.text_indexed.startswith("Q: ") for c in chunks)


@pytest.mark.asyncio
async def test_display_text_is_never_touched() -> None:
    """Показываемый текст остаётся исходным.

    Иначе в панели источников появились бы строки «Q:», которых в
    регламенте нет, и цитата перестала бы быть цитатой.
    """
    chunk = _chunk("Работы ведутся по наряду.")
    before = chunk.text

    await enrich([chunk], _Model())

    assert chunk.text == before
    assert chunk.text_indexed != chunk.text


@pytest.mark.asyncio
async def test_model_failure_does_not_break_indexing() -> None:
    """Сбой модели оставляет фрагмент как есть, а не роняет прогон.

    Индексация идёт часами; терять её из-за одного неудачного вызова
    нельзя, а фрагмент без вопросов просто кодируется как раньше.
    """

    class _Broken(_Model):
        async def ainvoke(self, messages):
            """Всегда падает.

            Args:
                messages: История сообщений.

            Raises:
                RuntimeError: Всегда.
            """
            raise RuntimeError("модель недоступна")

    chunk = _chunk("Работы ведутся по наряду.")
    stats = await enrich([chunk], _Broken())

    assert stats.failed == 1
    assert chunk.text_indexed == ""


@pytest.mark.asyncio
async def test_unusable_answer_counts_as_a_failure() -> None:
    """Ответ без единого вопроса не портит вектор фрагмента."""
    chunk = _chunk("Работы ведутся по наряду.")

    stats = await enrich([chunk], _Model(answer="Хорошо, вот вопросы:"))

    assert stats.failed == 1
    assert chunk.text_indexed == ""


@pytest.mark.asyncio
async def test_limit_caps_the_number_of_calls() -> None:
    """Предел ограничивает число обращений к модели."""
    chunks = [_chunk(f"Текст {n}.", chunk_id=f"ДОК-1#c000{n}") for n in range(5)]
    model = _Model()

    stats = await enrich(chunks, model, limit=2)

    assert model.calls == 2
    assert stats.considered == 2
    assert chunks[4].text_indexed == ""


@pytest.mark.asyncio
async def test_requests_do_not_all_go_at_once() -> None:
    """Одновременных запросов не больше заданного предела.

    Сервер модели свою очередь не разгоняет, а память под слоты ест: без
    ограничения прогон на большом корпусе упирается в память.
    """
    peak = 0
    live = 0

    class _Counting(_Model):
        async def ainvoke(self, messages):
            """Считает одновременные вызовы.

            Args:
                messages: История сообщений.

            Returns:
                object: Объект с полем ``content``.
            """
            nonlocal peak, live
            live += 1
            peak = max(peak, live)
            import asyncio

            await asyncio.sleep(0)
            live -= 1
            return await super().ainvoke(messages)

    chunks = [_chunk(f"Текст {n}.", chunk_id=f"ДОК-1#c000{n}") for n in range(12)]
    await enrich(chunks, _Counting(), concurrency=3)

    assert peak <= 3
