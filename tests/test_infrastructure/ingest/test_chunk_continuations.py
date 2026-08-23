"""Нарезка не оставляет фрагментов, начинающихся с середины фразы.

Дефект из issue #53: фрагмент `05d08d3e-...` начинался словами «различного
назначения (жилых, промышленных...)», а начало предложения - где сказано,
для чего средство и когда применяется, - уехало в соседний фрагмент. Вектор
такого хвоста не отвечает ни на один осмысленный вопрос, и фрагмент не
находился никогда.
"""

from __future__ import annotations

from infrastructure.ingest.chunker import chunk_document
from infrastructure.ingest.models import (
    Atom,
    ParsedDocument,
    ParsedSection,
    SectionType,
)


def _atom(text: str, clause: str | None = None) -> Atom:
    """Собирает атом разбора.

    Args:
        text: Текст атома.
        clause: Номер пункта.

    Returns:
        Atom: Атом документа.
    """
    return Atom(
        clause=clause,
        text=text,
        section_no="1",
        section_title="Общие положения",
        section_type=SectionType.OBSHCHIE,
        is_table_region=False,
    )


def _document(sections: list[ParsedSection]) -> ParsedDocument:
    """Собирает разобранный документ.

    Args:
        sections: Разделы документа.

    Returns:
        ParsedDocument: Документ для нарезки.
    """
    return ParsedDocument(
        doc_id="88540",
        doc_code="ДОК-8",
        doc_title="Меры пожарной безопасности",
        source="files",
        sections=tuple(sections),
        glossary={},
        updated_at=None,
    )


def _bodies(chunks) -> list[str]:
    """Возвращает тексты фрагментов без строки breadcrumb.

    Args:
        chunks: Фрагменты нарезки.

    Returns:
        list[str]: Тексты без заголовочной строки.
    """
    return [c.text.split("\n", 1)[-1].strip() for c in chunks]


def test_continuation_is_glued_to_the_previous_fragment() -> None:
    """Хвост фразы не становится отдельным фрагментом.

    Первый атом набирает почти весь целевой объём, поэтому продолжение по
    прежним правилам открывало новую группу - и получался фрагмент, который
    начинается с середины предложения.
    """
    head = "А" * 880 + ". Самоспасатель фильтрующий предназначен для"
    tail = "различного назначения, а также зон химического заражения."
    document = _document(
        [
            ParsedSection(
                no="1",
                title="Общие положения",
                type=SectionType.OBSHCHIE,
                atoms=(_atom(head), _atom(tail)),
            )
        ]
    )

    bodies = _bodies(chunk_document(document))

    assert not any(body.startswith("различного") for body in bodies)
    assert any("различного назначения" in body for body in bodies)


def test_continuation_is_glued_across_a_section_boundary() -> None:
    """Продолжение прирастает, даже если разделы у половин разные.

    Классификатор типов разделов иногда разрывает сплошной абзац надвое и
    приписывает половинам разные типы. Текст со строчной буквы - продолжение
    по определению, верить надо ему, а не разметке.
    """
    head = "Б" * 880 + ". Работы выполняются по наряду-допуску и"
    tail = "оформляются до начала работ ответственным руководителем."
    other = Atom(
        clause=None,
        text=tail,
        section_no="2",
        section_title="Ответственность",
        section_type=SectionType.OTVETSTVENNOST,
        is_table_region=False,
    )
    document = _document(
        [
            ParsedSection(
                no="1",
                title="Общие положения",
                type=SectionType.OBSHCHIE,
                atoms=(_atom(head),),
            ),
            ParsedSection(
                no="2",
                title="Ответственность",
                type=SectionType.OTVETSTVENNOST,
                atoms=(other,),
            ),
        ]
    )

    bodies = _bodies(chunk_document(document))

    assert not any(body.startswith("оформляются") for body in bodies)


def test_a_real_section_still_starts_its_own_fragment() -> None:
    """Настоящий раздел не приклеивается к предыдущему.

    Склейка ловит только строчную букву в начале. Заголовок раздела
    начинается с прописной или с номера, поэтому под правило не подпадает.
    """
    head = "В" * 880 + ". Конец предыдущего раздела."
    new_section = Atom(
        clause="2.1",
        text="Порядок проведения работ устанавливается настоящим разделом.",
        section_no="2",
        section_title="Порядок проведения",
        section_type=SectionType.PROCEDURA,
        is_table_region=False,
    )
    document = _document(
        [
            ParsedSection(
                no="1",
                title="Общие положения",
                type=SectionType.OBSHCHIE,
                atoms=(_atom(head),),
            ),
            ParsedSection(
                no="2",
                title="Порядок проведения",
                type=SectionType.PROCEDURA,
                atoms=(new_section,),
            ),
        ]
    )

    bodies = _bodies(chunk_document(document))

    assert any(body.startswith("Порядок проведения работ") for body in bodies)


def test_table_of_contents_lines_do_not_reach_the_index() -> None:
    """Строки оглавления с отточиями отбрасываются.

    Оглавление исключается по типу раздела, но тип распознаётся не всегда:
    в части документов оно попадает в ``prochee`` и занимает места в выдаче,
    не неся ответа.
    """
    document = _document(
        [
            ParsedSection(
                no="0",
                title="Содержание",
                type=SectionType.PROCHEE,
                atoms=(
                    _atom("Порядок и нормы хранения......................... 7"),
                    _atom("Требования к путям эвакуации..................... 12"),
                    _atom(
                        "Настоящая Инструкция устанавливает требования "
                        "пожарной безопасности на объекте защиты."
                    ),
                ),
            )
        ]
    )

    bodies = _bodies(chunk_document(document))

    assert not any("......" in body for body in bodies)
    assert any("устанавливает требования" in body for body in bodies)


def test_gluing_respects_the_hard_limit() -> None:
    """Склейка не раздувает фрагмент выше предела.

    Иначе длинная цепочка продолжений собралась бы в один огромный фрагмент,
    а он размывает смысл ровно так же, как и обрывок.
    """
    head = "Г" * 1580 + "."
    tail = "продолжение фразы, которое уже не помещается в предел."
    document = _document(
        [
            ParsedSection(
                no="1",
                title="Общие положения",
                type=SectionType.OBSHCHIE,
                atoms=(_atom(head), _atom(tail)),
            )
        ]
    )

    chunks = chunk_document(document)

    assert all(len(c.text_raw) <= 1600 for c in chunks)
    assert len(chunks) == 2, "склейка не должна нарушать предел объёма"
