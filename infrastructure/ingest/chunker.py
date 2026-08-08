"""Сборка чанков индексации из атомов документа.

Правила чанкинга продиктованы форматом регламента, а не общими соображениями:

* **атом = пункт** вместе со своим маркированным списком: разрывать пункт
  нельзя, иначе цитата «Р-У2-59, п. 4.3» перестаёт соответствовать тексту;
* **соседние короткие атомы склеиваются** до `TARGET_CHARS`: одиночный пункт
  на 60 символов не несёт контекста и шумит в выдаче;
* **overlap отсутствует**. Вместо перекрытия текста используется
  `breadcrumb` — путь «Документ > Раздел > Пункт» в начале чанка. Overlap
  дублирует содержимое в индексе и завышает счёт документа, у которого
  совпадение попало в зону перекрытия;
* **оглавление не индексируется**: оно дублирует заголовки разделов и даёт
  попадание почти на любой запрос по названию раздела.
"""

from __future__ import annotations

import logging

from infrastructure.ingest.deanon import restore_text
from infrastructure.ingest.models import (
    NOT_INDEXED,
    Atom,
    IngestChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

#: Целевой размер чанка в символах. Ориентир — медиана атома корпуса (~200
#: символов) и бюджет контекста: 8 чанков по 900 символов это ~2700 токенов,
#: что оставляет место под системный промпт и историю диалога.
TARGET_CHARS = 900

#: Жёсткий предел размера чанка. Атомы длиннее уже разрезаны парсером.
MAX_CHARS = 1_600

#: Минимальная длина чанка. Более короткие фрагменты (подписи, номера
#: страниц) не несут смысла и только шумят в выдаче.
MIN_CHARS = 40


def _breadcrumb(document: ParsedDocument, atom: Atom) -> str:
    """Строит навигационный путь чанка.

    Путь идёт в начало индексируемого текста и заменяет overlap: он даёт
    эмбеддингу и BM25 контекст раздела, не дублируя соседние чанки.

    Args:
        document: Документ, которому принадлежит атом.
        atom: Атом.

    Returns:
        str: Строка вида ``Р-У2-59 > 4. Порядок проведения > п. 4.1.4``.
    """
    parts = [document.doc_code]
    if atom.section_no or atom.section_title:
        head = f"{atom.section_no}. {atom.section_title}".strip(". ")
        parts.append(head)
    if atom.clause and not atom.is_table_region:
        parts.append(f"п. {atom.clause}")
    return " > ".join(p for p in parts if p)


def _mergeable(left: Atom, right: Atom) -> bool:
    """Проверяет, можно ли склеить два соседних атома в один чанк.

    Args:
        left: Предыдущий атом.
        right: Следующий атом.

    Returns:
        bool: ``True``, если атомы из одного раздела и одной зоны
        (табличной или обычной). Пункты из разных разделов не склеиваются:
        цитата стала бы неоднозначной.
    """
    return (
        left.section_no == right.section_no
        and left.section_type == right.section_type
        and left.is_table_region == right.is_table_region
    )


def chunk_document(
    document: ParsedDocument,
    *,
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[IngestChunk]:
    """Превращает разобранный документ в список чанков для индексации.

    Args:
        document: Документ после разбора и деанонимизации. Поле ``glossary``
            должно быть заполнено, иначе плейсхолдеры будут замаскированы.
        target_chars: Целевой размер чанка в символах.
        max_chars: Жёсткий предел размера чанка.
        min_chars: Минимальный размер, ниже которого чанк отбрасывается,
            если его не удалось ни с чем склеить.

    Returns:
        list[IngestChunk]: Чанки в порядке следования по документу.
        Идентификаторы стабильны между прогонами при неизменном входе.

    Raises:
        ValueError: Если `target_chars` больше `max_chars` — такая
            конфигурация молча породила бы чанки сверх предела.
    """
    if target_chars > max_chars:
        raise ValueError(
            f"target_chars ({target_chars}) не может превышать max_chars ({max_chars})"
        )

    groups: list[list[Atom]] = []
    for section in document.sections:
        if section.type in NOT_INDEXED:
            continue
        for atom in section.atoms:
            if groups and _mergeable(groups[-1][-1], atom):
                current_len = sum(len(a.text) + 1 for a in groups[-1])
                if current_len + len(atom.text) <= target_chars:
                    groups[-1].append(atom)
                    continue
            groups.append([atom])

    chunks: list[IngestChunk] = []
    dropped = 0
    for order, group in enumerate(groups, start=1):
        head = group[0]
        raw_text = " ".join(a.text for a in group).strip()
        if len(raw_text) > max_chars:
            raw_text = raw_text[:max_chars].rsplit(" ", 1)[0]

        # Восстановление идёт по СКЛЕЕННОМУ тексту вместе с breadcrumb:
        # заголовок раздела тоже содержит метки анонимизации, и без этого
        # они утекали бы в индекс и в панель источников через путь чанка.
        raw_breadcrumb = _breadcrumb(document, head)
        combined, restored, unresolved = restore_text(
            f"{raw_breadcrumb}\n{raw_text}", document.glossary
        )
        breadcrumb, _, text = combined.partition("\n")
        if len(text) < min_chars:
            dropped += 1
            continue

        # Предел считается по ИТОГОВОМУ индексируемому тексту: breadcrumb
        # добавляет длину, а подстановка расшифровок делает текст длиннее
        # исходного (маска и расшифровка длиннее метки).
        indexed_text = combined
        if len(indexed_text) > max_chars:
            indexed_text = indexed_text[:max_chars].rsplit(" ", 1)[0]

        chunk_id = f"{document.doc_code}#c{order:04d}"
        chunks.append(
            IngestChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                doc_code=document.doc_code,
                doc_title=document.doc_title,
                section_no=head.section_no,
                section_title=head.section_title,
                section_type=head.section_type,
                # В табличной зоне номер пункта недостоверен: цитата
                # деградирует до уровня раздела (правило §5.3 спецификации).
                clause=None if head.is_table_region else head.clause,
                breadcrumb=breadcrumb,
                text=indexed_text,
                text_raw=raw_text,
                aliases=tuple(dict.fromkeys(restored)),
                placeholders=tuple(dict.fromkeys(unresolved)),
                restored=bool(restored),
                is_table_region=head.is_table_region,
                order_no=order,
            )
        )

    logger.info(
        "документ %s (%s): чанков %d, отброшено коротких %d, "
        "с восстановленными терминами %d",
        document.doc_id,
        document.doc_code,
        len(chunks),
        dropped,
        sum(1 for c in chunks if c.restored),
    )
    return chunks
