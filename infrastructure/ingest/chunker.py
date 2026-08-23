from __future__ import annotations

import logging
import re

from infrastructure.ingest.config import DEFAULT_CHUNK, ChunkingConfig
from infrastructure.ingest.deanon import restore_text
from infrastructure.ingest.models import (
    NOT_INDEXED,
    Atom,
    IngestChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

#: Атом начинается с продолжения фразы, а не с её начала.
#:
#: Такой фрагмент, оторванный от начала предложения, описывает не то, о чём
#: документ: у самоспасателя из ДОК-8 в отдельный фрагмент уехал хвост
#: «различного назначения (жилых, промышленных...)», а начало - для чего
#: средство и когда применяется - осталось в соседнем. Вектор такого хвоста
#: не отвечает ни на один осмысленный вопрос.
_CONTINUATION = re.compile(r"^[а-яё(,;]")

#: Строка оглавления: отточия перед номером страницы.
#:
#: Оглавление исключается по типу раздела (``NOT_INDEXED``), но распознаётся
#: тип не всегда: в части документов оглавление попадает в ``prochee`` и
#: занимает места в выдаче, не неся ответа.
_LEADER = re.compile(r"\.{6,}")

#: Доля отточий, при которой текст считается оглавлением, а не прозой.
_LEADER_SHARE = 0.12

TARGET_CHARS = DEFAULT_CHUNK.target_chars
MAX_CHARS = DEFAULT_CHUNK.max_chars
MIN_CHARS = DEFAULT_CHUNK.min_chars


def _breadcrumb(document: ParsedDocument, atom: Atom) -> str:
    parts = [document.doc_code]
    if atom.section_no or atom.section_title:
        head = f"{atom.section_no}. {atom.section_title}".strip(". ")
        parts.append(head)
    if atom.clause and not atom.is_table_region:
        parts.append(f"п. {atom.clause}")
    return " > ".join(p for p in parts if p)


def _looks_like_contents(text: str) -> bool:
    """Определяет строку оглавления по отточиям.

    Args:
        text: Текст атома.

    Returns:
        bool: ``True``, если отточия занимают заметную долю строки. По
        заголовку раздела оглавление ловится не всегда: в части документов
        оно попадает в ``prochee`` и уходит в индекс.
    """
    if not text:
        return False
    dots = sum(len(m.group(0)) for m in _LEADER.finditer(text))
    return dots / len(text) >= _LEADER_SHARE


def _mergeable(left: Atom, right: Atom) -> bool:
    return (
        left.section_no == right.section_no
        and left.section_type == right.section_type
        and left.is_table_region == right.is_table_region
    )


class Chunker:
    def __init__(self, config: ChunkingConfig = DEFAULT_CHUNK) -> None:
        self.config = config

    def chunk(self, document: ParsedDocument) -> list[IngestChunk]:
        target_chars = self.config.target_chars
        max_chars = self.config.max_chars
        min_chars = self.config.min_chars
        if target_chars > max_chars:
            raise ValueError(
                f"target_chars ({target_chars}) не может превышать "
                f"max_chars ({max_chars})"
            )

        groups: list[list[Atom]] = []
        glued = 0
        skipped_contents = 0
        for section in document.sections:
            if section.type in NOT_INDEXED:
                continue
            for atom in section.atoms:
                if _looks_like_contents(atom.text):
                    skipped_contents += 1
                    continue
                if groups:
                    current_len = sum(len(a.text) + 1 for a in groups[-1])
                    if (
                        _mergeable(groups[-1][-1], atom)
                        and current_len + len(atom.text) <= target_chars
                    ):
                        groups[-1].append(atom)
                        continue
                    # Продолжение прирастает к предыдущей группе даже сверх
                    # целевого объёма и ДАЖЕ через границу раздела.
                    #
                    # Через границу - потому что текст со строчной буквы это
                    # продолжение предыдущего предложения по определению, а
                    # классификатор типов разделов иногда разрывает сплошной
                    # абзац надвое и приписывает половинам разные типы. Верить
                    # тут надо тексту, а не разметке. Настоящий раздел так не
                    # начинается: у него заголовок с прописной или с номера.
                    if _CONTINUATION.match(atom.text) and (
                        current_len + len(atom.text) <= max_chars
                    ):
                        groups[-1].append(atom)
                        glued += 1
                        continue
                groups.append([atom])

        chunks: list[IngestChunk] = []
        dropped = 0
        for order, group in enumerate(groups, start=1):
            head = group[0]
            raw_text = " ".join(a.text for a in group).strip()
            if len(raw_text) > max_chars:
                raw_text = raw_text[:max_chars].rsplit(" ", 1)[0]

            raw_breadcrumb = _breadcrumb(document, head)
            combined, restored, unresolved = restore_text(
                f"{raw_breadcrumb}\n{raw_text}", document.glossary
            )
            breadcrumb, _, text = combined.partition("\n")
            if len(text) < min_chars:
                dropped += 1
                continue

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
            "строк оглавления %d, приклеено продолжений %d, "
            "с восстановленными терминами %d",
            document.doc_id,
            document.doc_code,
            len(chunks),
            dropped,
            skipped_contents,
            glued,
            sum(1 for c in chunks if c.restored),
        )
        return chunks


def chunk_document(
    document: ParsedDocument,
    *,
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[IngestChunk]:
    config = ChunkingConfig(
        target_chars=target_chars, max_chars=max_chars, min_chars=min_chars
    )
    return Chunker(config).chunk(document)
