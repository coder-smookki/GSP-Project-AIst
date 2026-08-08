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

TARGET_CHARS = 900
MAX_CHARS = 1_600
MIN_CHARS = 40


def _breadcrumb(document: ParsedDocument, atom: Atom) -> str:
    parts = [document.doc_code]
    if atom.section_no or atom.section_title:
        head = f"{atom.section_no}. {atom.section_title}".strip(". ")
        parts.append(head)
    if atom.clause and not atom.is_table_region:
        parts.append(f"п. {atom.clause}")
    return " > ".join(p for p in parts if p)


def _mergeable(left: Atom, right: Atom) -> bool:
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
        "с восстановленными терминами %d",
        document.doc_id,
        document.doc_code,
        len(chunks),
        dropped,
        sum(1 for c in chunks if c.restored),
    )
    return chunks
