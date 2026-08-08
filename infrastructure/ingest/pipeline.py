from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from infrastructure.ingest.areas import classify_areas
from infrastructure.ingest.chunker import chunk_document
from infrastructure.ingest.deanon import deanonymize
from infrastructure.ingest.docx_parse import parse_docx
from infrastructure.ingest.html_parse import parse_article
from infrastructure.ingest.incremental import document_hash
from infrastructure.ingest.models import ParsedDocument
from infrastructure.retrieval.source import DataSource, RawDocument

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "2026.08.08-1"


@dataclass(slots=True)
class DocumentOutcome:
    doc_id: str
    title: str = ""
    doc_code: str = ""
    chunks: int = 0
    areas: list[str] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class IngestReport:
    collection: str
    outcomes: list[DocumentOutcome] = field(default_factory=list)
    switched_from: str | None = None

    @property
    def total_chunks(self) -> int:
        return sum(item.chunks for item in self.outcomes)

    @property
    def failed(self) -> list[DocumentOutcome]:
        return [item for item in self.outcomes if not item.ok]

    @property
    def areas_histogram(self) -> dict[str, int]:
        histogram: dict[str, int] = {}
        for item in self.outcomes:
            for area in item.areas:
                histogram[area] = histogram.get(area, 0) + 1
        return histogram


def parse_raw(document: RawDocument) -> ParsedDocument:
    if document.kind == "html":
        article = {
            "id": document.doc_id,
            "name": document.title,
            "content": document.content,
            "section_id": document.section_id,
            "updated_at": document.updated_at,
        }
        return deanonymize(parse_article(article, source="api"))

    if document.kind == "docx":
        return deanonymize(parse_docx(document.content, source="files"))

    raise ValueError(
        f"документ {document.doc_id}: неизвестный формат {document.kind!r}"
    )


async def ingest_documents(
    source: DataSource,
    store,
    embedder,
    *,
    collection: str,
    sync_tag: str,
    limit: int | None = None,
    dry_run: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
    known_hashes: dict[str, str] | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> IngestReport:
    refs = await source.list_documents()
    if limit is not None:
        refs = refs[:limit]

    report = IngestReport(collection=collection)
    total = len(refs)
    logger.info(
        "начата индексация: документов %d, коллекция %s, режим %s",
        total,
        collection,
        "проверка без записи" if dry_run else "запись",
    )

    for number, ref in enumerate(refs, start=1):
        if progress:
            progress(ref.title or ref.doc_id, number, total)
        outcome = DocumentOutcome(doc_id=ref.doc_id, title=ref.title)
        try:
            raw = await source.fetch_document(ref.doc_id)
            document = parse_raw(raw)
            chunks = chunk_document(document)
            areas = classify_areas(document.doc_title)

            outcome.doc_code = document.doc_code
            outcome.areas = areas
            outcome.title = document.doc_title or outcome.title

            fingerprint = document_hash(document, pipeline_version)

            if not chunks:
                outcome.error = "конвейер не дал ни одного чанка"
            elif (
                known_hashes is not None
                and known_hashes.get(document.doc_id) == fingerprint
            ):
                outcome.skipped = True
                outcome.chunks = 0
            elif dry_run:
                outcome.chunks = len(chunks)
            else:
                vectors = await embedder.aembed([chunk.text for chunk in chunks])
                outcome.chunks = await store.upsert_chunks(
                    collection, chunks, vectors, areas, sync_tag, fingerprint
                )
                await store.sweep_document(collection, document.doc_id, sync_tag)
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            logger.exception("документ %s: обработка не удалась", ref.doc_id)
        report.outcomes.append(outcome)

    logger.info(
        "индексация завершена: документов %d, чанков %d, пропущено без "
        "изменений %d, ошибок %d",
        total,
        report.total_chunks,
        sum(1 for item in report.outcomes if item.skipped),
        len(report.failed),
    )
    return report


def format_report(report: IngestReport, areas_title: Callable[[str], str]) -> str:
    skipped = sum(1 for item in report.outcomes if item.skipped)
    lines: list[str] = [
        f"Коллекция: {report.collection}",
        f"Документов: {len(report.outcomes)}, чанков: {report.total_chunks}"
        + (f", пропущено без изменений: {skipped}" if skipped else ""),
    ]
    if report.switched_from is not None:
        lines.append(f"Алиас переключён, прежняя коллекция: {report.switched_from}")

    histogram = report.areas_histogram
    if histogram:
        lines.append("Распределение по областям знаний:")
        for code, count in sorted(histogram.items(), key=lambda item: -item[1]):
            lines.append(f"  {count:>3}  {areas_title(code)}")

    unclassified: Sequence[DocumentOutcome] = [
        item for item in report.outcomes if item.areas == ["prochee"]
    ]
    if unclassified:
        lines.append("Не отнесены ни к одной области:")
        lines.extend(f"  - {item.title}" for item in unclassified)

    if report.failed:
        lines.append(f"Ошибки ({len(report.failed)}):")
        lines.extend(
            f"  - {item.doc_id} {item.title}: {item.error}" for item in report.failed
        )
    else:
        lines.append("Ошибок нет")
    return "\n".join(lines)
