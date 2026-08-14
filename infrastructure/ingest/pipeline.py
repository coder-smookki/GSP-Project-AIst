from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from infrastructure.ingest.areas import AreaClassifier
from infrastructure.ingest.chunker import Chunker
from infrastructure.ingest.config import (
    DEFAULT_CONFIG,
    PIPELINE_VERSION,
    IncrementalConfig,
    IngestConfig,
)
from infrastructure.ingest.deanon import Deanonymizer
from infrastructure.ingest.docx_parse import DocxParser
from infrastructure.ingest.html_parse import HtmlParser
from infrastructure.ingest.incremental import HashRepository
from infrastructure.ingest.models import ParsedDocument
from infrastructure.ingest.protocols import (
    Classifier,
    DocumentParser,
    VectorStore,
)
from infrastructure.ingest.protocols import (
    Deanonymizer as DeanonymizerProtocol,
)
from infrastructure.retrieval.source import DataSource, RawDocument

logger = logging.getLogger(__name__)


class _Embedder(Protocol):
    async def aembed(self, texts: list[str]) -> list[list[float]]: ...


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


class IngestPipeline:
    def __init__(
        self,
        *,
        config: IngestConfig = DEFAULT_CONFIG,
        parsers: dict[str, DocumentParser] | None = None,
        deanonymizer: DeanonymizerProtocol | None = None,
        chunker: Chunker | None = None,
        classifier: Classifier | None = None,
        repository: HashRepository | None = None,
    ) -> None:
        self.config = config
        self.parsers = (
            parsers
            if parsers is not None
            else {
                "html": HtmlParser(config.parsing),
                "docx": DocxParser(config.docx),
            }
        )
        self.deanonymizer = deanonymizer or Deanonymizer(config.deanonymization)
        self.chunker = chunker or Chunker(config.chunking)
        self.classifier = classifier or AreaClassifier(config.area)
        self.repository = repository or HashRepository(config.incremental)

    def parse_raw(self, document: RawDocument) -> ParsedDocument:
        parser = self.parsers.get(document.kind)
        if parser is None:
            raise ValueError(
                f"документ {document.doc_id}: неизвестный формат {document.kind!r}"
            )
        parsed = parser.parse(document)
        return self.deanonymizer.process(parsed)

    async def ingest(
        self,
        source: DataSource,
        store: VectorStore | None,
        embedder: _Embedder | None,
        *,
        collection: str,
        sync_tag: str,
        limit: int | None = None,
        dry_run: bool = False,
        progress: Callable[[str, int, int], None] | None = None,
        known_hashes: dict[str, str] | None = None,
        step: Callable[..., Awaitable[DocumentOutcome]] | None = None,
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
            if step is not None:
                outcome = await step(
                    self,
                    source,
                    store,
                    embedder,
                    ref,
                    collection=collection,
                    sync_tag=sync_tag,
                    dry_run=dry_run,
                    known_hashes=known_hashes,
                )
            else:
                outcome = await self.process_document(
                    source,
                    store,
                    embedder,
                    ref,
                    collection=collection,
                    sync_tag=sync_tag,
                    dry_run=dry_run,
                    known_hashes=known_hashes,
                )
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

    async def process_document(
        self,
        source: DataSource,
        store: VectorStore | None,
        embedder: _Embedder | None,
        ref,
        *,
        collection: str,
        sync_tag: str,
        dry_run: bool = False,
        known_hashes: dict[str, str] | None = None,
    ) -> DocumentOutcome:
        outcome = DocumentOutcome(doc_id=ref.doc_id, title=ref.title)
        try:
            raw = await source.fetch_document(ref.doc_id)
            document = self.parse_raw(raw)
            chunks = self.chunker.chunk(document)
            areas = self.classifier.classify(document.doc_title)

            outcome.doc_code = document.doc_code
            outcome.areas = areas
            outcome.title = document.doc_title or outcome.title

            fingerprint = self.repository.fingerprint(document)

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
            elif store is None or embedder is None:
                outcome.error = "нет индекса или эмбеддера для записи"
            else:
                vectors = await embedder.aembed([chunk.text for chunk in chunks])
                outcome.chunks = await store.upsert_chunks(
                    collection, chunks, vectors, areas, sync_tag, fingerprint
                )
                await store.sweep_document(collection, document.doc_id, sync_tag)
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            logger.exception("документ %s: обработка не удалась", ref.doc_id)
        return outcome


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
    config = replace(
        DEFAULT_CONFIG,
        incremental=IncrementalConfig(pipeline_version=pipeline_version),
    )
    pipeline = IngestPipeline(config=config)
    return await pipeline.ingest(
        source,
        store,
        embedder,
        collection=collection,
        sync_tag=sync_tag,
        limit=limit,
        dry_run=dry_run,
        progress=progress,
        known_hashes=known_hashes,
    )


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


def parse_raw(document: RawDocument) -> ParsedDocument:
    return IngestPipeline().parse_raw(document)
