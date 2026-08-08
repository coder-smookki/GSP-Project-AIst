"""Оркестрация индексации: источник -> разбор -> чанки -> Qdrant.

Устройство прогона:

1. создаётся коллекция-поколение и payload-индексы (до первой точки);
2. документы обрабатываются по одному: разбор, деанонимизация, чанкинг,
   эмбеддинги, загрузка, уборка устаревших точек этого документа;
3. после успешного прохода алиас переключается на новое поколение.

Порядок «сначала записать, потом подмести» выбран намеренно: между шагами
индекс содержит и старые, и новые точки, то есть поиск продолжает работать.

Одна битая статья не роняет прогон: ошибки собираются в отчёт, а код
возврата у CLI становится ненулевым. Молча пропустить документ нельзя —
пользователь должен знать, что база неполна.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from infrastructure.ingest.areas import classify_areas
from infrastructure.ingest.chunker import chunk_document
from infrastructure.ingest.deanon import deanonymize
from infrastructure.ingest.html_parse import parse_article
from infrastructure.ingest.models import ParsedDocument
from infrastructure.retrieval.source import DataSource, RawDocument

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentOutcome:
    """Итог обработки одного документа.

    Args:
        doc_id: Идентификатор документа в источнике.
        title: Название документа.
        doc_code: Код документа для цитат.
        chunks: Число загруженных чанков.
        areas: Определённые предметные области.
        error: Текст ошибки, если документ обработать не удалось.
    """

    doc_id: str
    title: str = ""
    doc_code: str = ""
    chunks: int = 0
    areas: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Признак успешной обработки.

        Returns:
            bool: ``True``, если ошибки не было.
        """
        return self.error is None


@dataclass(slots=True)
class IngestReport:
    """Отчёт о прогоне индексации.

    Args:
        collection: Имя коллекции-поколения, в которую шла загрузка.
        outcomes: Итоги по каждому документу.
        switched_from: Прежняя коллекция, если алиас переключался.
    """

    collection: str
    outcomes: list[DocumentOutcome] = field(default_factory=list)
    switched_from: str | None = None

    @property
    def total_chunks(self) -> int:
        """Возвращает общее число загруженных чанков.

        Returns:
            int: Сумма по всем документам.
        """
        return sum(item.chunks for item in self.outcomes)

    @property
    def failed(self) -> list[DocumentOutcome]:
        """Возвращает документы, которые обработать не удалось.

        Returns:
            list[DocumentOutcome]: Список неудач; пустой при полном успехе.
        """
        return [item for item in self.outcomes if not item.ok]

    @property
    def areas_histogram(self) -> dict[str, int]:
        """Считает распределение документов по областям знаний.

        Печать этого распределения обязательна: документ, не отнесённый ни к
        одной области, должен быть виден, а не раствориться в общей куче.

        Returns:
            dict[str, int]: Отображение ``код области -> число документов``.
        """
        histogram: dict[str, int] = {}
        for item in self.outcomes:
            for area in item.areas:
                histogram[area] = histogram.get(area, 0) + 1
        return histogram


def parse_raw(document: RawDocument) -> ParsedDocument:
    """Разбирает сырой документ подходящим парсером.

    Args:
        document: Документ, полученный от источника.

    Returns:
        ParsedDocument: Разобранный документ с заполненным словарём
        деанонимизации.

    Raises:
        NotImplementedError: Если формат пока не поддержан. Разбор .docx
            нужен только для файлового источника и пишется отдельно.
        ValueError: Если содержимое пустое или без идентификатора.
    """
    if document.kind == "html":
        article = {
            "id": document.doc_id,
            "name": document.title,
            "content": document.content,
            "section_id": document.section_id,
            "updated_at": document.updated_at,
        }
        return deanonymize(parse_article(article, source="api"))
    raise NotImplementedError(
        f"разбор формата {document.kind!r} пока не реализован "
        "(документ {document.doc_id})"
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
) -> IngestReport:
    """Прогоняет корпус источника через конвейер в коллекцию Qdrant.

    Args:
        source: Открытый источник корпуса.
        store: Хранилище `infrastructure.retrieval.store.QdrantStore`.
            При `dry_run` не используется.
        embedder: Клиент эмбеддингов с методом ``aembed(texts)``.
            При `dry_run` не используется.
        collection: Имя коллекции-поколения.
        sync_tag: Метка прогона для уборки устаревших точек.
        limit: Ограничение числа документов; ``None`` — все.
        dry_run: Прогон без записи: разбор и чанкинг выполняются, обращения
            к Qdrant и модели эмбеддингов не делаются. Нужен, чтобы проверить
            конвейер на корпусе, не поднимая инфраструктуру.
        progress: Необязательный обратный вызов ``(название, номер, всего)``
            для отображения хода работы.

    Returns:
        IngestReport: Отчёт с итогами по каждому документу.
    """
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
            # Название берём из разобранного документа: карточка источника
            # для файлов содержит лишь имя файла.
            outcome.title = document.doc_title or outcome.title

            if not chunks:
                outcome.error = "конвейер не дал ни одного чанка"
            elif dry_run:
                outcome.chunks = len(chunks)
            else:
                vectors = await embedder.aembed([chunk.text for chunk in chunks])
                outcome.chunks = await store.upsert_chunks(
                    collection, chunks, vectors, areas, sync_tag
                )
                await store.sweep_document(collection, document.doc_id, sync_tag)
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            logger.exception("документ %s: обработка не удалась", ref.doc_id)
        report.outcomes.append(outcome)

    logger.info(
        "индексация завершена: документов %d, чанков %d, ошибок %d",
        total,
        report.total_chunks,
        len(report.failed),
    )
    return report


def format_report(report: IngestReport, areas_title: Callable[[str], str]) -> str:
    """Собирает человекочитаемый отчёт о прогоне.

    Args:
        report: Отчёт прогона.
        areas_title: Функция перевода кода области в название.

    Returns:
        str: Многострочный текст для вывода в консоль.
    """
    lines: list[str] = [
        f"Коллекция: {report.collection}",
        f"Документов: {len(report.outcomes)}, чанков: {report.total_chunks}",
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
