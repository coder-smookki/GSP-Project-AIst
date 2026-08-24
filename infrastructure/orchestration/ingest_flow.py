from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import replace

from infrastructure.ingest.config import (
    DEFAULT_CONFIG,
    PIPELINE_VERSION,
    IncrementalConfig,
)
from infrastructure.ingest.models import IngestChunk, ParsedDocument
from infrastructure.ingest.pipeline import (
    DocumentOutcome,
    IngestPipeline,
    IngestReport,
)
from infrastructure.retrieval.source import (
    ApiSource,
    DataSource,
    DocumentRef,
    FilesSource,
)

logger = logging.getLogger("aist.ingest")

try:
    from prefect import flow, task  # type: ignore[import-not-found]
    from prefect.cache_policies import NO_CACHE  # type: ignore[import-not-found]
    from prefect.logging import get_run_logger  # type: ignore[import-not-found]
except ImportError:
    NO_CACHE = None

    def flow(func=None, **kwargs):
        return func if func is not None else (lambda fn: fn)

    def task(func=None, **kwargs):
        return func if func is not None else (lambda fn: fn)

    def get_run_logger():
        return logging.getLogger("aist.ingest")


def _document_ref_id(
    parameters: dict[str, object] | None = None,
    **_kwargs: object,
) -> str:
    params = parameters or {}
    ref = params.get("ref")
    doc_id = getattr(ref, "doc_id", "?") if ref is not None else "?"
    return f"документ {doc_id}"


def _phase_ref_id(label: str) -> Callable[[dict[str, object] | None], str]:
    def _name(parameters: dict[str, object] | None = None, **_kwargs: object) -> str:
        params = parameters or {}
        ref = params.get("ref")
        doc_id = getattr(ref, "doc_id", "?") if ref is not None else "?"
        return f"{label}: {doc_id}"

    return _name


@task(
    name="Загрузка и разбор",
    task_run_name=_phase_ref_id("загрузка и разбор"),
    log_prints=True,
    cache_policy=NO_CACHE,
)
async def fetch_and_parse_step(
    pipeline: IngestPipeline,
    source: DataSource,
    ref: DocumentRef,
) -> ParsedDocument:
    run_logger = get_run_logger()
    run_logger.info("документ %s: загрузка исходника", ref.doc_id)
    raw = await source.fetch_document(ref.doc_id)
    document = pipeline.parse_raw(raw)
    run_logger.info(
        "документ %s: разобран, разделов %d", ref.doc_id, len(document.sections)
    )
    return document


@task(
    name="Разбиение на чанки",
    task_run_name=_phase_ref_id("разбиение на чанки"),
    log_prints=True,
    cache_policy=NO_CACHE,
)
async def chunk_step(
    pipeline: IngestPipeline,
    document: ParsedDocument,
    ref: DocumentRef,
) -> list[IngestChunk]:
    run_logger = get_run_logger()
    chunks = pipeline.chunker.chunk(document)
    run_logger.info("документ %s: чанков %d", ref.doc_id, len(chunks))
    return chunks


@task(
    name="Классификация областей",
    task_run_name=_phase_ref_id("классификация областей"),
    log_prints=True,
    cache_policy=NO_CACHE,
)
async def classify_step(
    pipeline: IngestPipeline,
    document: ParsedDocument,
    ref: DocumentRef,
) -> list[str]:
    run_logger = get_run_logger()
    areas = pipeline.classifier.classify(document.doc_title)
    run_logger.info("документ %s: области %s", ref.doc_id, ", ".join(areas))
    return areas


@task(
    name="QA-обогащение",
    task_run_name=_phase_ref_id("QA-обогащение"),
    log_prints=True,
    cache_policy=NO_CACHE,
)
async def qa_enrich_step(
    pipeline: IngestPipeline,
    chunks: list[IngestChunk],
    ref: DocumentRef,
) -> None:
    from infrastructure.ingest import qa_index

    if pipeline.qa_model is None or not chunks:
        return
    run_logger = get_run_logger()
    stats = await qa_index.enrich(chunks, pipeline.qa_model)
    run_logger.info(
        "документ %s: QA-обогащение %d/%d чанков",
        ref.doc_id,
        stats.enriched,
        stats.considered,
    )


@task(
    name="Эмбеддинги",
    task_run_name=_phase_ref_id("эмбеддинги"),
    log_prints=True,
    cache_policy=NO_CACHE,
)
async def embed_step(
    embedder,
    chunks: list[IngestChunk],
    ref: DocumentRef,
) -> list[list[float]]:
    run_logger = get_run_logger()
    # Кодируется text_indexed, если он заполнен: при QA-индексации там
    # фрагмент вместе с вопросами, на которые он отвечает. Показывается
    # по-прежнему text (см. infrastructure/ingest/qa_index.py).
    vectors = await embedder.aembed(
        [chunk.text_indexed or chunk.text for chunk in chunks]
    )
    run_logger.info("документ %s: эмбеддингов %d", ref.doc_id, len(vectors))
    return vectors


@task(
    name="Запись в Qdrant",
    task_run_name=_phase_ref_id("запись"),
    log_prints=True,
    cache_policy=NO_CACHE,
)
async def upsert_step(
    store,
    collection: str,
    sync_tag: str,
    chunks: list[IngestChunk],
    vectors: list[list[float]],
    areas: list[str],
    fingerprint: str,
    ref: DocumentRef,
) -> int:
    run_logger = get_run_logger()
    written = await store.upsert_chunks(
        collection, chunks, vectors, areas, sync_tag, fingerprint
    )
    await store.sweep_document(collection, ref.doc_id, sync_tag)
    run_logger.info("документ %s: записано %d", ref.doc_id, written)
    return written


@task(
    name="Индексация документа",
    task_run_name=_document_ref_id,
    log_prints=True,
    cache_policy=NO_CACHE,
)
async def ingest_document_step(
    pipeline: IngestPipeline,
    source: DataSource,
    store,
    embedder,
    ref: DocumentRef,
    *,
    collection: str,
    sync_tag: str,
    dry_run: bool,
    known_hashes: dict[str, str] | None,
) -> DocumentOutcome:
    run_logger = get_run_logger()
    run_logger.info("документ %s: начало обработки", ref.doc_id)
    outcome = DocumentOutcome(doc_id=ref.doc_id, title=ref.title)
    try:
        document = await fetch_and_parse_step(pipeline, source, ref)
        outcome.doc_code = document.doc_code
        outcome.title = document.doc_title or outcome.title

        chunks = await chunk_step(pipeline, document, ref)
        # Имя модели входит в отпечаток: векторы разных моделей несовместимы
        # по смыслу даже при равной размерности. Без него смена эмбеддера
        # проходила как «документ не изменился», индекс оставался собранным
        # прежней моделью, и качество поиска молча падало.
        embedder_id = str(getattr(embedder, "model_name", "") or "")
        if pipeline.qa_model is not None:
            # Как в основном пути (pipeline.py): отпечаток обязан отличать
            # обычный прогон от QA, иначе повторная индексация с вопросами
            # сочла бы документы неизменившимися и не переписала бы ни
            # одного вектора.
            embedder_id = f"{embedder_id}+qa"
        fingerprint = pipeline.repository.fingerprint(document, embedder_id)

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
            areas = await classify_step(pipeline, document, ref)
            if pipeline.qa_model is not None:
                await qa_enrich_step(pipeline, chunks, ref)
            vectors = await embed_step(embedder, chunks, ref)
            outcome.chunks = await upsert_step(
                store,
                collection,
                sync_tag,
                chunks,
                vectors,
                areas,
                fingerprint,
                ref,
            )
            outcome.areas = areas
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        logger.exception("документ %s: обработка не удалась", ref.doc_id)

    if outcome.ok:
        run_logger.info("документ %s: готово, чанков %d", ref.doc_id, outcome.chunks)
    else:
        run_logger.error("документ %s: %s", ref.doc_id, outcome.error)
    return outcome


def build_source(
    settings,
    source_kind: str,
    corpus_dir: str | None,
    section_id: int | None,
) -> DataSource:
    if source_kind == "files":
        return FilesSource(corpus_dir or settings.corpus_dir)

    from infrastructure.retrieval.kb_client import KnowledgeBaseConfig

    if not settings.kb_email or not settings.kb_password:
        raise ValueError(
            "не заданы учётные данные базы знаний: задайте KB_EMAIL и KB_PASSWORD "
            "в окружении или в .env"
        )
    return ApiSource(
        KnowledgeBaseConfig(
            base_url=settings.kb_base_url,
            email=settings.kb_email,
            password=settings.kb_password,
            section_id=section_id or settings.kb_section_id,
            token_name=settings.kb_token_name,
            per_page=settings.kb_per_page,
            max_retries=settings.kb_max_retries,
        ),
        offline=settings.aist_offline,
    )


def build_index(
    settings,
    collection: str,
    batch_size: int,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_query_instruction: str | None = None,
):
    from qdrant_client import AsyncQdrantClient

    from infrastructure.config.model_registry import build_embedding_client
    from infrastructure.retrieval.store import QdrantStore

    # Создаём ТОЛЬКО эмбеддер (LLM клиенты для индексации не нужны).
    # Registry не задействуем: в одном процессе раннера живут несколько
    # deployment-ов, и глобальный слот "default" привёл бы к их взаимному
    # затиранию. Сборщик клиента общий с реестром, чтобы индексация знала
    # ровно тех же провайдеров, что и рантайм. embedding_query_instruction
    # применяется на query-стороне (retrieval/factory.py), при индексации
    # документов он не используется.
    embedder = build_embedding_client(
        settings, model=embedding_model, provider=embedding_provider
    )

    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=int(settings.qdrant_timeout),
    )
    store = QdrantStore(
        client,
        alias=settings.qdrant_collection,
        batch_size=batch_size,
    )
    return client, store, embedder


@flow(
    name="aist-ingest",
    description="Индексация корпуса базы знаний в Qdrant",
    log_prints=True,
)
async def ingest_flow(
    *,
    source_kind: str = "files",
    corpus_dir: str | None = None,
    section_id: int | None = None,
    collection: str | None = None,
    generation: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    full: bool = False,
    qa_index: bool = False,
    switch_alias: bool = False,
    sync_tag: str | None = None,
    pipeline_version: str = PIPELINE_VERSION,
    # NEW: параметры эмбеддингов (override настроек для этого прогона)
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_query_instruction: str | None = None,
    llm_model: str | None = None,
    llm_provider: str | None = None,
) -> IngestReport:
    run_logger = get_run_logger()
    # Настройки кэшируются lru_cache — без сброса долгоживущий раннер
    # продолжал бы жить со старыми LLM_*/EMBEDDING_* после смены .env.
    from infrastructure.config.settings import get_settings as _get_settings

    _get_settings.cache_clear()
    settings = _get_settings()
    # Параметры flow перекрывают окружение для этого прогона (видно в UI).
    if llm_model is not None:
        settings = settings.model_copy(update={"llm_model": llm_model})
    if llm_provider is not None:
        settings = settings.model_copy(update={"llm_provider": llm_provider})
    run_logger.info(
        "настройки моделей: LLM %s/%s (base %s), эмбеддер %s/%s",
        settings.llm_provider,
        settings.llm_model,
        settings.ollama_local_base_url,
        embedding_provider or settings.embedding_provider,
        embedding_model or settings.embedding_model,
    )

    source = build_source(settings, source_kind, corpus_dir, section_id)
    effective_collection = collection or f"{settings.qdrant_collection}_v{generation}"

    client = store = embedder = None
    known_hashes: dict[str, str] | None = None
    if not dry_run:
        client, store, embedder = build_index(
            settings,
            effective_collection,
            settings.ingest_batch_size,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_query_instruction=embedding_query_instruction,
        )
        dim = await asyncio.to_thread(lambda: embedder.embedding_dim)
        await store.create_generation(effective_collection, dim)
        if not full:
            from infrastructure.ingest.incremental import load_index_state

            known_hashes = await load_index_state(client, effective_collection)

    config = replace(
        DEFAULT_CONFIG,
        incremental=IncrementalConfig(pipeline_version=pipeline_version),
    )
    # Клиент QA-модели собирается напрямую, минуя реестр: в одном процессе
    # раннера живут несколько deployment-ов, и глобальный слот "default"
    # привёл бы к их взаимному затиранию (та же причина, что в build_index).
    qa_model = None
    if qa_index and not dry_run:
        from infrastructure.config.model_registry import build_model_client

        qa_model = build_model_client(settings, settings.llm_model)
        run_logger.info(
            "QA-индексация включена: %s/%s, на каждый фрагмент уходит "
            "вызов модели, прогон будет заметно дольше",
            settings.llm_provider,
            settings.llm_model,
        )
    pipeline = IngestPipeline(config=config, qa_model=qa_model)
    run_sync_tag = sync_tag or f"run-{int(time.time())}"

    report = None
    try:
        await source.open()
        report = await pipeline.ingest(
            source,
            store,
            embedder,
            collection=effective_collection,
            sync_tag=run_sync_tag,
            limit=limit,
            dry_run=dry_run,
            step=ingest_document_step,
            known_hashes=known_hashes,
        )
        if switch_alias and not dry_run and not report.failed and store is not None:
            report.switched_from = await store.switch_generation(effective_collection)
        elif switch_alias and report.failed:
            run_logger.warning(
                "алиас не переключён: в прогоне были ошибки, поколение не принято"
            )
    finally:
        await source.close()
        if client is not None:
            await client.close()

    try:
        from prefect.artifacts import acreate_markdown_artifact

        lines = [
            "## Отчёт прогона",
            "",
            f"- Источник: `{source_kind}`",
            f"- Коллекция: `{effective_collection}`",
            f"- Документов: **{len(report.outcomes)}**",
            f"- Чанков: **{report.total_chunks}**",
            f"- Ошибок: **{len(report.failed)}**",
            f"- Dry-run: `{dry_run}`",
            f"- QA-индексация: `{qa_index}`",
        ]
        if report.switched_from:
            lines.append(
                f"- Алиас переключён: `{report.switched_from}` → "
                f"`{effective_collection}`"
            )
        await acreate_markdown_artifact(
            key="ingest-report",
            description="Итог последнего прогона индексации",
            markdown="\n".join(lines),
        )
    except ImportError:
        pass
    except (RuntimeError, ValueError):
        pass

    return report
