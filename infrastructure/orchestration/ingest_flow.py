from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace

from infrastructure.config.settings import get_settings
from infrastructure.ingest.config import (
    DEFAULT_CONFIG,
    PIPELINE_VERSION,
    IncrementalConfig,
)
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

try:
    from prefect import flow, task
    from prefect.logging import get_run_logger
except ImportError:

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


@task(
    name="Индексация документа",
    task_run_name=_document_ref_id,
    log_prints=True,
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
    outcome = await pipeline.process_document(
        source,
        store,
        embedder,
        ref,
        collection=collection,
        sync_tag=sync_tag,
        dry_run=dry_run,
        known_hashes=known_hashes,
    )
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


def build_index(settings, collection: str, batch_size: int):
    from qdrant_client import AsyncQdrantClient

    from infrastructure.config.model_registry import get_embedding, setup_registry
    from infrastructure.retrieval.store import QdrantStore

    setup_registry(settings)
    embedder = get_embedding("default")
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
    switch_alias: bool = False,
    sync_tag: str | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> IngestReport:
    run_logger = get_run_logger()
    settings = get_settings()

    source = build_source(settings, source_kind, corpus_dir, section_id)
    effective_collection = collection or f"{settings.qdrant_collection}_v{generation}"

    client = store = embedder = None
    known_hashes: dict[str, str] | None = None
    if not dry_run:
        client, store, embedder = build_index(
            settings, effective_collection, settings.ingest_batch_size
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
    pipeline = IngestPipeline(config=config)
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

    return report
