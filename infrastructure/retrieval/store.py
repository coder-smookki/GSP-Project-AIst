from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from infrastructure.ingest.models import IngestChunk
from infrastructure.retrieval.bm25 import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    assert_server_supported,
    bm25_document,
)

logger = logging.getLogger(__name__)

PAYLOAD_INDEXES: dict[str, Any] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_code": models.PayloadSchemaType.KEYWORD,
    "area": models.PayloadSchemaType.KEYWORD,
    "section_type": models.PayloadSchemaType.KEYWORD,
    "sync_tag": models.PayloadSchemaType.KEYWORD,
    "is_table_region": models.PayloadSchemaType.BOOL,
}


class IndexError_(RuntimeError):
    pass


def chunk_payload(
    chunk: IngestChunk,
    areas: list[str],
    sync_tag: str,
    content_hash: str = "",
    chunk_total: int = 0,
) -> dict[str, Any]:
    """Собирает полезную нагрузку точки индекса.

    Args:
        chunk: Фрагмент документа.
        areas: Области знаний документа.
        sync_tag: Метка прогона индексации.
        content_hash: Отпечаток содержимого документа.
        chunk_total: Сколько точек всего должно быть у документа. Нужен, чтобы
            отличить оборвавшуюся загрузку от завершённой: при первой загрузке
            все записанные точки несут ОДИН отпечаток, поэтому по отпечаткам
            недописанный документ неотличим от целого.

    Returns:
        dict[str, Any]: Полезная нагрузка точки.
    """
    return {
        "chunk_total": chunk_total,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_code": chunk.doc_code,
        "doc_title": chunk.doc_title,
        "section_no": chunk.section_no,
        "section_title": chunk.section_title,
        "section_type": str(chunk.section_type),
        "clause": chunk.clause,
        "breadcrumb": chunk.breadcrumb,
        "text": chunk.text,
        "aliases": list(chunk.aliases),
        "placeholders": list(chunk.placeholders),
        "is_table_region": chunk.is_table_region,
        "order_no": chunk.order_no,
        "area": areas,
        "sync_tag": sync_tag,
        "content_hash": content_hash,
    }


class QdrantStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        alias: str,
        batch_size: int = 128,
    ) -> None:
        self._client = client
        self._alias = alias
        self._batch_size = batch_size

    @property
    def alias(self) -> str:
        return self._alias

    def generation_name(self, version: int) -> str:
        return f"{self._alias}_v{version}"

    async def create_generation(self, name: str, dim: int) -> None:
        await assert_server_supported(self._client)

        if await self._client.collection_exists(name):
            # Размерность вектора прошита в схему коллекции и на лету не
            # меняется. Раньше существующая коллекция принималась как есть, и
            # смена модели эмбеддингов проходила незамеченной: документы
            # пропускались как неизменившиеся (модель не входит в их
            # отпечаток), запись не выполнялась, а падало это позже и в другом
            # месте - на первом же запросе с «Vector dimension error: expected
            # dim: 2560, got 1024». Связать такое сообщение с индексацией
            # трудно, поэтому проверяем сразу и говорим, что делать.
            existing = await self._client.get_collection(name)
            vectors = existing.config.params.vectors or {}
            current = getattr(vectors.get(DENSE_VECTOR), "size", None)
            if current is not None and current != dim:
                raise ValueError(
                    f"коллекция {name} собрана под вектор размерности {current}, "
                    f"а текущий эмбеддер даёт {dim}. Размерность в схеме "
                    f"коллекции не меняется: соберите новое поколение "
                    f"(--generation N) и переключите алиас (--switch-alias), "
                    f"либо верните прежнюю модель эмбеддингов"
                )
            logger.info(
                "коллекция %s уже существует, размерность %s совпадает",
                name,
                current,
            )
        else:
            await self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
                on_disk_payload=False,
            )
            logger.info("создана коллекция %s, размерность dense-вектора %d", name, dim)

        for field_name, schema in PAYLOAD_INDEXES.items():
            await self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        logger.info(
            "коллекция %s: создано payload-индексов %d", name, len(PAYLOAD_INDEXES)
        )

    async def upsert_chunks(
        self,
        collection: str,
        chunks: list[IngestChunk],
        vectors: list[list[float]],
        areas: list[str],
        sync_tag: str,
        content_hash: str = "",
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"чанков {len(chunks)}, векторов {len(vectors)} — "
                "рассогласование сломало бы соответствие текста и вектора"
            )
        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            batch_vectors = vectors[start : start + self._batch_size]
            points = [
                models.PointStruct(
                    id=chunk.point_id(),
                    vector={
                        DENSE_VECTOR: vector,
                        SPARSE_VECTOR: bm25_document(chunk.text),
                    },
                    payload=chunk_payload(
                        chunk, areas, sync_tag, content_hash, len(chunks)
                    ),
                )
                for chunk, vector in zip(batch, batch_vectors, strict=True)
            ]
            result = await self._client.upsert(
                collection_name=collection, points=points, wait=True
            )
            if result.status != models.UpdateStatus.COMPLETED:
                raise IndexError_(
                    f"загрузка батча не завершена: статус {result.status}"
                )
            total += len(points)

        logger.info(
            "документ %s: загружено точек %d в коллекцию %s",
            chunks[0].doc_id,
            total,
            collection,
        )
        return total

    async def sweep_document(self, collection: str, doc_id: str, sync_tag: str) -> None:
        await self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="sync_tag", match=models.MatchValue(value=sync_tag)
                        )
                    ],
                )
            ),
            wait=True,
        )
        logger.debug("документ %s: устаревшие точки удалены", doc_id)

    async def current_generation(self) -> str | None:
        aliases = await self._client.get_aliases()
        for record in aliases.aliases:
            if record.alias_name == self._alias:
                return record.collection_name
        return None

    async def switch_generation(self, collection: str) -> str | None:
        # Алиас и коллекция делят одно пространство имён: Qdrant не даст
        # создать алиас с именем существующей коллекции и ответит сырым
        # «409 Conflict: Collection ... already exists». По такому сообщению
        # причину не угадать, а она в настройке: QDRANT_COLLECTION задаёт имя
        # АЛИАСА, а не поколения. Если туда записать готовое имя коллекции
        # (скажем, aist_chunks_v2), поколение соберётся как aist_chunks_v2_v1,
        # а переключение упадёт.
        # Проверяется СПИСОК настоящих коллекций, а не collection_exists:
        # тот разрешает алиасы и отвечает True на имя алиаса. Из-за этого
        # проверка срабатывала ложно на каждом переключении после первого
        # поколения - то есть блокировала ровно тот сценарий, ради которого
        # алиасы и заведены.
        listing = await self._client.get_collections()
        existing = {item.name for item in listing.collections}
        if self._alias in existing:
            raise ValueError(
                f"нельзя создать алиас {self._alias}: коллекция с таким именем "
                f"уже существует. QDRANT_COLLECTION задаёт имя алиаса, а не "
                f"поколения; поколения называются <алиас>_v1, <алиас>_v2. "
                f"Поставьте QDRANT_COLLECTION={self._alias.rsplit('_v', 1)[0]} "
                f"и выберите поколение флагом --generation"
            )

        previous = await self.current_generation()
        operations: list[models.AliasOperations] = []
        if previous:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self._alias)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection, alias_name=self._alias
                )
            )
        )
        await self._client.update_collection_aliases(
            change_aliases_operations=operations
        )
        logger.info(
            "алиас %s переключён на %s (было: %s)",
            self._alias,
            collection,
            previous or "ничего",
        )
        return previous

    async def count(self, collection: str | None = None) -> int:
        target = collection or self._alias
        result = await self._client.count(collection_name=target, exact=True)
        return result.count
