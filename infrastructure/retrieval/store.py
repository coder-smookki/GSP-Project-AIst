"""Хранилище чанков в Qdrant: коллекция, загрузка, уборка, поколения.

Схема коллекции:

* именованный dense-вектор ``dense`` размерности эмбеддера, метрика косинус
  (Qdrant сам нормализует векторы при загрузке);
* разреженный вектор ``bm25`` с модификатором IDF — статистику считает
  сервер, от нас идёт только текст;
* payload с метаданными чанка и payload-индексы для фильтров.

Работа идёт **через алиас**, а не через имя коллекции. Размерность
именованного вектора неизменяема, поэтому смена модели эмбеддера требует
новой коллекции; алиас позволяет переключить поколение атомарно и откатиться
обратно тем же вызовом.
"""

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

#: Поля payload, по которым выполняется фильтрация, и их типы индексов.
#:
#: Индексы создаются ДО первой загрузки: Qdrant строит фильтр-ориентированные
#: рёбра HNSW только в этом случае, иначе фильтрованный поиск деградирует и
#: индекс придётся перестраивать.
#:
#: `index_version` намеренно без индекса: у него кардинальность 1-2, а
#: документация Qdrant прямо говорит, что индекс тем эффективнее, чем больше
#: у поля различных значений.
PAYLOAD_INDEXES: dict[str, Any] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_code": models.PayloadSchemaType.KEYWORD,
    "area": models.PayloadSchemaType.KEYWORD,
    "section_type": models.PayloadSchemaType.KEYWORD,
    "sync_tag": models.PayloadSchemaType.KEYWORD,
    "is_table_region": models.PayloadSchemaType.BOOL,
}


class IndexError_(RuntimeError):
    """Ошибка работы с индексом."""


def chunk_payload(
    chunk: IngestChunk,
    areas: list[str],
    sync_tag: str,
    content_hash: str = "",
) -> dict[str, Any]:
    """Собирает payload точки из чанка.

    Args:
        chunk: Чанк конвейера индексации.
        areas: Предметные области документа (см. `infrastructure.ingest.areas`).
        sync_tag: Метка прогона индексации. По ней уборка отличает точки
            текущего прогона от устаревших.
        content_hash: Отпечаток документа. По нему следующий прогон решает,
            менялся ли документ, и пропускает неизменившиеся.

    Returns:
        dict[str, Any]: Payload точки Qdrant.
    """
    return {
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
    """Операции над коллекцией чанков."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        alias: str,
        batch_size: int = 128,
    ) -> None:
        """Создаёт хранилище.

        Args:
            client: Асинхронный клиент Qdrant.
            alias: Имя алиаса коллекции. Всё чтение и запись идут через него.
            batch_size: Размер батча загрузки. Документация Qdrant советует
                64-256 точек: меньше — недогружаем сеть, больше — растёт
                давление на память сервера.
        """
        self._client = client
        self._alias = alias
        self._batch_size = batch_size

    @property
    def alias(self) -> str:
        """Возвращает имя алиаса коллекции.

        Returns:
            str: Имя алиаса.
        """
        return self._alias

    def generation_name(self, version: int) -> str:
        """Строит имя коллекции-поколения.

        Args:
            version: Номер поколения.

        Returns:
            str: Имя вида ``aist_chunks_v3``.
        """
        return f"{self._alias}_v{version}"

    async def create_generation(self, name: str, dim: int) -> None:
        """Создаёт коллекцию нового поколения вместе с payload-индексами.

        Args:
            name: Имя коллекции-поколения.
            dim: Размерность dense-вектора, полученная от модели эмбеддера.

        Raises:
            RuntimeError: Если версия сервера не поддерживает встроенный BM25.
        """
        await assert_server_supported(self._client)

        if await self._client.collection_exists(name):
            logger.info("коллекция %s уже существует, создание пропущено", name)
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
        """Загружает чанки одного документа батчами.

        В каждой точке передаются **оба** вектора. Это обязательно: upsert
        перезаписывает точку целиком, и запись только dense-вектора молча
        обнулила бы лексический канал у этой точки.

        Args:
            collection: Имя коллекции-поколения.
            chunks: Чанки документа.
            vectors: Dense-векторы в том же порядке, что и чанки.
            areas: Предметные области документа.
            sync_tag: Метка текущего прогона индексации.
            content_hash: Отпечаток документа для инкрементального режима.

        Returns:
            int: Число загруженных точек.

        Raises:
            ValueError: Если число векторов не совпадает с числом чанков —
                иначе часть точек получила бы чужой вектор.
            IndexError_: Если сервер не подтвердил применение изменений.
        """
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
                    payload=chunk_payload(chunk, areas, sync_tag, content_hash),
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
        """Удаляет точки документа, оставшиеся от прошлых прогонов.

        Порядок «сначала записать, потом подмести» выбран намеренно: между
        двумя шагами индекс содержит и старые, и новые точки, то есть поиск
        продолжает работать. Обратный порядок открыл бы окно, в котором
        документ отсутствует в индексе целиком.

        Args:
            collection: Имя коллекции-поколения.
            doc_id: Идентификатор документа.
            sync_tag: Метка текущего прогона: всё, что помечено иначе,
                устарело.
        """
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
        """Возвращает коллекцию, на которую сейчас указывает алиас.

        `collection_exists(alias)` для этого не годится: он отвечает ``True``
        и не отличает алиас от коллекции.

        Returns:
            str | None: Имя коллекции или ``None``, если алиас не создан.
        """
        aliases = await self._client.get_aliases()
        for record in aliases.aliases:
            if record.alias_name == self._alias:
                return record.collection_name
        return None

    async def switch_generation(self, collection: str) -> str | None:
        """Атомарно переключает алиас на новое поколение.

        Обе операции идут одним запросом — Qdrant применяет их атомарно,
        поэтому окна недоступности нет и параллельные запросы не пострадают.

        Args:
            collection: Имя коллекции нового поколения.

        Returns:
            str | None: Имя прежней коллекции, если алиас уже существовал.
            Её не удаляем сразу: это путь отката.
        """
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
        """Возвращает точное число точек.

        Args:
            collection: Имя коллекции; по умолчанию — алиас.

        Returns:
            int: Число точек. Считается точно: приблизительный
            ``points_count`` не годится для приёмки поколения.
        """
        target = collection or self._alias
        result = await self._client.count(collection_name=target, exact=True)
        return result.count
