from __future__ import annotations

import hashlib
import logging

from qdrant_client import AsyncQdrantClient, models

from infrastructure.ingest.config import DEFAULT_INCREMENTAL, IncrementalConfig
from infrastructure.ingest.models import ParsedDocument

logger = logging.getLogger(__name__)

SCROLL_BATCH = DEFAULT_INCREMENTAL.scroll_batch


class HashRepository:
    def __init__(self, config: IncrementalConfig = DEFAULT_INCREMENTAL) -> None:
        self.config = config

    def fingerprint(self, document: ParsedDocument, embedder: str = "") -> str:
        """Считает отпечаток разобранного документа.

        Args:
            document: Разобранный документ.
            embedder: Имя модели эмбеддингов. Входит в отпечаток, потому что
                векторы разных моделей несовместимы по смыслу, даже когда
                совпадают по размерности. Без этого смена модели проходила бы
                как «документ не изменился», индекс оставался бы собранным
                прежней моделью, а качество поиска молча падало.

        Returns:
            str: Шестнадцатеричный отпечаток.
        """
        digest = hashlib.sha256()
        digest.update(self.config.pipeline_version.encode("utf-8"))
        digest.update(embedder.encode("utf-8"))
        digest.update(document.doc_code.encode("utf-8"))
        digest.update(document.doc_title.encode("utf-8"))
        for atom in document.atoms():
            digest.update(atom.section_no.encode("utf-8"))
            digest.update((atom.clause or "").encode("utf-8"))
            digest.update(atom.text.encode("utf-8"))
        return digest.hexdigest()


class StateManager:
    def __init__(
        self,
        repository: HashRepository | None = None,
        config: IncrementalConfig = DEFAULT_INCREMENTAL,
    ) -> None:
        self.repository = repository or HashRepository(config)
        self.config = config

    async def load(self, client: AsyncQdrantClient, collection: str) -> dict[str, str]:
        if not await client.collection_exists(collection):
            logger.info(
                "коллекция %s не существует: индексация будет полной", collection
            )
            return {}

        seen: dict[str, set[str]] = {}
        counted: dict[str, int] = {}
        expected: dict[str, set[int]] = {}
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=collection,
                limit=self.config.scroll_batch,
                offset=offset,
                with_payload=["doc_id", "content_hash", "chunk_total"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                content_hash = payload.get("content_hash")
                if isinstance(doc_id, str) and isinstance(content_hash, str):
                    seen.setdefault(doc_id, set()).add(content_hash)
                    counted[doc_id] = counted.get(doc_id, 0) + 1
                    expected.setdefault(doc_id, set()).add(
                        int(payload.get("chunk_total") or 0)
                    )
            if offset is None:
                break

        # Документ с несколькими отпечатками в индексе записан не полностью:
        # прошлый прогон оборвался посреди загрузки, и часть точек осталась от
        # предыдущей версии. Раньше здесь брался первый попавшийся отпечаток
        # (`setdefault` по первой встреченной точке), и если им оказывался
        # новый, документ на следующем прогоне считался неизменившимся и
        # пропускался НАВСЕГДА — недописанные чанки исчезали из поиска молча.
        # Порядок обхода `scroll` при этом ничем не задан, так что поведение
        # было ещё и невоспроизводимым.
        #
        # Правило простое: неоднозначность — это не «не изменилось». Такой
        # документ из состояния исключается и переиндексируется целиком.
        state: dict[str, str] = {}
        broken: list[str] = []
        partial: list[str] = []
        for doc_id, hashes in seen.items():
            if len(hashes) != 1:
                broken.append(doc_id)
                continue
            # Отпечатков один, но это ещё не значит, что документ дописан.
            # При ПЕРВОЙ загрузке все точки несут один и тот же отпечаток, и
            # обрыв посреди неё оставлял в индексе половину документа, которая
            # выглядела завершённой и потому не переиндексировалась никогда.
            # Поэтому сверяется и число точек.
            totals = expected.get(doc_id) or {0}
            total = max(totals)
            if len(totals) != 1 or total <= 0 or counted.get(doc_id, 0) != total:
                partial.append(doc_id)
                continue
            state[doc_id] = next(iter(hashes))

        if partial:
            logger.warning(
                "коллекция %s: документы записаны не полностью (%s) — "
                "загрузка обрывалась, они будут переиндексированы",
                collection,
                ", ".join(sorted(partial)),
            )

        if broken:
            logger.warning(
                "коллекция %s: документы с несколькими отпечатками (%s) — "
                "прошлая загрузка оборвалась, они будут переиндексированы",
                collection,
                ", ".join(sorted(broken)),
            )

        logger.info(
            "коллекция %s: состояние прочитано, документов в индексе %d",
            collection,
            len(state),
        )
        return state

    async def drop_documents(
        self, client: AsyncQdrantClient, collection: str, doc_ids: list[str]
    ) -> None:
        if not doc_ids:
            return
        await client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchAny(any=doc_ids)
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info(
            "удалены документы, исчезнувшие из источника: %s", ", ".join(doc_ids[:10])
        )


def document_hash(document: ParsedDocument, pipeline_version: str) -> str:
    return HashRepository(
        IncrementalConfig(pipeline_version=pipeline_version)
    ).fingerprint(document)


async def load_index_state(
    client: AsyncQdrantClient, collection: str
) -> dict[str, str]:
    return await StateManager().load(client, collection)


async def drop_documents(
    client: AsyncQdrantClient, collection: str, doc_ids: list[str]
) -> None:
    await StateManager().drop_documents(client, collection, doc_ids)
