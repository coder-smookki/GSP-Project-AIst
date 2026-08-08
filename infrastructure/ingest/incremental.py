from __future__ import annotations

import hashlib
import logging

from qdrant_client import AsyncQdrantClient, models

from infrastructure.ingest.models import ParsedDocument

logger = logging.getLogger(__name__)

SCROLL_BATCH = 1024


def document_hash(document: ParsedDocument, pipeline_version: str) -> str:
    digest = hashlib.sha256()
    digest.update(pipeline_version.encode("utf-8"))
    digest.update(document.doc_code.encode("utf-8"))
    digest.update(document.doc_title.encode("utf-8"))
    for atom in document.atoms():
        digest.update(atom.section_no.encode("utf-8"))
        digest.update((atom.clause or "").encode("utf-8"))
        digest.update(atom.text.encode("utf-8"))
    return digest.hexdigest()


async def load_index_state(
    client: AsyncQdrantClient, collection: str
) -> dict[str, str]:
    if not await client.collection_exists(collection):
        logger.info("коллекция %s не существует: индексация будет полной", collection)
        return {}

    state: dict[str, str] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["doc_id", "content_hash"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            content_hash = payload.get("content_hash")
            if isinstance(doc_id, str) and isinstance(content_hash, str):
                state.setdefault(doc_id, content_hash)
        if offset is None:
            break

    logger.info(
        "коллекция %s: состояние прочитано, документов в индексе %d",
        collection,
        len(state),
    )
    return state


async def drop_documents(
    client: AsyncQdrantClient, collection: str, doc_ids: list[str]
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
