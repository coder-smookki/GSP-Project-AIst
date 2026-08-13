#!/usr/bin/env python
"""Re-embed existing v1 corpus with qwen3-embedding:4b → v2 collection, switch alias."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["EMBEDDING_MODEL"] = "qwen3-embedding:4b"
os.environ["EMBEDDING_PROVIDER"] = "ollama-local"

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

from infrastructure.config.model_registry import get_embedding, setup_registry
from infrastructure.config.settings import Settings
from infrastructure.retrieval.bm25 import DENSE_VECTOR, SPARSE_VECTOR, assert_server_supported, bm25_document
from qdrant_client import AsyncQdrantClient, models

ALIAS = "aist_chunks"
V1 = "aist_chunks_v1"
V2 = "aist_chunks_v2"
BATCH = 128


async def main():
    settings = Settings(embedding_model="qwen3-embedding:4b")
    setup_registry(settings)
    embedder = get_embedding("default")
    dim = await asyncio.to_thread(lambda: embedder.embedding_dim)
    logging.info(f"qwen3-embedding:4b dim={dim}")

    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    await assert_server_supported(client)

    # Scroll all points from v1
    logging.info(f"Scrolling {V1}...")
    points: list[models.ScrolledPoint] = []
    next_page = None
    while True:
        result = await client.scroll(
            collection_name=V1,
            with_vectors=False,
            with_payload=True,
            limit=500,
            offset=next_page,
        )
        batch, next_page = result
        if not batch:
            break
        points.extend(batch)
        logging.info(f"  scrolled {len(points)} points so far")
        if next_page is None:
            break
    logging.info(f"Total points: {len(points)}")

    # Extract texts
    texts = [p.payload["text"] for p in points if p.payload and "text" in p.payload]
    logging.info(f"Texts to embed: {len(texts)}")

    # Batch embed
    all_vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vectors = await embedder.aembed(batch)
        all_vectors.extend(vectors)
        logging.info(f"  embedded {len(all_vectors)}/{len(texts)} texts")

    # Create v2 collection
    logging.info(f"Creating {V2} (dim={dim})...")
    if await client.collection_exists(V2):
        logging.info(f"  {V2} already exists, deleting")
        await client.delete_collection(V2)
    await client.create_collection(
        collection_name=V2,
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

    # Copy payload indexes
    from infrastructure.retrieval.store import PAYLOAD_INDEXES
    for field_name, schema in PAYLOAD_INDEXES.items():
        await client.create_payload_index(
            collection_name=V2,
            field_name=field_name,
            field_schema=schema,
            wait=True,
        )
    logging.info(f"  payload indexes created ({len(PAYLOAD_INDEXES)})")

    # Upsert points
    logging.info(f"Upserting {len(points)} points to {V2}...")
    total = 0
    for start in range(0, len(points), BATCH):
        batch_points = points[start:start + BATCH]
        batch_vectors = all_vectors[start:start + BATCH]
        point_structs = [
            models.PointStruct(
                id=p.id,
                vector={
                    DENSE_VECTOR: vec,
                    SPARSE_VECTOR: bm25_document(p.payload["text"]),
                },
                payload=p.payload,
            )
            for p, vec in zip(batch_points, batch_vectors, strict=True)
        ]
        result = await client.upsert(
            collection_name=V2, points=point_structs, wait=True
        )
        if result.status != models.UpdateStatus.COMPLETED:
            raise RuntimeError(f"upsert failed: {result.status}")
        total += len(point_structs)
        logging.info(f"  upserted {total}/{len(points)} points")

    logging.info(f"Upserted {total} points to {V2}")

    # Verify count
    count = await client.count(collection_name=V2, exact=True)
    logging.info(f"{V2} point count: {count.count}")

    # Switch alias
    logging.info(f"Switching alias {ALIAS} → {V2}...")
    aliases = await client.get_aliases()
    current = None
    for rec in aliases.aliases:
        if rec.alias_name == ALIAS:
            current = rec.collection_name
    operations = []
    if current:
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS)))
    operations.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=V2, alias_name=ALIAS)))
    await client.update_collection_aliases(change_aliases_operations=operations)
    logging.info(f"  alias switched from {current} to {V2}")

    await client.close()
    logging.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
