from __future__ import annotations

import logging

from infrastructure.retrieval.mock_search import MockSearch
from infrastructure.retrieval.search import Search

logger = logging.getLogger(__name__)

_client = None


def get_search(settings) -> Search:
    global _client

    provider = settings.resolved_retrieval_provider()
    if provider == "mock":
        logger.warning(
            "ПОИСК РАБОТАЕТ НА ЗАГЛУШКЕ: синтетический корпус из 4 чанков. "
            "Для боевого режима задайте RETRIEVAL_PROVIDER=qdrant"
        )
        return MockSearch()

    from qdrant_client import AsyncQdrantClient

    from infrastructure.config.model_registry import get_embedding
    from infrastructure.retrieval.qdrant_search import QdrantSearch
    from infrastructure.retrieval.rerank import build_reranker

    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=int(settings.qdrant_timeout),
        )
        logger.info("создан клиент Qdrant: %s", settings.qdrant_url)

    search = QdrantSearch(
        _client,
        get_embedding("default"),
        collection=settings.qdrant_collection,
        top_k=settings.retrieval_top_k,
        prefetch=settings.retrieval_prefetch,
        rrf_k=settings.retrieval_rrf_k,
        query_instruction=settings.embedding_query_instruction,
        min_relevance=settings.retrieval_min_relevance,
        reranker=build_reranker(settings),
    )
    logger.info(
        "поиск: Qdrant, алиас %s, top_k=%d, prefetch=%d, rrf_k=%d",
        settings.qdrant_collection,
        settings.retrieval_top_k,
        settings.retrieval_prefetch,
        settings.retrieval_rrf_k,
    )
    return search


async def shutdown() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("клиент Qdrant закрыт")
