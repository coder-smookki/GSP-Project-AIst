"""Выбор и жизненный цикл реализации поиска.

Фабрика скрывает от графа, что под протоколом `Search`: заглушка или
Qdrant. Граф получает объект и ничего не знает о его устройстве.

Ресурсы (клиент Qdrant) создаются один раз на процесс и закрываются при
остановке: открывать соединение на каждый запрос дорого, а оставлять
незакрытым — значит копить сокеты.
"""

from __future__ import annotations

import logging

from infrastructure.retrieval.mock_search import MockSearch
from infrastructure.retrieval.search import Search

logger = logging.getLogger(__name__)

#: Единственный на процесс клиент Qdrant. Закрывается через `shutdown()`.
_client = None


def get_search(settings) -> Search:
    """Создаёт реализацию поиска по настройкам.

    Args:
        settings: Настройки приложения (`infrastructure.config.settings`).

    Returns:
        Search: `QdrantSearch` в бою, `MockSearch` при выборе заглушки.
        При недоступности Qdrant возвращается **не** заглушка: подмена
        боевого поиска синтетическим корпусом из четырёх чанков выглядела бы
        как работающая система и обманула бы и нас, и жюри. Ошибка вылезет
        на первом же запросе как `SearchUnavailable`, что честно.
    """
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
    """Закрывает клиент Qdrant, если он создавался."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("клиент Qdrant закрыт")
