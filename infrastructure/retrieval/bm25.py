from __future__ import annotations

import logging
from typing import Any, Final

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)

BM25_MODEL: Final[str] = "Qdrant/bm25"

DENSE_VECTOR: Final[str] = "dense"

SPARSE_VECTOR: Final[str] = "bm25"

BM25_OPTIONS: Final[dict[str, Any]] = {"language": "russian"}

MIN_SERVER_VERSION: Final[tuple[int, int, int]] = (1, 16, 0)


def bm25_document(text: str) -> models.Document:
    return models.Document(text=text, model=BM25_MODEL, options=BM25_OPTIONS)


def _parse_version(raw: str) -> tuple[int, int, int]:
    head = raw.split("-", 1)[0]
    parts: list[int] = []
    for piece in head.split(".")[:3]:
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


async def assert_server_supported(client: AsyncQdrantClient) -> str:
    info = await client.info()
    version = str(getattr(info, "version", "") or "")
    if _parse_version(version) < MIN_SERVER_VERSION:
        expected = ".".join(str(p) for p in MIN_SERVER_VERSION)
        raise RuntimeError(
            f"Qdrant версии {version or 'неизвестно'}: требуется не ниже {expected} "
            "(встроенный BM25). Клиент версию не проверяет и отправит запрос "
            "на любой сервер, поэтому ошибка всплыла бы только в рантайме"
        )
    logger.info("Qdrant версии %s: встроенный BM25 доступен", version)
    return version
