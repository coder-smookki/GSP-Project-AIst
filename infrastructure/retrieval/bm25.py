"""Параметры лексического канала BM25 — единственная точка правды.

BM25 считает **сам сервер Qdrant** (встроен начиная с v1.15.2). Клиент
пропускает `models.Document` насквозь, если работает не в локальном режиме:
в `qdrant_client/embed/embedder.py` sparse-текст для модели ``Qdrant/bm25``
возвращается как `Document` без обращения к fastembed. Поэтому зависимость
`fastembed` проекту не нужна, а образ ничего не скачивает.

Два правила, нарушение которых не даёт ошибки, но ломает поиск:

1. **Язык обязателен.** Без ``language`` сервер считает текст английским —
   русская морфология не схлопывается, запрос «средству защиты» не находит
   «средства защиты», и лексический канал молча отдаёт ноль.
2. **Опции записи и запроса должны совпадать.** Документация Qdrant требует
   идентичности; расхождение тоже не даёт ошибки, а просто ухудшает выдачу.
   Ради этого опции и вынесены в отдельный модуль: физически неоткуда взять
   разные.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)

#: Имя встроенной в сервер модели разреженных векторов.
BM25_MODEL: Final[str] = "Qdrant/bm25"

#: Имя dense-вектора в коллекции.
DENSE_VECTOR: Final[str] = "dense"

#: Имя sparse-вектора в коллекции.
SPARSE_VECTOR: Final[str] = "bm25"

#: Опции BM25. Задаются в КАЖДОМ документе при записи и в КАЖДОМ запросе.
#:
#: ``language="russian"`` включает сразу и снежковый стеммер русского, и
#: русский стоп-лист. ``min_token_len`` намеренно не задаётся: он выбросил бы
#: короткие аббревиатуры «ГО», «ЧС», «СИЗ», а на них строится половина
#: запросов по этому корпусу.
BM25_OPTIONS: Final[dict[str, Any]] = {"language": "russian"}

#: Минимальная версия сервера. Клиент версию не проверяет и отправит
#: `Document` на любой сервер, получив 400 уже во время демонстрации.
MIN_SERVER_VERSION: Final[tuple[int, int, int]] = (1, 16, 0)


def bm25_document(text: str) -> models.Document:
    """Собирает документ лексического канала для записи или запроса.

    Args:
        text: Текст чанка (при индексации) или текст запроса (при поиске).

    Returns:
        models.Document: Объект, который сервер сам превратит в разреженный
        вектор. Опции берутся из `BM25_OPTIONS` и потому одинаковы на обеих
        сторонах.
    """
    return models.Document(text=text, model=BM25_MODEL, options=BM25_OPTIONS)


def _parse_version(raw: str) -> tuple[int, int, int]:
    """Разбирает строку версии сервера.

    Args:
        raw: Версия как её отдаёт сервер, например ``1.19.0`` или ``1.19.0-rc``.

    Returns:
        tuple[int, int, int]: Тройка чисел; недостающие позиции заполняются
        нулями, неразбираемые — тоже (тогда проверка версии не сработает,
        но и не уронит приложение на ровном месте).
    """
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
    """Проверяет, что сервер умеет встроенный BM25.

    Args:
        client: Клиент Qdrant.

    Returns:
        str: Версия сервера.

    Raises:
        RuntimeError: Если версия ниже `MIN_SERVER_VERSION`. Падать на старте
            с внятным текстом лучше, чем получить 400 от сервера посреди
            демонстрации.
    """
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
