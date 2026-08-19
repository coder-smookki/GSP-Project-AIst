"""Каталог базы знаний: дерево областей и сборка документа из индекса.

Интерфейсу нужны две вещи, которых нет в поиске: перечень того, что вообще
лежит в базе (обязательное по ТЗ дерево областей знаний), и текст документа
целиком, чтобы карточка источника раскрывалась в контекст.

Оба ответа собираются ИЗ ИНДЕКСА, а не из исходных файлов. Так сделано
намеренно: корпус не смонтирован в контейнеры приложения, пути к файлам в
payload не хранятся, а у документов из API их нет вовсе. Индекс же содержит
всё нужное - ``doc_code``, ``doc_title``, ``area``, ``section_type`` и
``order_no`` для восстановления порядка.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

logger = logging.getLogger(__name__)

#: Человеческие названия типов разделов для третьего уровня дерева.
SECTION_TITLES: dict[str, str] = {
    "naznachenie": "Назначение и область применения",
    "terminy": "Термины и определения",
    "normativnye_ssylki": "Нормативные ссылки",
    "obshchie": "Общие положения",
    "procedura": "Порядок проведения",
    "otvetstvennost": "Ответственность",
    "prilozhenie": "Приложения",
    "soderzhanie": "Содержание",
    "prochee": "Прочее",
}

#: Сколько точек тянуть за один проход прокрутки.
_SCROLL_BATCH = 512

#: Поля, без которых дерево не построить. Векторы не запрашиваются вовсе.
_TREE_FIELDS = ["doc_code", "doc_title", "area", "section_type"]

#: Поля для восстановления текста документа.
_DOC_FIELDS = [
    "chunk_id",
    "order_no",
    "section_no",
    "section_title",
    "section_type",
    "clause",
    "breadcrumb",
    "text",
    "is_table_region",
]


class CatalogUnavailable(RuntimeError):
    """Каталог недоступен: индекс не отвечает или коллекции нет."""


@dataclass(slots=True)
class DocumentEntry:
    """Документ во втором уровне дерева."""

    doc_code: str
    doc_title: str
    section_types: set[str] = field(default_factory=set)
    chunks: int = 0

    def to_dict(self) -> dict:
        """Приводит запись к виду, пригодному для передачи в интерфейс.

        Returns:
            dict: Код, заголовок, число фрагментов и типы разделов с
            человеческими названиями, отсортированные по названию.
        """
        return {
            "doc_code": self.doc_code,
            "doc_title": self.doc_title,
            "chunks": self.chunks,
            "section_types": sorted(
                (
                    {"key": key, "title": SECTION_TITLES.get(key, key)}
                    for key in self.section_types
                ),
                key=lambda item: item["title"],
            ),
        }


class Catalog:
    """Читает состав индекса: дерево областей и текст документа.

    Ответы кешируются на ``ttl`` секунд: состав базы меняется только при
    переиндексации, а дерево запрашивается на каждом открытии интерфейса.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        collection: str,
        ttl: float = 300.0,
    ) -> None:
        """Готовит каталог поверх коллекции индекса.

        Args:
            client: Асинхронный клиент Qdrant.
            collection: Имя коллекции или алиаса.
            ttl: Время жизни кеша дерева, с.
        """
        self._client = client
        self._collection = collection
        self._ttl = ttl
        self._tree_cache: tuple[float, list[dict]] | None = None
        self._lock = asyncio.Lock()

    async def _scroll(
        self,
        fields: list[str],
        query_filter: models.Filter | None = None,
    ):
        """Проходит коллекцию порциями, отдавая только нужные поля payload.

        Args:
            fields: Имена полей payload, которые надо вернуть.
            query_filter: Ограничение выборки; ``None`` - вся коллекция.

        Yields:
            dict: payload очередной точки.

        Raises:
            CatalogUnavailable: Если индекс недоступен или коллекции нет.
        """
        offset = None
        while True:
            try:
                points, offset = await self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=query_filter,
                    limit=_SCROLL_BATCH,
                    offset=offset,
                    with_payload=fields,
                    with_vectors=False,
                )
            except (UnexpectedResponse, ResponseHandlingException, ValueError) as exc:
                raise CatalogUnavailable(
                    f"каталог недоступен: коллекция {self._collection!r}, {exc}"
                ) from exc
            for point in points:
                yield point.payload or {}
            if offset is None:
                return

    async def tree(self) -> list[dict]:
        """Собирает дерево «область -> документ -> тип раздела».

        Returns:
            list[dict]: Области с вложенными документами. Один документ может
            попасть в несколько областей - поле ``area`` в payload это список.

        Raises:
            CatalogUnavailable: Если индекс недоступен.
        """
        now = time.monotonic()
        cached = self._tree_cache
        if cached is not None and now - cached[0] < self._ttl:
            return cached[1]

        async with self._lock:
            # Пока ждали замок, соседняя задача могла всё посчитать.
            cached = self._tree_cache
            if cached is not None and time.monotonic() - cached[0] < self._ttl:
                return cached[1]

            areas: dict[str, dict[str, DocumentEntry]] = {}
            async for payload in self._scroll(_TREE_FIELDS):
                doc_code = str(payload.get("doc_code") or "").strip()
                if not doc_code:
                    continue
                doc_title = str(payload.get("doc_title") or "")
                section_type = str(payload.get("section_type") or "")
                raw_areas = payload.get("area") or []
                if isinstance(raw_areas, str):
                    raw_areas = [raw_areas]
                for area in raw_areas or ["prochee"]:
                    bucket = areas.setdefault(str(area), {})
                    entry = bucket.get(doc_code)
                    if entry is None:
                        entry = DocumentEntry(doc_code=doc_code, doc_title=doc_title)
                        bucket[doc_code] = entry
                    if section_type:
                        entry.section_types.add(section_type)
                    entry.chunks += 1

            from infrastructure.ingest.areas import AREA_TITLES

            result = sorted(
                (
                    {
                        "area": area,
                        "title": AREA_TITLES.get(area, area),
                        "documents": sorted(
                            (entry.to_dict() for entry in docs.values()),
                            key=lambda item: item["doc_code"],
                        ),
                    }
                    for area, docs in areas.items()
                ),
                key=lambda item: item["title"],
            )
            self._tree_cache = (time.monotonic(), result)
            logger.info(
                "дерево каталога собрано: областей %d, документов %d",
                len(result),
                sum(len(item["documents"]) for item in result),
            )
            return result

    async def document(self, doc_code: str) -> dict:
        """Восстанавливает документ целиком из фрагментов индекса.

        Args:
            doc_code: Код документа, например ``Р-У2-59``.

        Returns:
            dict: Код, заголовок и фрагменты по возрастанию ``order_no``.
            Пустой список фрагментов означает, что документа в индексе нет.

        Raises:
            CatalogUnavailable: Если индекс недоступен.
        """
        code = doc_code.strip()
        if not code:
            return {"doc_code": "", "doc_title": "", "chunks": []}

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_code", match=models.MatchValue(value=code)
                )
            ]
        )
        chunks: list[dict] = []
        title = ""
        async for payload in self._scroll(_DOC_FIELDS, query_filter):
            title = title or str(payload.get("doc_title") or "")
            chunks.append(
                {
                    "chunk_id": str(payload.get("chunk_id") or ""),
                    "order_no": int(payload.get("order_no") or 0),
                    "section_no": str(payload.get("section_no") or ""),
                    "section_title": str(payload.get("section_title") or ""),
                    "section_type": str(payload.get("section_type") or ""),
                    "clause": payload.get("clause"),
                    "breadcrumb": str(payload.get("breadcrumb") or ""),
                    "text": str(payload.get("text") or ""),
                    "is_table_region": bool(payload.get("is_table_region")),
                }
            )

        chunks.sort(key=lambda item: item["order_no"])
        return {"doc_code": code, "doc_title": title, "chunks": chunks}
