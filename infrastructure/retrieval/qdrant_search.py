"""Гибридный поиск по индексу: dense + BM25, слияние RRF.

Четыре правила, каждое из которых нарушается молча — без ошибки, но с
испорченной выдачей:

1. **Фильтр ставится внутрь КАЖДОГО `Prefetch`.** Верхнеуровневый
   `query_filter` при fusion-корне не применяется вовсе: в ветке слияния
   сервер не получает фильтр. Проверено: запрос с `query_filter` вернул
   точки, нарушающие фильтр.
2. **`k` задаётся через `RrfQuery`**, а не через `FusionQuery` — у
   последнего попросту нет поля для параметров.
3. **Лимит предвыборки не меньше `limit + offset`**, иначе при листании
   получается пустой ответ.
4. **Порог по счёту на слитом результате не ставится**: счёт RRF при
   `k=60` не превышает ~0,033, и любой «разумный» порог отсечёт всё.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from infrastructure.retrieval.bm25 import DENSE_VECTOR, SPARSE_VECTOR, bm25_document
from infrastructure.retrieval.search import Chunk, Search

logger = logging.getLogger(__name__)


class SearchUnavailable(RuntimeError):
    """Поиск недоступен: индекс не отвечает или не готов.

    Это **не** то же самое, что «ничего не найдено», и разделение здесь
    принципиально. Если при недоступном индексе вернуть пустой список,
    ассистент честно ответит «В базе знаний информации нет» — то есть авария
    замаскируется под правильный ответ, и на защите жюри получит эту фразу
    на каждый вопрос, решив, что система просто ничего не умеет.
    """


class QdrantSearch(Search):
    """Реализация протокола `Search` поверх Qdrant."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder,
        *,
        collection: str,
        top_k: int = 8,
        prefetch: int = 30,
        rrf_k: int = 60,
        query_instruction: str = "",
    ) -> None:
        """Создаёт поиск.

        Args:
            client: Асинхронный клиент Qdrant.
            embedder: Клиент эмбеддингов с методом ``aembed(texts)``.
            collection: Имя алиаса коллекции.
            top_k: Число чанков по умолчанию.
            prefetch: Число кандидатов на канал до слияния.
            rrf_k: Константа RRF. У Qdrant значение по умолчанию 2, а ранги
                нумеруются с нуля, поэтому классические 60 ведут себя не так,
                как в статьях: при коротких списках кандидатов RRF вырождается
                в голосование «нашли оба канала». Значение подбирается замером
                на золотом наборе.
            query_instruction: Инструкция, добавляемая к тексту запроса перед
                кодированием. Нужна моделям qwen3-embedding, для bge-m3
                остаётся пустой.
        """
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._top_k = top_k
        self._prefetch = prefetch
        self._rrf_k = rrf_k
        self._query_instruction = query_instruction

    @staticmethod
    def _build_filter(
        areas: Sequence[str] | None, section_types: Sequence[str] | None
    ) -> models.Filter | None:
        """Собирает фильтр из двух независимых осей.

        Args:
            areas: Предметные области знаний, выбранные пользователем.
            section_types: Типы разделов документа (служебная ось).

        Returns:
            models.Filter | None: Фильтр или ``None``, если ограничений нет.
            Пустой выбор означает поиск по всей базе, а не по пустому
            множеству.
        """
        must: list[models.Condition] = []
        if areas:
            must.append(
                models.FieldCondition(
                    key="area", match=models.MatchAny(any=list(areas))
                )
            )
        if section_types:
            must.append(
                models.FieldCondition(
                    key="section_type", match=models.MatchAny(any=list(section_types))
                )
            )
        return models.Filter(must=must) if must else None

    @staticmethod
    def _to_chunk(payload: dict | None) -> Chunk:
        """Собирает `Chunk` из payload точки.

        Args:
            payload: Payload точки Qdrant.

        Returns:
            Chunk: Чанк в формате, который ожидает граф и функция цитирования.
        """
        data = payload or {}
        return Chunk(
            chunk_id=str(data.get("chunk_id") or ""),
            doc_code=str(data.get("doc_code") or ""),
            doc_title=str(data.get("doc_title") or ""),
            section=str(data.get("section_no") or ""),
            section_type=str(data.get("section_type") or ""),
            clause=data.get("clause"),
            text=str(data.get("text") or ""),
            contains_table=bool(data.get("is_table_region")),
        )

    async def query(
        self,
        query: str,
        sections: list[str] | None = None,
        limit: int = 8,
        areas: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Выполняет гибридный поиск.

        Args:
            query: Текст запроса пользователя.
            sections: Типы разделов документа для фильтрации (служебная ось).
            limit: Число возвращаемых чанков.
            areas: Предметные области знаний, выбранные в интерфейсе.

        Returns:
            list[tuple[Chunk, float]]: Пары «чанк, счёт RRF» по убыванию счёта.
            Пустой список означает именно «ничего не найдено».

        Raises:
            SearchUnavailable: Если индекс или модель эмбеддингов недоступны.
                Отличать этот случай от пустой выдачи обязательно.
        """
        text = query.strip()
        if not text:
            return []

        effective_limit = limit or self._top_k
        prefetch_limit = max(self._prefetch, effective_limit)
        query_filter = self._build_filter(areas, sections)

        try:
            encoded = (
                f"{self._query_instruction}{text}" if self._query_instruction else text
            )
            dense_vector = (await self._embedder.aembed([encoded]))[0]
            response = await self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    # Фильтр обязателен в ОБЕИХ ветках: при fusion-корне
                    # верхнеуровневый query_filter не применяется.
                    models.Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=bm25_document(text),
                        using=SPARSE_VECTOR,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.RrfQuery(rrf=models.Rrf(k=self._rrf_k)),
                limit=effective_limit,
                with_payload=True,
                with_vectors=False,
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            logger.exception("поиск в Qdrant не выполнен")
            raise SearchUnavailable(f"индекс недоступен: {exc}") from exc
        except (OSError, TimeoutError) as exc:
            logger.exception("сеть или таймаут при обращении к индексу")
            raise SearchUnavailable(f"индекс недоступен: {exc}") from exc
        except Exception as exc:
            logger.exception("непредвиденная ошибка поиска")
            raise SearchUnavailable(f"поиск не выполнен: {exc}") from exc

        hits = [
            (self._to_chunk(point.payload), point.score) for point in response.points
        ]
        logger.info(
            "поиск «%s»: найдено %d (области: %s, разделы: %s)",
            text[:60],
            len(hits),
            ", ".join(areas) if areas else "все",
            ", ".join(sections) if sections else "все",
        )
        return hits

    async def preflight(self) -> None:
        """Проверяет готовность индекса до начала работы.

        Проверки `/readyz` недостаточно: она говорит лишь о том, что сервер
        принимает трафик. Она зелёная и когда алиас не переключён, и когда
        коллекция пуста — в обоих случаях поиск отдаст ноль хитов, и это
        снова прочтётся как «информации нет».

        Raises:
            SearchUnavailable: Если алиас не создан или коллекция пуста.
        """
        try:
            aliases = await self._client.get_aliases()
            known = {record.alias_name for record in aliases.aliases}
            if self._collection not in known:
                raise SearchUnavailable(
                    f"алиас {self._collection} не указывает ни на одну коллекцию: "
                    "индекс не собран"
                )
            total = await self._client.count(
                collection_name=self._collection, exact=True
            )
            if total.count == 0:
                raise SearchUnavailable(
                    f"коллекция под алиасом {self._collection} пуста"
                )
        except SearchUnavailable:
            raise
        except Exception as exc:
            raise SearchUnavailable(f"индекс недоступен: {exc}") from exc
        logger.info("индекс готов: алиас %s, точек %d", self._collection, total.count)
