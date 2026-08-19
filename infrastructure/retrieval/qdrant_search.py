from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from infrastructure.retrieval.bm25 import DENSE_VECTOR, SPARSE_VECTOR, bm25_document
from infrastructure.retrieval.rerank import RerankerUnavailable
from infrastructure.retrieval.search import Chunk, Search

logger = logging.getLogger(__name__)


class SearchUnavailable(RuntimeError):
    pass


class QdrantSearch(Search):
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
        min_relevance: float = 0.0,
        reranker=None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._top_k = top_k
        self._prefetch = prefetch
        self._rrf_k = rrf_k
        self._query_instruction = query_instruction
        self._min_relevance = min_relevance
        self._reranker = reranker

    @staticmethod
    def _build_filter(
        areas: Sequence[str] | None,
        section_types: Sequence[str] | None,
        doc_codes: Sequence[str] | None = None,
    ) -> models.Filter | None:
        """Собирает фильтр Qdrant по трём уровням дерева областей знаний.

        Все три поля проиндексированы как KEYWORD (см. ``PAYLOAD_INDEXES``),
        поэтому фильтрация идёт по индексу, а не перебором.

        Args:
            areas: Области знаний (payload ``area``, список на чанк).
            section_types: Типы разделов (payload ``section_type``).
            doc_codes: Коды документов (payload ``doc_code``).

        Returns:
            models.Filter | None: Фильтр или ``None``, если ограничений нет.
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
        if doc_codes:
            must.append(
                models.FieldCondition(
                    key="doc_code", match=models.MatchAny(any=list(doc_codes))
                )
            )
        return models.Filter(must=must) if must else None

    @staticmethod
    def _to_chunk(payload: dict | None) -> Chunk:
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
        limit: int | None = None,
        areas: list[str] | None = None,
        docs: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        text = query.strip()
        if not text:
            return []

        effective_limit = limit or self._top_k
        fetch_limit = (
            max(effective_limit, self._reranker._candidates)
            if self._reranker is not None
            else effective_limit
        )
        prefetch_limit = max(self._prefetch, fetch_limit)
        query_filter = self._build_filter(areas, sections, docs)

        try:
            encoded = (
                f"{self._query_instruction}{text}" if self._query_instruction else text
            )
            dense_vector = (await self._embedder.aembed([encoded]))[0]
            response = await self._client.query_points(
                collection_name=self._collection,
                prefetch=[
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
                limit=fetch_limit,
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

        if self._reranker is not None and hits:
            try:
                # Переоценка синхронная: это либо HTTP к хост-сервису, либо
                # инференс кросс-энкодера. Прямой вызов из корутины замораживал
                # весь событийный цикл на всё её время - параллельные ветки
                # поиска выстраивались в очередь, а стриминг ответа замирал.
                hits = await asyncio.to_thread(
                    self._reranker.rerank, text, hits, effective_limit
                )
            except RerankerUnavailable as exc:
                # Реранкер не выбрасывается: у него своё остывание на 30 с
                # (REMOTE_COOLDOWN_SECONDS), и обнуление ссылки его обходило -
                # один сетевой сбой отключал переоценку до перезапуска
                # процесса, а с ней уходило +17 пунктов hit@1, молча.
                logger.error("переоценка пропущена для этого запроса: %s", exc)
                hits = hits[:effective_limit]
        else:
            hits = hits[:effective_limit]

        if hits and self._min_relevance > 0:
            best = await self._best_dense_score(dense_vector, query_filter)
            if best < self._min_relevance:
                logger.info(
                    "запрос «%s»: лучшая близость %.3f ниже порога %.3f — "
                    "считаем, что ответа в базе нет",
                    text[:60],
                    best,
                    self._min_relevance,
                )
                return []
        logger.info(
            "поиск «%s»: найдено %d (области: %s, разделы: %s)",
            text[:60],
            len(hits),
            ", ".join(areas) if areas else "все",
            ", ".join(sections) if sections else "все",
        )
        return hits

    async def _best_dense_score(
        self, vector: list[float], query_filter: models.Filter | None
    ) -> float:
        try:
            probe = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                using=DENSE_VECTOR,
                query_filter=query_filter,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("проверка порога релевантности не выполнена: %s", exc)
            return 0.0
        return probe.points[0].score if probe.points else 0.0

    async def preflight(self) -> None:
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
