from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any

from infrastructure.retrieval.search import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankerUnavailable(RuntimeError):
    pass


class RerankerHttpError(RuntimeError):
    pass


class RerankerHttpClient:
    """Клиент хост-сервиса переоценки (deploy/rerank_server.py, MPS на macOS).

    Colima не пробрасывает GPU, поэтому инференс кросс-энкодера в контейнере
    сильно медленнее нативного macOS-торча. Хост-сервис считает оценки на
    хосте и отдаёт их по HTTP; контейнер ходит к нему через
    host.docker.internal. Сбои сервиса считаются некритичными: вызывающий код
    переключается на локальный расчёт.
    """

    def __init__(self, endpoint: str, *, timeout: float = 10.0) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def predict(self, query: str, texts: list[str]) -> list[float]:
        body = json.dumps({"query": query, "candidates": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._endpoint}/rerank",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise RerankerHttpError(
                f"хост-сервис переоценки недоступен: {exc}"
            ) from exc
        scores = payload.get("scores")
        if not isinstance(scores, list) or len(scores) != len(texts):
            raise RerankerHttpError("некорректный ответ хост-сервиса переоценки")
        return [float(score) for score in scores]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
        half_precision: bool = True,
        candidates: int = 10,
        endpoint: str = "",
        http_timeout: float = 10.0,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._half_precision = half_precision
        self._candidates = candidates
        self._endpoint = endpoint
        self._http = (
            RerankerHttpClient(endpoint, timeout=http_timeout) if endpoint else None
        )
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerUnavailable(
                "переоценка включена, но пакет sentence-transformers не "
                "установлен: выполните `uv sync --extra rerank` либо "
                "выключите переоценку (RERANK_ENABLED=0)"
            ) from exc

        kwargs: dict[str, Any] = {"max_length": self._max_length}
        if self._device:
            kwargs["device"] = self._device
        if self._half_precision:
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

        try:
            self._model = CrossEncoder(self._model_name, **kwargs)
        except Exception as exc:
            raise RerankerUnavailable(
                f"не удалось загрузить модель переоценки {self._model_name}: {exc}"
            ) from exc

        logger.info(
            "модель переоценки %s загружена (устройство %s, точность %s)",
            self._model_name,
            self._device or "по умолчанию",
            "половинная" if self._half_precision else "полная",
        )
        return self._model

    def _local_scores(self, query: str, texts: list[str]) -> list[float]:
        model: Any = self._load()
        return model.predict([(query, text) for text in texts])

    @staticmethod
    def _timed(callable_: Any, *args: Any) -> tuple[list[float], float]:
        start = time.perf_counter()
        result = callable_(*args)
        return result, (time.perf_counter() - start) * 1000

    def _score(self, query: str, texts: list[str]) -> list[float]:
        client = self._http
        if client is not None:
            try:
                scores, ms = self._timed(client.predict, query, texts)
                logger.info(
                    "переоценка через хост-сервис %s: %d кандидатов за %.0f мс",
                    self._endpoint,
                    len(texts),
                    ms,
                )
                return scores
            except RerankerHttpError as exc:
                logger.warning(
                    "хост-сервис переоценки не отвечает, считаем локально: %s",
                    exc,
                )
                self._http = None
        scores, ms = self._timed(self._local_scores, query, texts)
        logger.info("переоценка локально: %d кандидатов за %.0f мс", len(texts), ms)
        return scores

    def rerank(
        self, query: str, hits: list[tuple[Chunk, float]], limit: int
    ) -> list[tuple[Chunk, float]]:
        if not hits:
            return []

        head = hits[: self._candidates]
        tail = hits[self._candidates :]
        texts = [chunk.text for chunk, _ in head]

        try:
            scores = self._score(query, texts)
        except RerankerUnavailable:
            raise
        except Exception as exc:
            logger.warning(
                "переоценка не выполнена, используем порядок поиска: %s", exc
            )
            return hits[:limit]

        ranked = sorted(
            (
                (chunk, float(score))
                for (chunk, _), score in zip(head, scores, strict=True)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        result = (ranked + tail)[:limit]
        logger.debug(
            "переоценено кандидатов %d, лучшая оценка %.3f",
            len(head),
            ranked[0][1] if ranked else 0.0,
        )
        return result


def build_reranker(settings) -> CrossEncoderReranker | None:
    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(
        settings.rerank_model,
        device=settings.rerank_device or None,
        half_precision=settings.rerank_half_precision,
        candidates=settings.rerank_candidates,
        endpoint=settings.rerank_endpoint,
    )
