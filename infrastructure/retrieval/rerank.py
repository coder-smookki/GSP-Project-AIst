from __future__ import annotations

import logging
from typing import Any

from infrastructure.retrieval.search import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankerUnavailable(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
        half_precision: bool = True,
        candidates: int = 10,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._half_precision = half_precision
        self._candidates = candidates
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

    def rerank(
        self, query: str, hits: list[tuple[Chunk, float]], limit: int
    ) -> list[tuple[Chunk, float]]:
        if not hits:
            return []

        head = hits[: self._candidates]
        tail = hits[self._candidates :]

        try:
            model = self._load()
            scores = model.predict([(query, chunk.text) for chunk, _ in head])
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
    )
