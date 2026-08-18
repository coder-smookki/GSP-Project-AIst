from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Protocol

import httpx

from infrastructure.retrieval.search import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

#: Пауза перед повторной попыткой обратиться к удалённому счётчику.
#:
#: Полминуты: достаточно, чтобы не долбить упавший сервис на каждом запросе,
#: и достаточно мало, чтобы после его перезапуска качество вернулось само,
#: а не ждало перезапуска всего приложения.
REMOTE_COOLDOWN_SECONDS = 30

#: Предел ожидания удалённой переоценки, с. Короткий намеренно: переоценка
#: стоит на пути ответа пользователю, и лучше посчитать локально, чем ждать.
DEFAULT_HTTP_TIMEOUT = 10.0


class RerankerUnavailable(RuntimeError):
    pass


class RerankerHttpError(RuntimeError):
    pass


class RerankScorer(Protocol):
    """Удалённый счётчик релевантности пар «запрос - документ»."""

    def predict(self, query: str, texts: list[str]) -> list[float]:
        """Оценивает кандидатов относительно запроса.

        Args:
            query: Текст запроса.
            texts: Тексты кандидатов.

        Returns:
            list[float]: Оценки в порядке кандидатов.
        """
        ...


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


class VLLMRerankerClient:
    """Клиент переоценки на сервере vLLM (эндпоинт ``/v1/rerank``).

    vLLM обслуживает модель-реранкер по Cohere-совместимому пути: тело
    запроса - ``{"model", "query", "documents", "top_n"}``, ответ -
    ``{"results": [{"index", "relevance_score", ...}]}``, где ``index`` -
    номер документа во входном списке, а сами результаты отсортированы по
    убыванию оценки. Порядок восстанавливается по ``index``, потому что
    вызывающий код ждёт оценки в порядке кандидатов.

    Один процесс vLLM обслуживает одну модель, поэтому реранкер живёт на
    отдельном адресе от генерации и эмбеддингов.

    Источник: https://docs.vllm.ai/en/stable/models/pooling_models/scoring
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        """Создаёт клиент.

        Args:
            base_url: Адрес сервера vLLM без суффикса ``/v1``.
            model: Имя модели-реранкера на сервере.
            api_key: Ключ доступа; пусто - заголовок авторизации не шлётся.
            timeout: Предел ожидания ответа, с.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout)

    @property
    def endpoint(self) -> str:
        """Возвращает полный адрес эндпоинта переоценки.

        Returns:
            str: Адрес вида ``http://host:8000/v1/rerank``.
        """
        return f"{self._base_url}/v1/rerank"

    def predict(self, query: str, texts: list[str]) -> list[float]:
        """Оценивает кандидатов через ``/v1/rerank``.

        Args:
            query: Текст запроса.
            texts: Тексты кандидатов.

        Returns:
            list[float]: Оценки в порядке кандидатов.

        Raises:
            RerankerHttpError: Сервер недоступен, ответил ошибкой либо
                прислал ответ, который не удалось разобрать.
        """
        if not texts:
            return []
        payload = {
            "model": self._model,
            "query": query,
            "documents": texts,
            # По умолчанию vLLM возвращает не все результаты, а нам нужны
            # оценки для каждого кандидата.
            "top_n": len(texts),
        }
        try:
            response = self._client.post(
                self.endpoint, json=payload, headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise RerankerHttpError(
                f"vLLM недоступен по адресу {self.endpoint}: {exc}"
            ) from exc
        return self._parse(response, len(texts))

    def _parse(self, response: httpx.Response, count: int) -> list[float]:
        """Разбирает ответ сервера.

        Args:
            response: Ответ vLLM.
            count: Сколько кандидатов было отправлено.

        Returns:
            list[float]: Оценки в порядке кандидатов.

        Raises:
            RerankerHttpError: Код ответа не 200 либо тело не разобралось.
        """
        if response.status_code == 404:
            raise RerankerHttpError(
                f"vLLM не отдаёт {self.endpoint}: переоценку обслуживает "
                f"отдельный процесс, запущенный с моделью-реранкером "
                f"(`vllm serve {self._model}`), либо версия сервера старее "
                f"эндпоинта /v1/rerank"
            )
        if response.status_code >= 400:
            raise RerankerHttpError(
                f"vLLM ответил {response.status_code} на {self.endpoint}: "
                f"{response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RerankerHttpError(
                f"vLLM прислал не JSON на {self.endpoint}: {exc}"
            ) from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != count:
            raise RerankerHttpError(
                f"некорректный ответ vLLM: ожидалось {count} оценок, "
                f"получено {len(results) if isinstance(results, list) else 'ничего'}"
            )
        scores: dict[int, float] = {}
        for item in results:
            if not isinstance(item, dict):
                raise RerankerHttpError("некорректный ответ vLLM: элемент не объект")
            index = item.get("index")
            score = item.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankerHttpError(
                    f"некорректный ответ vLLM: индекс {index!r} не число"
                )
            if not 0 <= index < count:
                raise RerankerHttpError(
                    f"некорректный ответ vLLM: индекс {index} вне списка кандидатов"
                )
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise RerankerHttpError(
                    f"некорректный ответ vLLM: оценка {score!r} не число"
                )
            scores[index] = float(score)
        if len(scores) != count:
            raise RerankerHttpError(
                f"некорректный ответ vLLM: оценки пришли не для всех кандидатов "
                f"({len(scores)} из {count})"
            )
        return [scores[index] for index in range(count)]

    def close(self) -> None:
        """Закрывает HTTP-соединения клиента."""
        self._client.close()


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
        http_timeout: float = DEFAULT_HTTP_TIMEOUT,
        client: RerankScorer | None = None,
    ) -> None:
        """Создаёт реранкер.

        Args:
            model_name: Имя модели кросс-энкодера.
            device: Устройство локального расчёта; ``None`` - по умолчанию.
            max_length: Предел длины пары в токенах.
            half_precision: Считать ли локально в половинной точности.
            candidates: Сколько верхних кандидатов переоценивать.
            endpoint: Адрес хост-сервиса переоценки; пусто - не использовать.
            http_timeout: Предел ожидания хост-сервиса, с.
            client: Готовый удалённый счётчик (например, vLLM). Имеет
                приоритет над ``endpoint``.
        """
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._half_precision = half_precision
        self._candidates = candidates
        self._endpoint = endpoint
        remote: RerankScorer | None = client
        if remote is None and endpoint:
            remote = RerankerHttpClient(endpoint, timeout=http_timeout)
        self._http = remote
        self._remote_error: str | None = None
        #: До какого момента не трогать удалённый счётчик после сбоя.
        #:
        #: Раньше первая же ошибка выставляла `self._http = None` навсегда, и
        #: короткий сбой сети или перезапуск сервиса выключали переоценку до
        #: перезапуска процесса. Качество при этом падало молча: hit@1 0,829
        #: против 0,561 без переоценки, и в логах об этом было одно давнее
        #: предупреждение. Пауза даёт сервису вернуться самому.
        self._remote_retry_at = 0.0
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

    def _remote_name(self, client: RerankScorer) -> str:
        """Возвращает адрес удалённого счётчика для журнала.

        Args:
            client: Удалённый счётчик оценок.

        Returns:
            str: Адрес эндпоинта либо имя класса, если адреса нет.
        """
        return (
            getattr(client, "endpoint", None) or self._endpoint or type(client).__name__
        )

    @staticmethod
    def _timed(callable_: Any, *args: Any) -> tuple[list[float], float]:
        start = time.perf_counter()
        result = callable_(*args)
        return result, (time.perf_counter() - start) * 1000

    def _score(self, query: str, texts: list[str]) -> list[float]:
        client = self._http if time.monotonic() >= self._remote_retry_at else None
        if client is not None:
            try:
                scores, ms = self._timed(client.predict, query, texts)
                logger.info(
                    "переоценка через сервис %s: %d кандидатов за %.0f мс",
                    self._remote_name(client),
                    len(texts),
                    ms,
                )
                return scores
            except RerankerHttpError as exc:
                logger.warning(
                    "сервис переоценки не отвечает, считаем локально: %s",
                    exc,
                )
                self._remote_error = str(exc)
                self._remote_retry_at = time.monotonic() + REMOTE_COOLDOWN_SECONDS
                logger.info(
                    "удалённая переоценка отключена на %d с, затем повторим",
                    REMOTE_COOLDOWN_SECONDS,
                )
        try:
            scores, ms = self._timed(self._local_scores, query, texts)
        except RerankerUnavailable as exc:
            if self._remote_error is None:
                raise
            # Иначе сообщение говорило бы только про отсутствие
            # sentence-transformers и уводило от настоящей причины: пакета в
            # образе нет намеренно, оценки должен был посчитать сервис.
            raise RerankerUnavailable(
                f"сервис переоценки не ответил ({self._remote_error}), "
                f"а локальный расчёт недоступен: {exc}"
            ) from exc
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
    """Собирает реранкер по настройкам.

    Кто считает оценки, задаётся ``rerank_provider``: ``auto`` - хост-сервис,
    если задан ``rerank_endpoint``, иначе локальный расчёт; ``local`` - только
    локально; ``http`` - только хост-сервис; ``vllm`` - эндпоинт
    ``/v1/rerank`` сервера vLLM.

    Args:
        settings: Настройки приложения.

    Returns:
        CrossEncoderReranker | None: Реранкер либо ``None``, если переоценка
        выключена.

    Raises:
        ValueError: Способ переоценки требует адреса, которого нет в
            настройках.
    """
    if not settings.rerank_enabled:
        return None

    provider = getattr(settings, "rerank_provider", "auto")
    endpoint = settings.rerank_endpoint
    client: RerankScorer | None = None

    if provider == "local":
        endpoint = ""
    elif provider == "http":
        if not endpoint:
            raise ValueError(
                "RERANK_PROVIDER=http требует адреса хост-сервиса в "
                "RERANK_ENDPOINT (например http://host.docker.internal:8899)"
            )
    elif provider == "vllm":
        base_url = settings.resolved_vllm_rerank_base_url()
        if not base_url:
            raise ValueError(
                "RERANK_PROVIDER=vllm требует адреса сервера в "
                "VLLM_RERANK_BASE_URL либо VLLM_BASE_URL"
            )
        client = VLLMRerankerClient(
            base_url,
            model=settings.rerank_model,
            api_key=getattr(settings, "vllm_api_key", ""),
        )
        endpoint = ""
        logger.info("переоценка: vLLM, %s", client.endpoint)

    return CrossEncoderReranker(
        settings.rerank_model,
        device=settings.rerank_device or None,
        half_precision=settings.rerank_half_precision,
        candidates=settings.rerank_candidates,
        endpoint=endpoint,
        client=client,
    )
