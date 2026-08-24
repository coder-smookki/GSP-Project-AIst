"""Интеграционные тесты клиентов vLLM: живой сервер.

Организаторы кейса дают облачный VPS, и связка проверяется на нём целиком:
генерация, эмбеддинги и переоценка. Один процесс vLLM обслуживает одну
модель, поэтому адреса берутся из разных настроек и каждая группа тестов
пропускается независимо.

Запуск: ``pytest -m vllm``. Без сервера тесты пропускаются с подсказкой, а не
падают.
"""

import httpx
import pytest

from infrastructure.config.settings import get_settings
from infrastructure.embeddings.vllm_client import VLLMEmbeddingClient
from infrastructure.llm.vllm_client import VLLMClient
from infrastructure.retrieval.rerank import RerankerHttpError, VLLMRerankerClient

pytestmark = [pytest.mark.integration, pytest.mark.vllm]


def _served_models(base_url: str, api_key: str) -> list[str]:
    """Спрашивает у сервера список обслуживаемых моделей.

    Args:
        base_url: Адрес сервера без суффикса ``/v1``.
        api_key: Ключ доступа; пусто - заголовок не шлётся.

    Returns:
        list[str]: Имена моделей; пустой список, если сервер недоступен.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/v1/models", headers=headers, timeout=3.0
        )
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", [])]
    except Exception:
        return []


def _require_model(base_url: str, api_key: str, wanted: str, hint: str) -> str:
    """Возвращает имя модели на сервере либо пропускает тест.

    Args:
        base_url: Адрес сервера без суффикса ``/v1``.
        api_key: Ключ доступа.
        wanted: Желаемая модель из настроек.
        hint: Имя переменной окружения с адресом сервера.

    Returns:
        str: Имя модели, которую реально обслуживает сервер.
    """
    models = _served_models(base_url, api_key)
    if not models:
        pytest.skip(
            f"сервер vLLM недоступен по адресу {base_url}: запустите "
            f"`vllm serve <модель>` и задайте {hint}"
        )
    # Один процесс обслуживает одну модель: если имя из настроек не совпало,
    # берём то, что сервер реально отдаёт, - тест про клиент, а не про имена.
    return wanted if wanted in models else models[0]


class TestVLLMChatModel:
    @pytest.fixture
    def client(self):
        """Создаёт клиент генерации к живому серверу.

        Returns:
            VLLMClient: Клиент чат-модели.
        """
        settings = get_settings()
        model = _require_model(
            settings.vllm_base_url,
            settings.vllm_api_key,
            settings.llm_model,
            "VLLM_BASE_URL",
        )
        return VLLMClient(
            model=model,
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            timeout=settings.vllm_timeout,
        )

    def test_invoke_returns_response(self, client):
        """Модель отвечает текстом."""
        result = client.invoke([{"role": "user", "content": "Ответь одним словом: да"}])
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    async def test_astream_returns_chunks(self, client):
        """Ответ приходит потоком по частям."""
        chunks = [
            chunk
            async for chunk in client.astream(
                [{"role": "user", "content": "Посчитай от 1 до 5, по одному в строке"}]
            )
        ]
        assert len(chunks) > 0
        text = "".join(chunks)
        assert "1" in text
        assert "2" in text


class TestVLLMEmbeddings:
    @pytest.fixture
    def client(self):
        """Создаёт клиент эмбеддингов к живому серверу.

        Returns:
            VLLMEmbeddingClient: Клиент эмбеддингов.
        """
        settings = get_settings()
        base_url = settings.resolved_vllm_embedding_base_url()
        model = _require_model(
            base_url,
            settings.vllm_api_key,
            settings.embedding_model,
            "VLLM_EMBEDDING_BASE_URL",
        )
        return VLLMEmbeddingClient(
            model=model,
            base_url=base_url,
            api_key=settings.vllm_api_key,
            timeout=settings.vllm_timeout,
        )

    def test_embed_returns_vectors(self, client):
        """Документы кодируются в векторы одинаковой длины."""
        vectors = client.embed(["привет", "мир"])
        assert len(vectors) == 2
        assert len(vectors[0]) == len(vectors[1]) > 0

    def test_embedding_dim_is_positive(self, client):
        """Размерность вектора определяется и она положительная."""
        assert client.embedding_dim > 0


class TestVLLMReranker:
    @pytest.fixture
    def client(self):
        """Создаёт клиент переоценки к живому серверу.

        Returns:
            VLLMRerankerClient: Клиент переоценки.
        """
        settings = get_settings()
        base_url = settings.resolved_vllm_rerank_base_url()
        model = _require_model(
            base_url,
            settings.vllm_api_key,
            settings.rerank_model,
            "VLLM_RERANK_BASE_URL",
        )
        client = VLLMRerankerClient(
            base_url, model=model, api_key=settings.vllm_api_key
        )
        try:
            client.predict("проверка", ["текст"])
        except RerankerHttpError as exc:
            pytest.skip(f"переоценка на {base_url} недоступна: {exc}")
        return client

    def test_relevant_candidate_scores_higher(self, client):
        """Релевантный кандидат получает оценку выше постороннего."""
        scores = client.predict(
            "Какой срок действия акта освидетельствования?",
            [
                "Небо сегодня ясное, ветер северный.",
                "Акт освидетельствования скрытых работ действует до сдачи объекта.",
            ],
        )
        assert len(scores) == 2
        assert scores[1] > scores[0]
