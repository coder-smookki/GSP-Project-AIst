"""Юнит-тесты клиента эмбеддингов на vLLM.

Сеть подменяет pytest-httpx: проверяется тело запроса и разбор ответа
``/v1/embeddings``.
"""

import json

import pytest
from openai import NotFoundError

from infrastructure.embeddings.vllm_client import VLLMEmbeddingClient
from infrastructure.llm.vllm_client import NO_KEY

BASE_URL = "http://vllm.test:8001"
EMBED_URL = f"{BASE_URL}/v1/embeddings"
MODEL = "BAAI/bge-m3"


def _vectors(*vectors: list[float]) -> dict:
    """Собирает ответ ``/v1/embeddings`` в формате OpenAI.

    Args:
        *vectors: Векторы в порядке входных текстов.

    Returns:
        dict: Тело ответа сервера.
    """
    return {
        "object": "list",
        "model": MODEL,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": len(vectors), "total_tokens": len(vectors)},
    }


@pytest.fixture
def client():
    """Создаёт клиент к подменённому серверу.

    Returns:
        VLLMEmbeddingClient: Клиент эмбеддингов.
    """
    return VLLMEmbeddingClient(model=MODEL, base_url=BASE_URL, api_key="token-abc123")


class TestVLLMEmbeddingClient:
    def test_base_url_gets_v1_suffix(self, client):
        """Адрес из настроек дополняется OpenAI-совместимым префиксом."""
        assert client._client.openai_api_base == f"{BASE_URL}/v1"

    def test_empty_api_key_becomes_placeholder(self):
        """Без ключа подставляется заглушка: SDK не работает с пустым."""
        other = VLLMEmbeddingClient(model=MODEL, base_url=BASE_URL)
        assert other._client.openai_api_key.get_secret_value() == NO_KEY

    def test_embed_returns_vectors_in_order(self, httpx_mock, client):
        """Векторы возвращаются в порядке входных текстов."""
        httpx_mock.add_response(url=EMBED_URL, json=_vectors([0.1, 0.2], [0.3, 0.4]))
        assert client.embed(["раз", "два"]) == [[0.1, 0.2], [0.3, 0.4]]

    def test_texts_are_sent_as_text(self, httpx_mock, client):
        """На сервер уходит текст, а не номера токенов чужого токенизатора.

        Резка по контексту силами langchain идёт токенизатором OpenAI; для
        модели на vLLM это чужие номера токенов и мусорные векторы.
        """
        httpx_mock.add_response(url=EMBED_URL, json=_vectors([0.1, 0.2]))
        client.embed(["очень длинный текст"])
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["input"] == ["очень длинный текст"]
        assert body["model"] == MODEL

    def test_api_key_goes_into_authorization_header(self, httpx_mock, client):
        """Ключ уходит заголовком ``Authorization: Bearer``."""
        httpx_mock.add_response(url=EMBED_URL, json=_vectors([0.1, 0.2]))
        client.embed_query("вопрос")
        request = httpx_mock.get_requests()[0]
        assert request.headers["Authorization"] == "Bearer token-abc123"

    def test_embedding_dim_is_measured_once(self, httpx_mock, client):
        """Размерность определяется одним пробным запросом и запоминается."""
        httpx_mock.add_response(url=EMBED_URL, json=_vectors([0.1, 0.2, 0.3]))
        assert client.embedding_dim == 3
        assert client.embedding_dim == 3
        assert len(httpx_mock.get_requests()) == 1

    async def test_aembed_returns_vectors(self, httpx_mock, client):
        """Асинхронный путь разбирает ответ так же, как синхронный."""
        httpx_mock.add_response(url=EMBED_URL, json=_vectors([0.1], [0.2]))
        assert await client.aembed(["раз", "два"]) == [[0.1], [0.2]]

    def test_missing_model_raises(self, httpx_mock, client):
        """Отсутствие модели на сервере видно как ошибка, а не пустой вектор."""
        httpx_mock.add_response(
            url=EMBED_URL,
            status_code=404,
            json={"error": {"message": "model not found"}},
        )
        with pytest.raises(NotFoundError):
            client.embed(["раз"])
