"""Тесты реестра клиентов: сборка провайдеров и отказ на неизвестном.

Сеть не задействована: клиенты создаются лениво и при создании никуда не
ходят, поэтому проверяется именно сборка.
"""

from types import SimpleNamespace

import pytest

from infrastructure.config import model_registry
from infrastructure.config.model_registry import (
    build_embedding_client,
    build_model_client,
    get_embedding,
    get_model,
    setup_registry,
)
from infrastructure.config.settings import Settings
from infrastructure.embeddings.llamacpp_client import LlamaCppEmbeddingClient
from infrastructure.embeddings.ollama_client import OllamaEmbeddingClient
from infrastructure.embeddings.vllm_client import VLLMEmbeddingClient
from infrastructure.llm.llamacpp_client import LlamaCppClient
from infrastructure.llm.ollama_client import OllamaClient
from infrastructure.llm.vllm_client import VLLMClient

VLLM_URL = "http://vllm.test:8000"


@pytest.fixture(autouse=True)
def clean_registry():
    """Очищает глобальные слоты реестра до и после теста.

    Yields:
        None: Управление тесту.
    """
    model_registry._MODELS.clear()
    model_registry._EMBEDDINGS.clear()
    yield
    model_registry._MODELS.clear()
    model_registry._EMBEDDINGS.clear()


class TestBuildModelClient:
    def test_ollama_is_built(self):
        """Провайдер по умолчанию собирает нативный клиент Ollama."""
        client = build_model_client(Settings(llm_provider="ollama-local"), "qwen3:8b")
        assert isinstance(client, OllamaClient)

    def test_llamacpp_is_built(self):
        """llama.cpp собирается, а не игнорируется молча."""
        client = build_model_client(Settings(llm_provider="llamacpp"), "model")
        assert isinstance(client, LlamaCppClient)

    def test_vllm_is_built_with_settings(self):
        """vLLM получает адрес, ключ и режим размышлений из настроек."""
        settings = Settings(
            llm_provider="vllm",
            vllm_base_url=VLLM_URL,
            vllm_api_key="token-abc123",
            llm_repeat_penalty=1.2,
        )
        client = build_model_client(settings, "Qwen/Qwen3-8B")
        assert isinstance(client, VLLMClient)
        assert client.raw.openai_api_base == f"{VLLM_URL}/v1"
        assert client.raw.openai_api_key.get_secret_value() == "token-abc123"
        assert client._extra_body()["repetition_penalty"] == 1.2

    def test_unknown_provider_names_supported_values(self):
        """Нереализованный провайдер отвергается с перечислением вариантов.

        Раньше ветка просто не выполнялась, и приложение падало позже с
        невнятным ``KeyError: 'default'``.
        """
        settings = SimpleNamespace(llm_provider="openai")
        with pytest.raises(ValueError, match="ollama-local, llamacpp, vllm"):
            build_model_client(settings, "model")


class TestBuildEmbeddingClient:
    def test_ollama_is_built(self):
        """Провайдер по умолчанию собирает эмбеддер Ollama."""
        client = build_embedding_client(Settings(embedding_provider="ollama-local"))
        assert isinstance(client, OllamaEmbeddingClient)

    def test_llamacpp_is_built(self):
        """llama.cpp собирается, а не игнорируется молча."""
        client = build_embedding_client(Settings(embedding_provider="llamacpp"))
        assert isinstance(client, LlamaCppEmbeddingClient)

    def test_vllm_uses_its_own_address(self):
        """Эмбеддер vLLM живёт на своём адресе: один процесс - одна модель."""
        settings = Settings(
            embedding_provider="vllm",
            vllm_base_url=VLLM_URL,
            vllm_embedding_base_url="http://vllm.test:8001",
        )
        client = build_embedding_client(settings)
        assert isinstance(client, VLLMEmbeddingClient)
        assert client._client.openai_api_base == "http://vllm.test:8001/v1"

    def test_overrides_win_over_settings(self):
        """Параметры прогона индексации перекрывают настройки."""
        settings = Settings(embedding_provider="ollama-local", embedding_model="bge-m3")
        client = build_embedding_client(settings, model="BAAI/bge-m3", provider="vllm")
        assert isinstance(client, VLLMEmbeddingClient)
        assert client.model_name == "BAAI/bge-m3"

    def test_unknown_provider_names_supported_values(self):
        """Нереализованный провайдер отвергается с перечислением вариантов."""
        settings = SimpleNamespace(embedding_provider="openai", embedding_model="m")
        with pytest.raises(ValueError, match="ollama-local, llamacpp, vllm"):
            build_embedding_client(settings)


class TestSetupRegistry:
    def test_vllm_fills_all_slots(self):
        """С vLLM реестр наполняется целиком, включая слот support."""
        settings = Settings(
            llm_provider="vllm",
            embedding_provider="vllm",
            llm_model="Qwen/Qwen3-8B",
            embedding_model="BAAI/bge-m3",
            vllm_base_url=VLLM_URL,
        )
        setup_registry(settings)
        assert isinstance(get_model("default"), VLLMClient)
        assert get_model("support") is get_model("default")
        assert isinstance(get_embedding("default"), VLLMEmbeddingClient)

    def test_support_model_can_differ(self):
        """Отдельная вспомогательная модель собирается своим клиентом."""
        settings = Settings(
            llm_provider="vllm",
            llm_model="Qwen/Qwen3-8B",
            llm_support_model="Qwen/Qwen3-1.7B",
            vllm_base_url=VLLM_URL,
        )
        setup_registry(settings)
        assert get_model("support") is not get_model("default")
        assert get_model("support").model_name == "Qwen/Qwen3-1.7B"
