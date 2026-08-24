import pytest

from infrastructure.config.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield


class TestSettings:
    def test_default_settings(self):
        settings = get_settings()
        assert settings.llm_provider == "ollama-local"

    def test_retrieval_budget_defaults(self):
        settings = get_settings()
        assert settings.retrieval_max_requests == 10
        assert settings.retrieval_branch_retries == 1

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "vllm")
        settings = get_settings()
        assert settings.llm_provider == "vllm"

    def test_retrieval_budget_env_overrides(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_MAX_REQUESTS", "20")
        monkeypatch.setenv("RETRIEVAL_BRANCH_RETRIES", "0")
        settings = get_settings()
        assert settings.retrieval_max_requests == 20
        assert settings.retrieval_branch_retries == 0

    def test_negative_retries_rejected(self):
        with pytest.raises(ValueError):
            Settings(retrieval_branch_retries=-1)


class TestProviders:
    """Провайдеры моделей проверяются на входе, а не при первом обращении."""

    def test_unknown_llm_provider_names_supported_values(self):
        """Неизвестный провайдер LLM отвергается с перечислением вариантов."""
        with pytest.raises(ValueError, match="ollama-local, llamacpp, vllm"):
            Settings(llm_provider="openai")

    def test_unknown_embedding_provider_rejected(self):
        """Неизвестный провайдер эмбеддингов отвергается так же."""
        with pytest.raises(ValueError, match="embedding_provider"):
            Settings(embedding_provider="openai")

    def test_provider_is_normalized(self):
        """Регистр и пробелы вокруг значения не мешают."""
        assert Settings(llm_provider=" VLLM ").llm_provider == "vllm"

    def test_unknown_rerank_provider_rejected(self):
        """Неизвестный способ переоценки отвергается с перечислением."""
        with pytest.raises(ValueError, match="auto, local, http, vllm"):
            Settings(rerank_provider="cohere")


class TestVLLMSettings:
    """Адреса vLLM: у генерации, эмбеддингов и переоценки они разные."""

    def test_defaults(self):
        """По умолчанию переоценка и ключ доступа настроены консервативно."""
        settings = get_settings()
        assert settings.rerank_provider == "auto"
        assert settings.vllm_api_key == ""

    def test_embedding_url_falls_back_to_common(self):
        """Без отдельного адреса эмбеддер идёт на общий адрес vLLM."""
        settings = Settings(vllm_base_url="http://vllm.test:8000")
        assert settings.resolved_vllm_embedding_base_url() == "http://vllm.test:8000"

    def test_embedding_url_wins_when_set(self):
        """Отдельный адрес эмбеддера перекрывает общий."""
        settings = Settings(
            vllm_base_url="http://vllm.test:8000",
            vllm_embedding_base_url="http://vllm.test:8001",
        )
        assert settings.resolved_vllm_embedding_base_url() == "http://vllm.test:8001"

    def test_rerank_url_wins_when_set(self):
        """Отдельный адрес реранкера перекрывает общий."""
        settings = Settings(
            vllm_base_url="http://vllm.test:8000",
            vllm_rerank_base_url="http://vllm.test:8002",
        )
        assert settings.resolved_vllm_rerank_base_url() == "http://vllm.test:8002"

    def test_api_key_is_hidden_in_repr(self):
        """Ключ доступа не светится в repr настроек."""
        settings = Settings(vllm_api_key="token-abc123")
        assert "token-abc123" not in repr(settings)
