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
