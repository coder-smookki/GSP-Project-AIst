import pytest

from infrastructure.config.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield


class TestSettings:
    def test_default_settings(self):
        settings = get_settings()
        assert settings.llm_provider == "ollama-local"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "vllm")
        settings = get_settings()
        assert settings.llm_provider == "vllm"
