import pytest

from infrastructure.config.settings import get_settings
from infrastructure.llm.ollama_client import OllamaClient


def _model_is_available(base_url: str, model: str) -> bool:
    """Проверяет, скачана ли модель в Ollama.

    Args:
        base_url: Адрес Ollama.
        model: Имя модели из настроек.

    Returns:
        bool: ``True``, если модель есть в списке скачанных.
    """
    try:
        import httpx

        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        response.raise_for_status()
        names = {item["name"] for item in response.json().get("models", [])}
    except Exception:
        return False
    # Ollama показывает модели с тегом; без тега подразумевается latest.
    return model in names or f"{model}:latest" in names


class TestOllamaClient:
    @pytest.fixture
    def client(self):
        settings = get_settings()
        if not _model_is_available(settings.ollama_local_base_url, settings.llm_model):
            # Внятный пропуск вместо сырого 404 из недр клиента. Модель по
            # умолчанию в настройках рассчитана на демо-машину, и на машине
            # разработчика её обычно нет; падение теста здесь означало бы
            # поломку кода, которой на самом деле нет.
            pytest.skip(
                f"модель {settings.llm_model} не скачана в Ollama: "
                f"`ollama pull {settings.llm_model}` либо задайте LLM_MODEL"
            )
        return OllamaClient(
            model=settings.llm_model, base_url=settings.ollama_local_base_url
        )

    @pytest.mark.llm
    @pytest.mark.integration
    def test_invoke_returns_response(self, client):
        result = client.invoke([{"role": "user", "content": "Say 'hello' in one word"}])
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    @pytest.mark.llm
    @pytest.mark.integration
    def test_invoke_accepts_tools(self, client):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = client.invoke(
            [{"role": "user", "content": "What's the weather like?"}],
            tools=tools,
        )
        assert isinstance(result.content, str)

    @pytest.mark.asyncio
    @pytest.mark.llm
    @pytest.mark.integration
    async def test_astream_returns_chunks(self, client):
        chunks = []
        async for chunk in client.astream(
            [{"role": "user", "content": "Count from 1 to 5, one per line"}]
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        text = "".join(chunks)
        assert "1" in text
        assert "2" in text
