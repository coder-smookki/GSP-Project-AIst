import pytest

from infrastructure.config.settings import get_settings
from infrastructure.llm.ollama_client import OllamaClient


class TestOllamaClient:
    @pytest.fixture
    def client(self):
        settings = get_settings()
        return OllamaClient(
            model=settings.llm_model, base_url=settings.ollama_local_base_url
        )

    @pytest.mark.integration
    def test_invoke_returns_response(self, client):
        result = client.invoke([{"role": "user", "content": "Say 'hello' in one word"}])
        assert isinstance(result.content, str)
        assert len(result.content) > 0

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
