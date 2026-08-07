from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage

from infrastructure.llm.llamacpp_client import LlamaCppClient
from infrastructure.llm.ollama_client import OllamaClient


@dataclass
class _Call:
    method: str
    tools: list | None
    kwargs: dict = field(default_factory=dict)


class FakeClient:
    """Фейк ChatOpenAI: записывает kwarg'и, без сетевых вызовов."""

    def __init__(self):
        self.calls: list[_Call] = []
        self.model = "fake-model"

    def _record(self, method, tools, kwargs):
        self.calls.append(_Call(method, tools, dict(kwargs)))

    def bind_tools(self, tools):
        parent = self
        bound_tools = tools

        class _Bound:
            def invoke(self, messages, **kwargs):
                parent._record("invoke", bound_tools, kwargs)
                return AIMessage(content="hi")

            async def ainvoke(self, messages, **kwargs):
                parent._record("ainvoke", bound_tools, kwargs)
                return AIMessage(content="hi")

            def stream(self, messages, **kwargs):
                parent._record("stream", bound_tools, kwargs)
                yield AIMessage(content="tok")

            async def astream(self, messages, **kwargs):
                parent._record("astream", bound_tools, kwargs)
                yield AIMessage(content="tok")

        return _Bound()

    def invoke(self, messages, **kwargs):
        self._record("invoke", None, kwargs)
        return AIMessage(content="hi")

    async def ainvoke(self, messages, **kwargs):
        self._record("ainvoke", None, kwargs)
        return AIMessage(content="hi")

    def stream(self, messages, **kwargs):
        self._record("stream", None, kwargs)
        yield AIMessage(content="tok")

    async def astream(self, messages, **kwargs):
        self._record("astream", None, kwargs)
        yield AIMessage(content="tok")


EXPECTED = {"num_ctx": 2048, "repeat_penalty": 1.1}
EXPECTED_BODY = {"options": EXPECTED}


@pytest.fixture
def ollama(monkeypatch):
    client = OllamaClient(
        model="test-model",
        base_url="http://fake:11434",
        num_ctx=EXPECTED["num_ctx"],
        repeat_penalty=EXPECTED["repeat_penalty"],
    )
    fake = FakeClient()
    monkeypatch.setattr(client, "_client", fake)
    return client, fake


class TestOptionsPropagation:
    def test_invoke_gets_extra_body(self, ollama):
        impl, fake = ollama
        impl.invoke([{"role": "user", "content": "q"}])
        assert fake.calls[0].kwargs["extra_body"] == EXPECTED_BODY

    @pytest.mark.asyncio
    async def test_ainvoke_gets_extra_body(self, ollama):
        impl, fake = ollama
        await impl.ainvoke([{"role": "user", "content": "q"}])
        assert fake.calls[0].kwargs["extra_body"] == EXPECTED_BODY

    def test_stream_gets_extra_body(self, ollama):
        impl, fake = ollama
        chunks = list(impl.stream([{"role": "user", "content": "q"}]))
        assert chunks == ["tok"]
        assert fake.calls[0].kwargs["extra_body"] == EXPECTED_BODY

    @pytest.mark.asyncio
    async def test_astream_gets_extra_body(self, ollama):
        impl, fake = ollama
        chunks = [c async for c in impl.astream([{"role": "user", "content": "q"}])]
        assert chunks == ["tok"]
        assert fake.calls[0].kwargs["extra_body"] == EXPECTED_BODY

    def test_user_kwargs_not_erased(self, ollama):
        impl, fake = ollama
        impl.invoke([{"role": "user", "content": "q"}], temperature=0.2)
        assert fake.calls[0].kwargs["temperature"] == 0.2
        assert fake.calls[0].kwargs["extra_body"] == EXPECTED_BODY

    def test_tools_use_bound_path(self, ollama):
        impl, fake = ollama
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
        impl.invoke([{"role": "user", "content": "q"}], tools=tools)
        assert fake.calls[0].tools is tools
        assert fake.calls[0].kwargs["extra_body"] == EXPECTED_BODY


class TestNoOptions:
    def test_llama_sends_no_extra_body(self, monkeypatch):
        client = LlamaCppClient(model="m", base_url="http://fake:8080")
        fake = FakeClient()
        monkeypatch.setattr(client, "_client", fake)
        client.invoke([{"role": "user", "content": "q"}])
        assert "extra_body" not in fake.calls[0].kwargs
