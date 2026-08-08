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


class TestOllamaNativeClient:
    """Клиент Ollama работает через нативный API, а не через слой OpenAI.

    Переход сделан ради скорости: через OpenAI-совместимый эндпоинт нельзя
    выключить размышления у моделей Qwen3, и они стоят шестикратного времени
    ответа (замер на qwen3:8b: 16,3 с против 2,5 с при одном и том же тексте
    ответа). Поэтому `num_ctx`, `repeat_penalty` и `reasoning` задаются при
    создании клиента, а не подмешиваются в каждый вызов через `extra_body`.
    """

    def test_options_go_into_client_not_into_calls(self, ollama):
        """Параметры модели заданы у клиента, а не приходят с каждым вызовом."""
        impl, fake = ollama
        impl.invoke([{"role": "user", "content": "q"}])
        assert "extra_body" not in fake.calls[0].kwargs

    def test_reasoning_is_disabled_by_default(self):
        """Размышления выключены по умолчанию — это выбор в пользу скорости."""
        client = OllamaClient(model="m", base_url="http://fake:11434")
        assert client.raw.reasoning is False

    def test_reasoning_can_be_enabled(self):
        """Размышления включаются явно, если замер покажет выигрыш качества."""
        client = OllamaClient(model="m", base_url="http://fake:11434", reasoning=True)
        assert client.raw.reasoning is True

    def test_model_options_reach_the_client(self):
        """Размер контекста и штраф за повторы доходят до модели."""
        client = OllamaClient(
            model="m",
            base_url="http://fake:11434",
            num_ctx=EXPECTED["num_ctx"],
            repeat_penalty=EXPECTED["repeat_penalty"],
        )
        assert client.raw.num_ctx == EXPECTED["num_ctx"]
        assert client.raw.repeat_penalty == EXPECTED["repeat_penalty"]

    def test_user_kwargs_not_erased(self, ollama):
        """Параметры конкретного вызова доходят без изменений."""
        impl, fake = ollama
        impl.invoke([{"role": "user", "content": "q"}], temperature=0.2)
        assert fake.calls[0].kwargs["temperature"] == 0.2

    def test_tools_use_bound_path(self, ollama):
        """Инструменты уходят через bind_tools, а не в теле вызова."""
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

    @pytest.mark.asyncio
    async def test_astream_yields_tokens(self, ollama):
        """Асинхронный поток отдаёт содержимое фрагментов."""
        impl, _ = ollama
        chunks = [c async for c in impl.astream([{"role": "user", "content": "q"}])]
        assert chunks == ["tok"]


class TestNoOptions:
    def test_llama_sends_no_extra_body(self, monkeypatch):
        client = LlamaCppClient(model="m", base_url="http://fake:8080")
        fake = FakeClient()
        monkeypatch.setattr(client, "_client", fake)
        client.invoke([{"role": "user", "content": "q"}])
        assert "extra_body" not in fake.calls[0].kwargs
