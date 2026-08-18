"""Юнит-тесты клиента LLM на vLLM: сборка запроса и разбор ответа.

Сеть не задействована: HTTP подменяет pytest-httpx, поэтому тесты проверяют
ровно то, что уходит на сервер и что клиент понимает в ответе.
"""

import json

import pytest
from openai import AuthenticationError
from pytest_httpx import IteratorStream

from infrastructure.llm.vllm_client import NO_KEY, VLLMClient

BASE_URL = "http://vllm.test:8000"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
MODEL = "Qwen/Qwen3-8B"


def _completion(text: str) -> dict:
    """Собирает ответ ``/v1/chat/completions`` в формате OpenAI.

    Args:
        text: Текст ответа модели.

    Returns:
        dict: Тело ответа сервера.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chunk(text: str) -> bytes:
    """Собирает событие потока в формате SSE.

    Args:
        text: Кусок ответа модели.

    Returns:
        bytes: Одно событие ``data:`` с переводами строк.
    """
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class TestRequestShape:
    """Тело запроса собирается по правилам vLLM, а не Ollama."""

    def test_base_url_gets_v1_suffix(self):
        """Адрес из настроек дополняется OpenAI-совместимым префиксом."""
        client = VLLMClient(model=MODEL, base_url=BASE_URL)
        assert client.raw.openai_api_base == f"{BASE_URL}/v1"

    def test_empty_api_key_becomes_placeholder(self):
        """Без ключа подставляется заглушка: SDK не работает с пустым."""
        client = VLLMClient(model=MODEL, base_url=BASE_URL, api_key="")
        assert client.raw.openai_api_key.get_secret_value() == NO_KEY

    def test_options_of_ollama_are_not_sent(self):
        """Расширение Ollama ``options`` в запрос к vLLM не попадает."""
        client = VLLMClient(model=MODEL, base_url=BASE_URL, repeat_penalty=1.1)
        assert "options" not in (client._extra_body() or {})

    def test_repeat_penalty_uses_vllm_name(self):
        """Штраф за повторы уходит как ``repetition_penalty``."""
        client = VLLMClient(model=MODEL, base_url=BASE_URL, repeat_penalty=1.2)
        assert client._extra_body()["repetition_penalty"] == 1.2

    def test_reasoning_is_disabled_by_default(self):
        """Размышления выключены по умолчанию - это выбор в пользу скорости."""
        client = VLLMClient(model=MODEL, base_url=BASE_URL)
        assert client._extra_body()["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_reasoning_can_be_enabled(self):
        """Размышления включаются явно."""
        client = VLLMClient(model=MODEL, base_url=BASE_URL, reasoning=True)
        assert client._extra_body()["chat_template_kwargs"] == {"enable_thinking": True}

    def test_nothing_is_sent_when_there_are_no_options(self):
        """Пустые параметры не превращаются в пустое ``extra_body``."""
        client = VLLMClient(
            model=MODEL, base_url=BASE_URL, repeat_penalty=None, reasoning=None
        )
        assert client._extra_body() is None


class TestHttpExchange:
    """Обмен с сервером: заголовки, тело запроса, разбор ответа."""

    @pytest.fixture
    def client(self):
        """Создаёт клиент с ключом и штрафом за повторы.

        Returns:
            VLLMClient: Клиент к подменённому серверу.
        """
        return VLLMClient(
            model=MODEL,
            base_url=BASE_URL,
            repeat_penalty=1.1,
            api_key="token-abc123",
        )

    def test_invoke_returns_model_answer(self, httpx_mock, client):
        """Ответ сервера превращается в сообщение с текстом."""
        httpx_mock.add_response(url=CHAT_URL, json=_completion("привет"))
        result = client.invoke([{"role": "user", "content": "вопрос"}])
        assert result.content == "привет"

    def test_api_key_goes_into_authorization_header(self, httpx_mock, client):
        """Ключ уходит заголовком ``Authorization: Bearer``."""
        httpx_mock.add_response(url=CHAT_URL, json=_completion("привет"))
        client.invoke([{"role": "user", "content": "вопрос"}])
        request = httpx_mock.get_requests()[0]
        assert request.headers["Authorization"] == "Bearer token-abc123"

    def test_extra_options_reach_the_server(self, httpx_mock, client):
        """Штраф за повторы и выключенные размышления доходят до сервера."""
        httpx_mock.add_response(url=CHAT_URL, json=_completion("привет"))
        client.invoke([{"role": "user", "content": "вопрос"}])
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["repetition_penalty"] == 1.1
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["model"] == MODEL

    def test_call_kwargs_are_not_erased(self, httpx_mock, client):
        """Параметры конкретного вызова доходят без изменений."""
        httpx_mock.add_response(url=CHAT_URL, json=_completion("привет"))
        client.invoke([{"role": "user", "content": "вопрос"}], temperature=0.7)
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["temperature"] == 0.7

    def test_unauthorized_raises(self, httpx_mock, client):
        """Неверный ключ виден как ошибка авторизации, а не как пустой ответ."""
        httpx_mock.add_response(
            url=CHAT_URL, status_code=401, json={"error": "Unauthorized"}
        )
        with pytest.raises(AuthenticationError):
            client.invoke([{"role": "user", "content": "вопрос"}])

    async def test_astream_yields_tokens(self, httpx_mock, client):
        """Поток разбирается по событиям SSE и отдаёт куски текста."""
        httpx_mock.add_response(
            url=CHAT_URL,
            stream=IteratorStream([_chunk("при"), _chunk("вет"), b"data: [DONE]\n\n"]),
            headers={"Content-Type": "text/event-stream"},
        )
        chunks = [
            chunk async for chunk in client.astream([{"role": "user", "content": "q"}])
        ]
        assert "".join(chunks) == "привет"
