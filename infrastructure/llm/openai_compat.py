from collections.abc import AsyncIterator, Iterator

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class OpenAICompatClient:
    """Общий клиент для OpenAI-совместимых серверов (Ollama, llama.cpp, vLLM).

    Единая сборка extra_body={"options": {...}} передаётся во все вызовы,
    чтобы num_ctx / repeat_penalty одинаково работали в invoke, stream,
    ainvoke и astream.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 60.0,
        num_ctx: int | None = None,
        repeat_penalty: float | None = None,
        api_key: str = "sk-no-key",
    ):
        self._client = ChatOpenAI(
            model=model,
            base_url=f"{base_url}/v1",
            api_key=SecretStr(api_key),
            timeout=timeout,
        )
        self._num_ctx = num_ctx
        self._repeat_penalty = repeat_penalty

    @property
    def model_name(self) -> str:
        return self._client.model

    @property
    def raw(self) -> ChatOpenAI:
        return self._client

    def _extra_body(self) -> dict | None:
        options = {}
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        if self._repeat_penalty is not None:
            options["repeat_penalty"] = self._repeat_penalty
        return {"options": options} if options else None

    def _bound(self, tools: list | None):
        return self._client.bind_tools(tools) if tools else self._client

    def invoke(self, messages, *, tools: list | None = None, **kwargs) -> AIMessage:
        extra = self._extra_body()
        if extra is not None:
            kwargs.setdefault("extra_body", extra)
        return self._bound(tools).invoke(messages, **kwargs)

    async def ainvoke(
        self, messages: list, *, tools: list | None = None, **kwargs
    ) -> AIMessage:
        extra = self._extra_body()
        if extra is not None:
            kwargs.setdefault("extra_body", extra)
        return await self._bound(tools).ainvoke(messages, **kwargs)

    def stream(self, messages: list, **kwargs) -> Iterator[str]:
        extra = self._extra_body()
        if extra is not None:
            kwargs.setdefault("extra_body", extra)
        for chunk in self._client.stream(messages, **kwargs):
            if chunk.content:
                yield str(chunk.content)

    async def astream(self, messages: list, **kwargs) -> AsyncIterator[str]:
        extra = self._extra_body()
        if extra is not None:
            kwargs.setdefault("extra_body", extra)
        async for chunk in self._client.astream(messages, **kwargs):
            if chunk.content:
                yield str(chunk.content)