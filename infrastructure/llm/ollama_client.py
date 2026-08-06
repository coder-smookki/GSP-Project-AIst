from collections.abc import AsyncIterator, Iterator

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        num_ctx: int = 8192,
        repeat_penalty: float = 1.1,
    ):
        self._client = ChatOpenAI(
            model=model,
            base_url=f"{base_url}/v1",
            api_key=SecretStr("sk-no-key"),
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

    def invoke(self, messages, *, tools=None, **kwargs):
        if tools:
            bound_model = self._client.bind_tools(tools)
            return bound_model.invoke(messages, **kwargs)
        return self._client.invoke(messages, **kwargs)

    async def ainvoke(
        self, messages: list, *, tools: list | None = None, **kwargs
    ) -> AIMessage:
        extra = {
            "extra_body": {
                "options": {
                    "num_ctx": self._num_ctx,
                    "repeat_penalty": self._repeat_penalty,
                }
            }
        }
        if tools:
            bound_model = self._client.bind_tools(tools)
            return await bound_model.ainvoke(messages, **extra, **kwargs)
        return await self._client.ainvoke(messages, **extra, **kwargs)

    def stream(self, messages: list, **kwargs) -> Iterator[str]:
        for chunk in self._client.stream(messages, **kwargs):
            if chunk.content:
                yield str(chunk.content)

    async def astream(self, messages: list, **kwargs) -> AsyncIterator[str]:
        async for chunk in self._client.astream(messages, **kwargs):
            if chunk.content:
                yield str(chunk.content)
