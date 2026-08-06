from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol

from langchain_core.messages import AIMessage


class ModelClient(Protocol):
    def invoke(
        self, messages: list, *, tools: list | None = None, **kwargs: Any
    ) -> AIMessage: ...

    async def ainvoke(
        self, messages: list, *, tools: list | None = None, **kwargs: Any
    ) -> AIMessage: ...

    def stream(self, messages: list, **kwargs: Any) -> Iterator[str]: ...

    def astream(self, messages: list, **kwargs: Any) -> AsyncIterator[str]: ...
