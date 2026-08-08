"""Клиент Ollama поверх нативного API.

Почему не OpenAI-совместимый слой, как у остальных провайдеров: через него
**невозможно выключить размышления** у моделей семейства Qwen3. Замерено на
`qwen3:8b`, один и тот же вопрос с одним и тем же контекстом:

===========================================  =========  ==========
Способ                                       Время      Токенов
===========================================  =========  ==========
нативный ``/api/chat``, как есть              16,3 с    271
нативный ``/api/chat`` с ``think: false``    **2,5 с**  **40**
``/v1/chat/completions`` + ``think: false``   37,7 с    265 (флаг проигнорирован)
``/no_think`` в тексте промпта                13,8 с    205 (не работает)
===========================================  =========  ==========

Шестикратная разница при **одинаковом ответе** — это прямо тот показатель,
который жюри мерит секундомером.

Хорошая новость, тоже проверенная: утечки размышлений в ответ нет ни на одном
из путей. OpenAI-совместимый слой кладёт их в отдельное поле ``reasoning``,
нативный — в ``thinking``; в ``content`` ``<think>`` не попадает. То есть
выключение размышлений нужно ради скорости, а не ради чистоты вывода.

Для gpt-oss управление другое — ``reasoning_effort`` (low/medium/high), и оно
работает через оба слоя. Поэтому параметр `reasoning` здесь трактуется
широко: ``False`` означает «не тратить токены на размышления любым доступным
модели способом».
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama


class OllamaClient:
    """Клиент чат-модели Ollama через нативный API."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        num_ctx: int = 8192,
        repeat_penalty: float = 1.1,
        reasoning: bool | None = False,
        temperature: float = 0.0,
    ):
        """Создаёт клиент.

        Args:
            model: Имя модели в Ollama, например ``qwen3:8b``.
            base_url: Адрес сервера Ollama.
            timeout: Предел ожидания ответа, с. Значение по умолчанию
                намеренно большое: на слабой машине первый запрос ещё и
                загружает модель в память, а обрыв на середине выглядит как
                поломка приложения.
            num_ctx: Размер контекста модели.
            repeat_penalty: Штраф за повторы.
            reasoning: Режим размышлений. ``False`` — выключить (быстро),
                ``True`` — включить, ``None`` — оставить как у модели.
            temperature: Температура. Ноль нужен для стабильности отказов и
                структурированного вывода.
        """
        self._model_name = model
        self._client = ChatOllama(
            model=model,
            base_url=base_url,
            num_ctx=num_ctx,
            repeat_penalty=repeat_penalty,
            temperature=temperature,
            reasoning=reasoning,
            client_kwargs={"timeout": timeout},
        )

    @property
    def model_name(self) -> str:
        """Возвращает имя модели.

        Returns:
            str: Имя модели в Ollama.
        """
        return self._model_name

    @property
    def raw(self) -> ChatOllama:
        """Возвращает клиент langchain.

        Returns:
            ChatOllama: Нижележащий клиент для случаев, когда нужен он сам.
        """
        return self._client

    def _bound(self, tools: list | None):
        """Привязывает инструменты, если они заданы.

        Args:
            tools: Список инструментов или ``None``.

        Returns:
            Клиент с привязанными инструментами либо исходный клиент.
        """
        return self._client.bind_tools(tools) if tools else self._client

    def invoke(
        self, messages: list, *, tools: list | None = None, **kwargs: Any
    ) -> AIMessage:
        """Синхронно получает ответ модели.

        Args:
            messages: Сообщения диалога.
            tools: Инструменты для вызова моделью.
            **kwargs: Дополнительные параметры вызова.

        Returns:
            AIMessage: Ответ модели.
        """
        return self._bound(tools).invoke(messages, **kwargs)

    async def ainvoke(
        self, messages: list, *, tools: list | None = None, **kwargs: Any
    ) -> AIMessage:
        """Асинхронно получает ответ модели.

        Args:
            messages: Сообщения диалога.
            tools: Инструменты для вызова моделью.
            **kwargs: Дополнительные параметры вызова.

        Returns:
            AIMessage: Ответ модели.
        """
        return await self._bound(tools).ainvoke(messages, **kwargs)

    def stream(self, messages: list, **kwargs: Any) -> Iterator[str]:
        """Синхронно отдаёт ответ по частям.

        Args:
            messages: Сообщения диалога.
            **kwargs: Дополнительные параметры вызова.

        Yields:
            str: Очередной фрагмент ответа.
        """
        for chunk in self._client.stream(messages, **kwargs):
            if chunk.content:
                yield str(chunk.content)

    async def astream(self, messages: list, **kwargs: Any) -> AsyncIterator[str]:
        """Асинхронно отдаёт ответ по частям.

        Args:
            messages: Сообщения диалога.
            **kwargs: Дополнительные параметры вызова.

        Yields:
            str: Очередной фрагмент ответа.
        """
        async for chunk in self._client.astream(messages, **kwargs):
            if chunk.content:
                yield str(chunk.content)
