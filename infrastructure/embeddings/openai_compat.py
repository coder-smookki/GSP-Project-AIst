"""Общий клиент эмбеддингов для OpenAI-совместимых серверов.

Ollama, llama.cpp и vLLM отдают эмбеддинги по одному и тому же пути
``/v1/embeddings``, поэтому различия сводятся к адресу, ключу и тому, надо ли
резать текст по контексту силами langchain.
"""

from typing import Any

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


class OpenAICompatEmbeddingClient:
    """Эмбеддер поверх OpenAI-совместимого сервера."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str = "sk-no-key",
        check_embedding_ctx_length: bool = True,
        timeout: float | None = None,
    ):
        """Создаёт клиент.

        Args:
            model: Имя модели эмбеддингов на сервере.
            base_url: Адрес сервера без суффикса ``/v1``; суффикс
                добавляется сам.
            api_key: Ключ доступа. Локальные серверы его не проверяют, но
                OpenAI-совместимый SDK не работает с пустым значением.
            check_embedding_ctx_length: Резать ли текст по контексту силами
                langchain. Резка идёт токенизатором OpenAI (tiktoken) и на
                сервер уходят номера токенов, а не текст, поэтому для
                моделей не от OpenAI её выключают.
            timeout: Предел ожидания ответа, с; ``None`` - умолчание SDK.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "base_url": f"{base_url}/v1",
            "api_key": SecretStr(api_key),
            "check_embedding_ctx_length": check_embedding_ctx_length,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = OpenAIEmbeddings(**kwargs)
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        """Возвращает имя модели.

        Returns:
            str: Имя модели эмбеддингов.
        """
        return self._client.model

    @property
    def embedding_dim(self) -> int:
        """Возвращает размерность вектора.

        Определяется одним пробным запросом и запоминается: размерность
        прошита в схему коллекции Qdrant и на лету не меняется.

        Returns:
            int: Число измерений вектора.
        """
        if self._dim is None:
            self._dim = len(self.embed_query("test"))
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Кодирует документы.

        Args:
            texts: Тексты документов.

        Returns:
            list[list[float]]: Векторы в порядке входных текстов.
        """
        return self._client.embed_documents(texts)

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Асинхронно кодирует документы.

        Args:
            texts: Тексты документов.

        Returns:
            list[list[float]]: Векторы в порядке входных текстов.
        """
        return await self._client.aembed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Кодирует запрос.

        Args:
            text: Текст запроса.

        Returns:
            list[float]: Вектор запроса.
        """
        return self._client.embed_query(text)
