"""Клиент эмбеддингов на сервере vLLM.

vLLM отдаёт эмбеддинги по OpenAI-совместимому пути ``/v1/embeddings``;
авторизация, если сервер запущен с ``--api-key``, идёт заголовком
``Authorization: Bearer <ключ>``.

Источник: https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server
"""

from infrastructure.llm.vllm_client import NO_KEY

from .openai_compat import OpenAICompatEmbeddingClient


class VLLMEmbeddingClient(OpenAICompatEmbeddingClient):
    """Эмбеддер, обслуживаемый vLLM."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        *,
        api_key: str = "",
        timeout: float | None = None,
    ):
        """Создаёт клиент.

        Args:
            model: Имя модели эмбеддингов на сервере, например
                ``BAAI/bge-m3``.
            base_url: Адрес сервера без суффикса ``/v1``.
            api_key: Ключ доступа; пусто - подставляется заглушка.
            timeout: Предел ожидания ответа, с; ``None`` - умолчание SDK.
        """
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or NO_KEY,
            # Токенизатор OpenAI не имеет отношения к модели на vLLM: с
            # включённой резкой на сервер уходят чужие номера токенов вместо
            # текста, и векторы получаются мусорными.
            check_embedding_ctx_length=False,
            timeout=timeout,
        )
