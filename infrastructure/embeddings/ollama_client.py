"""Клиент эмбеддингов Ollama через OpenAI-совместимый слой."""

from .openai_compat import OpenAICompatEmbeddingClient


class OllamaEmbeddingClient(OpenAICompatEmbeddingClient):
    """Эмбеддер, обслуживаемый Ollama."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float | None = None,
    ):
        """Создаёт клиент.

        Args:
            model: Имя модели эмбеддингов в Ollama, например ``bge-m3``.
            base_url: Адрес сервера Ollama.
            timeout: Предел ожидания ответа, с. Без него подвисший сервер
                модели держит запрос бесконечно: ни ответа, ни ошибки.
        """
        super().__init__(
            model=model,
            base_url=base_url,
            # Резка по контексту идёт токенизатором OpenAI и к моделям
            # Ollama отношения не имеет.
            check_embedding_ctx_length=False,
            timeout=timeout,
        )
