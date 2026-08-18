"""Клиент эмбеддингов llama.cpp через OpenAI-совместимый слой."""

from .openai_compat import OpenAICompatEmbeddingClient


class LlamaCppEmbeddingClient(OpenAICompatEmbeddingClient):
    """Эмбеддер, обслуживаемый сервером llama.cpp."""

    def __init__(self, model: str, base_url: str = "http://localhost:8080"):
        """Создаёт клиент.

        Args:
            model: Имя модели эмбеддингов на сервере.
            base_url: Адрес сервера llama.cpp.
        """
        super().__init__(model=model, base_url=base_url)
