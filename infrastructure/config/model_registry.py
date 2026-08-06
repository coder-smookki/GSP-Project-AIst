from infrastructure.embeddings.client import EmbeddingClient
from infrastructure.embeddings.ollama_client import OllamaEmbeddingClient
from infrastructure.llm.client import ModelClient
from infrastructure.llm.ollama_client import OllamaClient

_MODELS: dict[str, ModelClient] = {}
_EMBEDDINGS: dict[str, EmbeddingClient] = {}


def setup_registry(settings) -> None:
    """Create clients and put them into registry"""

    if settings.llm_provider == "ollama-local":
        _MODELS["default"] = OllamaClient(
            model=settings.llm_model,
            base_url=settings.ollama_local_base_url,
            num_ctx=settings.llm_num_ctx,
            repeat_penalty=settings.llm_repeat_penalty,
        )
        if settings.llm_support_model != "None":
            _MODELS["support"] = OllamaClient(
                model=settings.llm_support_model,
                base_url=settings.ollama_local_base_url,
                num_ctx=settings.llm_num_ctx,
                repeat_penalty=settings.llm_repeat_penalty,
            )
        else:
            _MODELS["support"] = _MODELS["default"]

    if settings.embedding_provider == "ollama-local":
        _EMBEDDINGS["default"] = OllamaEmbeddingClient(
            model=settings.embedding_model,
            base_url=settings.ollama_local_base_url,
        )


def get_model(name: str = "default") -> ModelClient:
    return _MODELS[name]


def get_embedding(name: str = "default") -> EmbeddingClient:
    return _EMBEDDINGS[name]
