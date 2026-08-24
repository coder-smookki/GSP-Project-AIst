"""Реестр клиентов моделей.

Собирает клиенты LLM и эмбеддингов по настройкам и раздаёт их по именам
слотов (``default``, ``support``). Слоты глобальные: в одном процессе живёт
одна сборка агента.

Неизвестный или нереализованный провайдер отвергается сразу, при сборке, с
перечислением поддерживаемых значений. Раньше ветка просто не выполнялась, и
приложение падало позже и невнятно - ``KeyError: 'default'`` при первом
обращении к модели.
"""

import logging

from infrastructure.config.settings import MODEL_PROVIDERS
from infrastructure.embeddings.client import EmbeddingClient
from infrastructure.llm.client import ModelClient

logger = logging.getLogger(__name__)

_MODELS: dict[str, ModelClient] = {}
_EMBEDDINGS: dict[str, EmbeddingClient] = {}


def _unknown_provider(kind: str, provider: str, setting: str) -> ValueError:
    """Строит ошибку о неподдерживаемом провайдере.

    Args:
        kind: Что именно собиралось: ``LLM`` или ``эмбеддингов``.
        provider: Значение из настроек.
        setting: Имя переменной окружения, которую надо поправить.

    Returns:
        ValueError: Готовая ошибка с перечислением поддерживаемых значений.
    """
    return ValueError(
        f"провайдер {kind} {provider!r} не поддерживается: задайте "
        f"{setting} одним из {', '.join(MODEL_PROVIDERS)}"
    )


def build_model_client(settings, model: str) -> ModelClient:
    """Создаёт клиент чат-модели по настройкам.

    Args:
        settings: Настройки приложения.
        model: Имя модели на стороне провайдера.

    Returns:
        ModelClient: Клиент чат-модели.

    Raises:
        ValueError: Провайдер из ``llm_provider`` не поддерживается.
    """
    provider = settings.llm_provider

    if provider == "ollama-local":
        from infrastructure.llm.ollama_client import OllamaClient

        return OllamaClient(
            model=model,
            base_url=settings.ollama_local_base_url,
            num_ctx=settings.llm_num_ctx,
            repeat_penalty=settings.llm_repeat_penalty,
            reasoning=settings.llm_reasoning,
        )

    if provider == "llamacpp":
        from infrastructure.llm.llamacpp_client import LlamaCppClient

        return LlamaCppClient(model=model, base_url=settings.llamacpp_base_url)

    if provider == "vllm":
        from infrastructure.llm.vllm_client import VLLMClient

        return VLLMClient(
            model=model,
            base_url=settings.vllm_base_url,
            timeout=settings.vllm_timeout,
            repeat_penalty=settings.llm_repeat_penalty,
            reasoning=settings.llm_reasoning,
            api_key=settings.vllm_api_key,
        )

    raise _unknown_provider("LLM", provider, "LLM_PROVIDER")


def build_embedding_client(
    settings, model: str | None = None, provider: str | None = None
) -> EmbeddingClient:
    """Создаёт клиент эмбеддингов по настройкам.

    Args:
        settings: Настройки приложения.
        model: Имя модели; ``None`` - взять ``embedding_model`` из настроек.
        provider: Провайдер; ``None`` - взять ``embedding_provider`` из
            настроек. Нужен индексации, где провайдер задаётся параметром
            прогона.

    Returns:
        EmbeddingClient: Клиент эмбеддингов.

    Raises:
        ValueError: Провайдер не поддерживается.
    """
    name = model or settings.embedding_model
    kind = provider or settings.embedding_provider

    if kind == "ollama-local":
        from infrastructure.embeddings.ollama_client import OllamaEmbeddingClient

        return OllamaEmbeddingClient(
            model=name,
            base_url=settings.ollama_local_base_url,
            timeout=settings.embedding_timeout,
        )

    if kind == "llamacpp":
        from infrastructure.embeddings.llamacpp_client import LlamaCppEmbeddingClient

        return LlamaCppEmbeddingClient(
            model=name,
            base_url=settings.llamacpp_base_url,
            timeout=settings.embedding_timeout,
        )

    if kind == "vllm":
        from infrastructure.embeddings.vllm_client import VLLMEmbeddingClient

        return VLLMEmbeddingClient(
            model=name,
            base_url=settings.resolved_vllm_embedding_base_url(),
            api_key=settings.vllm_api_key,
            timeout=settings.vllm_timeout,
        )

    raise _unknown_provider("эмбеддингов", kind, "EMBEDDING_PROVIDER")


def setup_registry(settings) -> None:
    """Наполняет реестр клиентами по настройкам.

    Args:
        settings: Настройки приложения.

    Raises:
        ValueError: Провайдер LLM или эмбеддингов не поддерживается.
    """
    _MODELS["default"] = build_model_client(settings, settings.llm_model)
    if settings.llm_support_model != "None":
        _MODELS["support"] = build_model_client(settings, settings.llm_support_model)
    else:
        _MODELS["support"] = _MODELS["default"]
    logger.info(
        "LLM: провайдер %s, основная модель %s, вспомогательная %s",
        settings.llm_provider,
        settings.llm_model,
        settings.llm_support_model,
    )

    _EMBEDDINGS["default"] = build_embedding_client(settings)
    logger.info(
        "эмбеддинги: провайдер %s, модель %s",
        settings.embedding_provider,
        settings.embedding_model,
    )


def get_model(name: str = "default") -> ModelClient:
    """Возвращает клиент чат-модели из реестра.

    Args:
        name: Имя слота: ``default`` или ``support``.

    Returns:
        ModelClient: Клиент чат-модели.

    Raises:
        KeyError: Реестр не собран (не вызывался ``setup_registry``) либо
            слота с таким именем нет.
    """
    return _MODELS[name]


def get_embedding(name: str = "default") -> EmbeddingClient:
    """Возвращает клиент эмбеддингов из реестра.

    Args:
        name: Имя слота.

    Returns:
        EmbeddingClient: Клиент эмбеддингов.

    Raises:
        KeyError: Реестр не собран либо слота с таким именем нет.
    """
    return _EMBEDDINGS[name]
