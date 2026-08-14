from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from agent.graph import create_app_agent
from domain.context import ContextManager
from infrastructure.config.model_registry import get_model, setup_registry
from infrastructure.config.settings import Settings, get_settings
from infrastructure.observability.tracer import setup_tracing
from infrastructure.retrieval.factory import get_search


def create_application(settings: Settings | None = None):
    """Собирает граф агента целиком.

    Settings передаётся явно, чтобы один процесс мог собрать несколько
    вариантов системы (разные модели, настройки поиска) для A/B-бенчмарка.
    Без аргумента используются глобальные настройки окружения.

    Args:
        settings: Настройки варианта; ``None`` — глобальные настройки.

    Returns:
        Скомпилированный граф агента.
    """
    if settings is None:
        settings = get_settings()
    setup_tracing(settings)
    setup_registry(settings)
    model = get_model("default")
    support_model = get_model("support")
    context_manager = ContextManager(llm=support_model, budget=16000)
    return create_app_agent(
        model,
        support_model,
        context_manager,
        store=InMemoryStore(),
        checkpointer=MemorySaver(),
        max_memories_count=15,
        max_rewrites=settings.retrieval_max_rewrites,
        # Без этого CLI-сборка агента шла вообще без поиска: узел retrieve
        # молча возвращал пустоту, и агент отвечал по памяти модели.
        search=get_search(settings),
        grading=settings.grading_enabled,
        grading_keep_partial=settings.grading_keep_partial,
        verify_numbers=settings.verify_numbers,
    )
