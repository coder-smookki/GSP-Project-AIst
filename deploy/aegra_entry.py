"""Точка входа Aegra: сборка графа агента.

Фабрика вызывается один раз при загрузке модуля, поэтому тяжёлые ресурсы
(клиент Qdrant, реестр моделей) создаются здесь и живут весь процесс.
"""

import logging

from agent.graph import create_app_agent
from domain.context import ContextManager
from infrastructure.config.model_registry import get_model, setup_registry
from infrastructure.config.settings import get_settings
from infrastructure.observability.tracer import setup_tracing
from infrastructure.retrieval.factory import get_search

logger = logging.getLogger(__name__)


def graph():
    """Собирает граф агента со всеми зависимостями.

    Returns:
        Скомпилированный граф LangGraph.
    """
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
        max_memories_count=15,
        search=get_search(settings),
    )
