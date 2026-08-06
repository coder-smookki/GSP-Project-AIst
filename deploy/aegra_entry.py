from agent.graph import create_app_agent
from domain.context import ContextManager
from infrastructure.config.model_registry import get_model, setup_registry
from infrastructure.config.settings import get_settings
from infrastructure.observability.tracer import setup_tracing
from tests.fakes.fake_search import MockSearch


def graph():
    settings = get_settings()
    setup_tracing(settings)
    setup_registry(settings)
    model = get_model("default")
    support_model = get_model("support")
    context_manager = ContextManager(llm=support_model, budget=16000)

    # ВРЕМЕННО: до подключения реального слоя данных (Qdrant) держим источники с синтетического MockSearch, чтобы панель работала в UI.
    return create_app_agent(
        model,
        support_model,
        context_manager,
        max_memories_count=15,
        search=MockSearch(),
    )
