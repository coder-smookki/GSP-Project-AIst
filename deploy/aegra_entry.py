import logging

from agent.graph import create_app_agent
from domain.context import ContextManager
from infrastructure.config.model_registry import get_model, setup_registry
from infrastructure.config.settings import get_settings
from infrastructure.observability.tracer import setup_tracing
from infrastructure.retrieval.search import Search

logger = logging.getLogger(__name__)


def graph():
    settings = get_settings()
    setup_tracing(settings)
    setup_registry(settings)
    model = get_model("default")
    support_model = get_model("support")
    context_manager = ContextManager(llm=support_model, budget=16000)

    configured_search: Search | None = None
    if settings.use_mock_search:
        from tests.fakes.fake_search import MockSearch

        logger.warning(
            "USE_MOCK_SEARCH=True — синтетические источники вместо реального "
            "слоя данных"
        )
        configured_search = MockSearch()

    return create_app_agent(
        model,
        support_model,
        context_manager,
        max_memories_count=15,
        search=configured_search,
    )
