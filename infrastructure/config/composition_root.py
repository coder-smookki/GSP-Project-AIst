from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from agent.graph import create_app_agent
from domain.context import ContextManager
from infrastructure.config.model_registry import get_model, setup_registry
from infrastructure.config.settings import get_settings
from infrastructure.retrieval.factory import get_search
from infrastructure.observability.tracer import setup_tracing


def create_application():
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
        # Без этого CLI-сборка агента шла вообще без поиска: узел retrieve
        # молча возвращал пустоту, и агент отвечал по памяти модели.
        search=get_search(settings),
    )
