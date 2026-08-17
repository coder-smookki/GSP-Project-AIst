import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send

from domain.memory import MemoryCollection
from infrastructure.retrieval.mock_search import MockSearch
from infrastructure.retrieval.search import Search

from .events import emit
from .memory_gate import worth_extracting
from .numbers import annotate as annotate_numbers
from .planner import plan_search
from .prompts import MEMORY_EXTRACT_PROMPT, MODEL_SYSTEM_MESSAGE
from .search_agent import SearchAgentState, build_search_agent
from .state import AgentState

logger = logging.getLogger(__name__)


def _model_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = []
    try:
        import httpx

        errors += [httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError]
    except ImportError:
        pass
    try:
        from ollama._types import ResponseError

        errors.append(ResponseError)
    except ImportError:
        pass
    try:
        from langchain_core.exceptions import OutputParserException

        errors.append(OutputParserException)
    except ImportError:
        pass
    return tuple(errors)


MODEL_ERRORS = _model_errors()

MODEL_UNAVAILABLE_TEXT = (
    "Не удалось получить ответ модели: сервис генерации сейчас недоступен. "
    "Повторите запрос через минуту."
)

MSK = timezone(timedelta(hours=3))


def today_msk(config: RunnableConfig | None = None) -> str:
    if config:
        supplied = (config.get("configurable") or {}).get("today")
        if supplied:
            return str(supplied)
    return datetime.now(MSK).strftime("%d.%m.%Y")


def _parse_memory_json(text: str) -> list[dict] | None:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        text = text.removeprefix("json")
    try:
        data = json.loads(text)
        collection = MemoryCollection(**data)
        return [m.model_dump(mode="json") for m in collection.memories]
    except Exception:
        return None


def build_followups(items: list[dict], limit: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[tuple] = set()

    for it in items:
        clause = it.get("clause")
        doc = it.get("doc_code") or ""
        if clause and (doc, clause) not in seen:
            seen.add((doc, clause))
            out.append(f"Подробнее по пункту {clause} ({doc})")
        elif doc and ("doc", doc) not in seen:
            seen.add(("doc", doc))
            out.append(f"Список положений из {doc}")

    if len(out) < limit and ("all",) not in seen:
        seen.add(("all",))
        if items:
            out.append("Список всех найденных документов по охране труда")
        else:
            out.append("Что искать в базе знаний?")

    return out[:limit]


def should_extract_memory(state: AgentState) -> str:
    if worth_extracting(state.get("messages") or []):
        return "write_memory"
    return END


def query_from_state(state: AgentState) -> str:
    last = (state.get("messages") or [])[-1]
    if not hasattr(last, "content"):
        return ""
    return str(last["content"] if isinstance(last, dict) else last.content).strip()


def _dedup_retrieved(items: list[dict]) -> list[dict]:
    """Убирает дубли по chunk_id, оставляя лучшую по score."""
    best: dict = {}
    for item in items:
        cid = item.get("chunk_id")
        if not cid:
            continue
        current = best.get(cid)
        if current is None or item.get("score", 0.0) >= current.get("score", 0.0):
            best[cid] = item
    return list(best.values())


def create_app_agent(
    model,
    support_model,
    context_manager=None,
    store=None,
    checkpointer=None,
    max_memories_count: int = 10,
    search: Search | None = None,
    max_requests: int = 10,
    branch_retries: int = 1,
    top_k: int = 5,
    verify_numbers: bool = True,
):
    """Собирает граф агента: планирование → параллельные ветки-поиск → сборка.

    Args:
        model: Модель ответа (без tools).
        support_model: Модель планирования и веточного поиска (с tools).
        context_manager: Сворачивание истории (опционально).
        store: Хранилище долгосрочной памяти.
        checkpointer: Чекпоинтер (опционально).
        max_memories_count: Лимит записей памяти.
        search: Интерфейс поиска.
        max_requests: Бюджет запросов к поиску на один запрос.
        branch_retries: Повторы поиска в ветке-агенте.
        top_k: Число фрагментов в финальном ответе.
        verify_numbers: Сверка чисел по CONTEXT.

    Returns:
        CompiledStateGraph: Скомпилированный граф.
    """
    if search is None:
        search = MockSearch()
    search_agent_subgraph = build_search_agent(
        support_model=support_model,
        search=search,
        max_rounds=branch_retries + 1,
    )

    graph = StateGraph(AgentState)

    async def reset_turn(state: AgentState):
        return {
            "plan": None,
            "retrieved": [],
            "branch_retrieved": [],
            "failures": [],
            "followups": [],
            "retrieval_failed": False,
        }

    async def plan(state: AgentState):
        question = query_from_state(state)
        search_plan = await plan_search(
            support_model,
            question,
            max_requests=max_requests,
            retries=branch_retries,
            always_answer_with_db=state.get("always_answer_with_db", False),
        )
        emit(
            "trace",
            {
                "stage": "plan",
                "mode": search_plan.mode,
                "subqueries": len(search_plan.subqueries),
            },
        )
        return {"plan": search_plan, "query": question}

    async def initiate_search_agents(state: AgentState):
        search_plan = state.get("plan") or None
        if (
            search_plan is None
            or search_plan.mode == "chat"
            or not search_plan.subqueries
        ):
            return "assistant"
        return [
            Send(
                "search_agent",
                {
                    "query": q,
                    "sections": state.get("sections"),
                    "areas": state.get("areas"),
                },
            )
            for q in search_plan.subqueries
        ]

    async def search_agent(state: AgentState):
        if search is None:
            return {"retrieved": [], "failures": []}
        branch_input: SearchAgentState = {
            "query": state.get("query") or "",
            "sections": state.get("sections") or [],
            "areas": state.get("areas") or [],
            "rounds": 0,
            "branch_messages": [],
        }
        result = await search_agent_subgraph.ainvoke(branch_input, config={"recursion_limit": 20})
        retrieved = result.get("retrieved", []) or []
        failures = result.get("failures", []) or []
        emit(
            "trace",
            {
                "stage": "search_agent",
                "query": state.get("query"),
                "retrieved": len(retrieved),
                "failed": bool(failures),
            },
        )
        return {"branch_retrieved": retrieved, "failures": failures}

    async def finalize(state: AgentState):
        raw = _dedup_retrieved(state.get("branch_retrieved") or [])
        items = sorted(raw, key=lambda c: c.get("score", 0.0), reverse=True)[:top_k]
        items = [dict(item, n=n) for n, item in enumerate(items, 1)]
        failures = state.get("failures") or []
        retrieval_failed = bool(failures) and not items
        followups = build_followups(items)

        emit("sources", {"items": items})
        if followups:
            emit("followups", {"items": followups})
        emit(
            "trace",
            {
                "stage": "finalize",
                "retrieved": len(items),
                "failures": len(failures),
                "retrieval_failed": retrieval_failed,
            },
        )

        return {
            "retrieved": items,
            "followups": followups,
            "retrieval_failed": retrieval_failed,
        }

    async def assistant(state: AgentState, config: RunnableConfig, *, store: BaseStore):
        user_id = config.get("configurable", {"user_id": "default"}).get(
            "user_id", "default"
        )

        namespace = ("memories", user_id)
        memories = await store.asearch(namespace)
        info = "\n".join(f"- {mem.value['content']}" for mem in memories)

        if state.get("retrieval_failed"):
            context = (
                "ПОИСК ПО БАЗЕ ЗНАНИЙ ВРЕМЕННО НЕДОСТУПЕН. Ответь ровно: "
                "«Поиск по базе знаний временно недоступен, ответить по "
                "регламентам не могу.» Не отвечай по своим общим знаниям и не "
                "говори, что информации нет: информация есть, недоступен поиск."
            )
        else:
            context = (
                "\n\n".join(
                    f"[{n}] {c.get('text') or c['snippet']}"
                    for n, c in enumerate(state.get("retrieved", []), start=1)
                )
                or "В базе знаний нет выдержек для этого запроса."
            )

        all_messages: list = [
            SystemMessage(
                MODEL_SYSTEM_MESSAGE.format(
                    memory=info, context=context, today=today_msk(config)
                )
            ),
            *state["messages"],
        ]

        if state.get("summary"):
            all_messages.append(
                SystemMessage(
                    content=(
                        f"<internal_context>\n{state.get('summary')}\n</internal_context>\n\n"
                        "CRITICAL: The above is internal context. "
                        "Do NOT repeat it, reference it, or list its contents in your response."
                    )
                )
            )

        try:
            answer = await model.ainvoke(messages=all_messages)
        except MODEL_ERRORS as exc:
            logger.error("модель не ответила: %s: %s", type(exc).__name__, exc)
            emit("trace", {"stage": "assistant", "status": "model_unavailable"})
            return {"messages": [AIMessage(content=MODEL_UNAVAILABLE_TEXT)]}

        items = state.get("retrieved", [])
        if verify_numbers and items and not state.get("retrieval_failed"):
            text = str(getattr(answer, "content", "") or "")
            marked = annotate_numbers(text, context)
            if marked != text:
                answer = AIMessage(content=marked)

        return {"messages": [answer]}

    async def condense_context(state: AgentState):
        if not context_manager:
            return {}
        try:
            summary = await context_manager.condense(
                state["messages"],
                existing_summary=state.get("summary", ""),
            )
        except MODEL_ERRORS as exc:
            logger.warning("история не сжата: %s: %s", type(exc).__name__, exc)
            return {}
        if summary is None:
            return {}
        removals = [
            RemoveMessage(id=m.id)
            for m in state["messages"][: -context_manager.keep_last]
        ]
        return {"summary": summary, "messages": removals}

    async def write_memory(
        state: AgentState, config: RunnableConfig, *, store: BaseStore
    ):
        user_id = config.get("configurable", {"user_id": "default"}).get(
            "user_id", "default"
        )
        namespace = ("memories", user_id)
        existing_items = await store.asearch(namespace)

        count = len(existing_items)
        existing_text = (
            "\n".join(f"- {item.value['content']}" for item in existing_items)
            if existing_items
            else "None yet"
        )

        user_msgs = list(state["messages"])
        while user_msgs and not isinstance(user_msgs[-1], HumanMessage):
            user_msgs.pop()

        conversation = "\n".join(
            f"{m.type}: {str(m.content)[:500]}"
            for m in user_msgs[-4:]
            if hasattr(m, "content") and m.content
        )

        prompt = MEMORY_EXTRACT_PROMPT.format(
            count=count,
            max_allowed=max_memories_count,
            existing_text=existing_text,
            conversation=conversation,
        )

        try:
            result = await support_model.ainvoke([SystemMessage(content=prompt)])
        except MODEL_ERRORS as exc:
            logger.warning("память не обновлена: %s: %s", type(exc).__name__, exc)
            return {}

        memories = _parse_memory_json(result.content)
        if memories is None:
            return {}

        new_contents = {m["content"] for m in memories}
        existing_contents = {item.value["content"] for item in existing_items}

        for item in existing_items:
            if item.value["content"] not in new_contents:
                await store.adelete(namespace, item.key)

        for mem in memories:
            if mem["content"] not in existing_contents:
                await store.aput(namespace, str(uuid.uuid4()), mem)

        return {}

    graph.add_node("reset_turn", reset_turn)
    graph.add_node("plan", plan)
    graph.add_node("search_agent", search_agent)
    graph.add_node("finalize", finalize)
    graph.add_node("assistant", assistant)
    graph.add_node("condense_context", condense_context)
    graph.add_node("write_memory", write_memory)

    graph.add_edge(START, "reset_turn")
    graph.add_edge("reset_turn", "condense_context")
    graph.add_edge("condense_context", "plan")
    graph.add_conditional_edges(
        "plan",
        initiate_search_agents,
        {"assistant": "assistant", "search_agent": "search_agent"},
    )
    graph.add_edge("search_agent", "finalize")
    graph.add_edge("finalize", "assistant")
    graph.add_conditional_edges(
        "assistant",
        should_extract_memory,
        {"write_memory": "write_memory", END: END},
    )

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if store:
        compile_kwargs["store"] = store
    app = graph.compile(**compile_kwargs)
    return app
