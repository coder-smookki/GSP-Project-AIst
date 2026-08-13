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

from domain.memory import MemoryCollection
from infrastructure.retrieval.qdrant_search import SearchUnavailable
from infrastructure.retrieval.search import Search, to_citation

from .correction import rewrite_query as rewrite_query_model
from .events import emit
from .grading import grade_chunks
from .memory_gate import worth_extracting
from .numbers import annotate as annotate_numbers
from .prompts import MEMORY_EXTRACT_PROMPT, MODEL_SYSTEM_MESSAGE
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


def route_after_retrieve(state: AgentState) -> str:
    if state.get("needs_rewrite"):
        return "rewrite"
    return "assistant"


def query_from_state(state: AgentState) -> str:
    last = (state.get("messages") or [])[-1]
    if not hasattr(last, "content"):
        return ""
    return str(last["content"] if isinstance(last, dict) else last.content).strip()


def create_app_agent(
    model,
    support_model,
    context_manager=None,
    store=None,
    checkpointer=None,
    max_memories_count: int = 10,
    max_rewrites: int = 1,
    search: Search | None = None,
    grading: bool = False,
    grading_keep_partial: bool = False,
    verify_numbers: bool = True,
):

    graph = StateGraph(AgentState)

    async def reset_turn(state: AgentState):
        return {
            "query": "",
            "rewrite_attempts": 0,
            "needs_rewrite": False,
            "retrieval_failed": False,
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

        if verify_numbers and not state.get("retrieval_failed"):
            text = str(getattr(answer, "content", "") or "")
            marked = annotate_numbers(text, context)
            if marked != text:
                answer = AIMessage(content=marked)

        return {"messages": [answer]}

    async def retrieve(state: AgentState):
        if not search:
            return {}
        query = state.get("query") or query_from_state(state)
        if not query:
            return {"needs_rewrite": False}

        try:
            hits = await search.query(
                query,
                sections=state.get("sections"),
                areas=state.get("areas"),
            )
        except SearchUnavailable as exc:
            logger.error("поиск недоступен: %s", exc)
            emit("trace", {"stage": "retrieve", "status": "unavailable"})
            return {
                "query": query,
                "retrieved": [],
                "followups": [],
                "retrieval_failed": True,
                "needs_rewrite": False,
            }

        items = [
            to_citation(chunk, n, score).to_dict()
            for n, (chunk, score) in enumerate(hits, 1)
        ]

        grading_before_rewrite = grading and items and support_model is not None
        if grading_before_rewrite:
            emit("trace", {"stage": "retrieve", "status": "grading"})
            graded = await grade_chunks(
                support_model,
                query,
                items,
                keep_partial=grading_keep_partial,
            )
            if graded is not None:
                items = [dict(item, n=n) for n, item in enumerate(graded, 1)]

        if (
            not items
            and support_model is not None
            and state.get("rewrite_attempts", 0) < max_rewrites
        ):
            emit("trace", {"stage": "retrieve", "status": "rewriting"})
            return {"query": query, "needs_rewrite": True}

        emit("sources", {"items": items})

        followups = build_followups(items)
        if followups:
            emit("followups", {"items": followups})
        return {
            "query": query,
            "retrieved": items,
            "followups": followups,
            "needs_rewrite": False,
        }

    async def rewrite_query(state: AgentState):
        query = state.get("query") or ""
        rewritten = await rewrite_query_model(support_model, query)
        if rewritten:
            logger.info("коррекция запроса дала источник: %s", rewritten[:80])
        return {
            "query": rewritten or query,
            "rewrite_attempts": state.get("rewrite_attempts", 0) + 1,
            "needs_rewrite": False,
        }

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
    graph.add_node("assistant", assistant)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("condense_context", condense_context)
    graph.add_node("write_memory", write_memory)

    graph.add_edge(START, "reset_turn")
    graph.add_edge("reset_turn", "condense_context")
    graph.add_edge("condense_context", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"rewrite": "rewrite_query", "assistant": "assistant"},
    )
    graph.add_edge("rewrite_query", "retrieve")
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
