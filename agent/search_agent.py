"""Веточный поисковый агент: ReAct-петля с одним инструментом.

Сабграф на подзапрос планировщика. Модель сама решает, когда поиск
достаточен; инструмент ``retrieve_kb`` можно вызвать не более
``max_rounds`` раз (первый поиск плюс повторы). Результаты складываются
в каналы ``retrieved``/``failures`` и возвращаются в родительский граф,
который агрегирует их по всем веткам.
"""

import logging
import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from infrastructure.retrieval.qdrant_search import SearchUnavailable
from infrastructure.retrieval.search import Search, to_citation

from .prompts import SEARCH_AGENT_PROMPT

logger = logging.getLogger(__name__)


class SearchAgentState(TypedDict):
    query: str
    rounds: NotRequired[int]
    sections: NotRequired[list[str]]
    areas: NotRequired[list[str]]
    retrieved: NotRequired[Annotated[list[dict], operator.add]]
    failures: NotRequired[Annotated[list[str], operator.add]]
    retrieval_failed: NotRequired[bool]
    branch_messages: Annotated[list, add_messages]


class RetrieveQuery(BaseModel):
    query: str = Field(description="поисковый запрос к базе знаний")


def build_search_agent(support_model, search: Search, *, max_rounds: int = 2):
    """Собирает скомпилированный сабграф веточного поиска.

    Args:
        support_model: Модель веточного агента (tool-calling).
        search: Поиск по базе знаний.
        max_rounds: Максимум обращений к инструменту на ветку.

    Returns:
        Скомпилированный граф с состоянием ``SearchAgentState``.
    """
    graph = StateGraph(SearchAgentState)

    async def agent_assistant(state: SearchAgentState):
        messages = [
            SystemMessage(
                SEARCH_AGENT_PROMPT.format(
                    query=state["query"], max_rounds=max_rounds
                )
            ),
            *state.get("branch_messages", []),
        ]
        response = await support_model.ainvoke(messages=messages, tools=[RetrieveQuery])
        return {"branch_messages": [response]}

    async def retrieve_tool(state: SearchAgentState):
        last = state["branch_messages"][-1]
        call = last.tool_calls[0]
        query = str(call["args"].get("query", "")).strip() or state["query"]
        rounds = state.get("rounds", 0) + 1

        try:
            hits = await search.query(
                query,
                sections=state.get("sections"),
                areas=state.get("areas"),
            )
        except SearchUnavailable as exc:
            logger.error("веточный поиск недоступен: %s", exc)
            return {
                "rounds": rounds,
                "failures": [query],
                "retrieval_failed": True,
                "branch_messages": [
                    ToolMessage(
                        content="Поиск по базе знаний временно недоступен.",
                        tool_call_id=call["id"],
                    )
                ],
            }

        items = [
            to_citation(chunk, n, score).to_dict()
            for n, (chunk, score) in enumerate(hits, 1)
        ]
        content = "\n\n".join(f"[{n}] {c['snippet']}" for n, c in enumerate(items, 1))
        if not items:
            content = (
                "По запросу ничего не найдено. Переформулируй запрос "
                "официальными терминами регламентов и повтори поиск."
            )

        return {
            "rounds": rounds,
            "retrieved": items,
            "branch_messages": [
                ToolMessage(content=content, tool_call_id=call["id"])
            ],
        }

    def route_after_assistant(state: SearchAgentState) -> str:
        last = state["branch_messages"][-1]
        if getattr(last, "tool_calls", None):
            return "retrieve_tool"
        return END

    def route_after_tool(state: SearchAgentState) -> str:
        if state.get("rounds", 0) >= max_rounds:
            return END
        return "agent_assistant"

    graph.add_node("agent_assistant", agent_assistant)
    graph.add_node("retrieve_tool", retrieve_tool)
    graph.add_edge(START, "agent_assistant")
    graph.add_conditional_edges(
        "agent_assistant",
        route_after_assistant,
        {"retrieve_tool": "retrieve_tool", END: END},
    )
    graph.add_conditional_edges(
        "retrieve_tool",
        route_after_tool,
        {"agent_assistant": "agent_assistant", END: END},
    )
    return graph.compile()