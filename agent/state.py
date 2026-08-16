import operator
from typing import Annotated, Literal, NotRequired

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class SearchPlan(BaseModel):
    """План поиска: режим и независимые подзапросы."""

    mode: Literal["chat", "kb"]
    subqueries: list[str] = Field(
        default_factory=list,
        description="независимые поисковые подзапросы для kb-режима",
    )


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: NotRequired[str]
    sections: NotRequired[list[str]]
    areas: NotRequired[list[str]]
    query: NotRequired[str]
    plan: NotRequired[SearchPlan]
    always_answer_with_db: NotRequired[bool]
    retrieved: NotRequired[list[dict]]
    branch_retrieved: NotRequired[Annotated[list[dict], operator.add]]
    failures: NotRequired[Annotated[list[str], operator.add]]
    followups: NotRequired[list[str]]
    retrieval_failed: NotRequired[bool]
    rewrite_attempts: NotRequired[int]
    needs_rewrite: NotRequired[bool]


class SearchAgentState(TypedDict):
    query: str
    rounds: NotRequired[int]
    sections: NotRequired[list[str]]
    areas: NotRequired[list[str]]
    retrieved: NotRequired[Annotated[list[dict], operator.add]]
    failures: NotRequired[Annotated[list[str], operator.add]]
    retrieval_failed: NotRequired[bool]
    branch_messages: Annotated[list, add_messages]
