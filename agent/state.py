import operator
from typing import Annotated, Literal, NotRequired

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

#: Сигнал «очистить канал» для накопительных полей состояния.
#:
#: Каналы ``branch_retrieved`` и ``failures`` накапливают вклад параллельных
#: веток, поэтому у них reducer-сумматор. Обнулить такой канал возвратом
#: пустого списка нельзя: reducer применит его как ``current + []``, то есть
#: не сделает ничего. Пустой список и не годится на роль сигнала - ветка,
#: честно ничего не нашедшая, возвращает ровно его и стёрла бы находки
#: соседних веток. Отсюда отдельное значение.
RESET_CHANNEL = "__reset__"


def merge_or_reset(current: list | None, update):
    """Складывает вклады веток, но понимает сигнал сброса.

    Args:
        current: Накопленное значение канала.
        update: Вклад узла или ``RESET_CHANNEL`` для очистки.

    Returns:
        list: Пустой список при сигнале сброса, иначе сумма вкладов.
    """
    if update == RESET_CHANNEL:
        return []
    return (current or []) + list(update or [])


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
    docs: NotRequired[list[str]]
    query: NotRequired[str]
    plan: NotRequired[SearchPlan]
    always_answer_with_db: NotRequired[bool]
    retrieved: NotRequired[list[dict]]
    branch_retrieved: NotRequired[Annotated[list[dict], merge_or_reset]]
    failures: NotRequired[Annotated[list[str], merge_or_reset]]
    followups: NotRequired[list[str]]
    retrieval_failed: NotRequired[bool]
    rewrite_attempts: NotRequired[int]
    needs_rewrite: NotRequired[bool]


class SearchAgentState(TypedDict):
    query: str
    rounds: NotRequired[int]
    sections: NotRequired[list[str]]
    areas: NotRequired[list[str]]
    docs: NotRequired[list[str]]
    retrieved: NotRequired[Annotated[list[dict], operator.add]]
    failures: NotRequired[Annotated[list[str], operator.add]]
    retrieval_failed: NotRequired[bool]
    branch_messages: Annotated[list, add_messages]
