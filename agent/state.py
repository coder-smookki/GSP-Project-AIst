from typing import Annotated, NotRequired

from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: NotRequired[str]
    sections: NotRequired[list[str]]
    retrieved: NotRequired[list[dict]]
    followups: NotRequired[list[str]]
    areas: NotRequired[list[str]]
    retrieval_failed: NotRequired[bool]
    query: NotRequired[str]
    rewrite_attempts: NotRequired[int]
    needs_rewrite: NotRequired[bool]
