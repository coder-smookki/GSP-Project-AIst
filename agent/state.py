from typing import Annotated, NotRequired

from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: NotRequired[str]
