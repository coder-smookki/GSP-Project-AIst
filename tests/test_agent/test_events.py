from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph, add_messages

from agent.events import ALLOWED_KINDS, emit


class _TestState(TypedDict):
    messages: Annotated[list, add_messages]
    kind: str
    payload: dict


def _build_graph():
    graph = StateGraph(_TestState)

    def node(state: _TestState):
        emit(state["kind"], state["payload"])
        return {"messages": [{"role": "assistant", "content": "ответ"}]}

    graph.add_node("test_emit", node)
    graph.add_edge(START, "test_emit")
    graph.add_edge("test_emit", END)
    return graph.compile()


class TestEmitEnvelope:
    def test_stream_delivers_aist_envelope(self):
        app = _build_graph()
        chunks = list(
            app.stream(
                {"messages": [], "kind": "sources", "payload": {"items": [{"n": 1}]}},
                stream_mode=["custom"],
            )
        )
        assert chunks == [
            (
                "custom",
                {
                    "aist": {
                        "v": 1,
                        "kind": "sources",
                        "payload": {"items": [{"n": 1}]},
                    }
                },
            )
        ]

    def test_all_allowed_kinds_deliver(self):
        payload = {"x": 1}
        for kind in ALLOWED_KINDS:
            app = _build_graph()
            chunks = list(
                app.stream(
                    {"messages": [], "kind": kind, "payload": payload},
                    stream_mode=["custom"],
                )
            )
            assert chunks == [
                (
                    "custom",
                    {"aist": {"v": 1, "kind": kind, "payload": payload}},
                )
            ]


class TestEmitSafety:
    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown custom event kind"):
            emit("zzz", {})

    def test_invoke_without_writer_is_noop(self):
        app = _build_graph()
        result = app.invoke(
            {"messages": [], "kind": "sources", "payload": {"items": []}}
        )
        assert result["messages"][-1].content == "ответ"

    def test_direct_call_outside_stream_is_noop(self):
        emit("followups", {"items": []})
        emit("related", {})
        emit("trace", {})
