import logging

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

ALLOWED_KINDS = frozenset({"trace", "sources", "related", "followups"})


def emit(kind: str, payload: dict) -> None:
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unknown custom event kind: {kind!r}")
    try:
        get_stream_writer()({"aist": {"v": 1, "kind": kind, "payload": payload}})
    except Exception as exc:  # noqa: BLE001
        logger.debug("custom emit skipped: %s", exc)
