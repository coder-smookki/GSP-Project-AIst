"""Тесты восстановления состояния индекса для инкрементальной индексации.

Состояние читается из самой коллекции Qdrant: у каждой точки в payload
лежит отпечаток документа. Ошибка здесь не видна снаружи и приводит к
худшему из возможных исходов — документ молча выпадает из поиска, а
система продолжает уверенно отвечать по неполному регламенту.
"""

from __future__ import annotations

import pytest

from infrastructure.ingest.incremental import StateManager


class _Point:
    """Точка Qdrant с payload."""

    def __init__(self, doc_id: str, content_hash: str):
        self.payload = {"doc_id": doc_id, "content_hash": content_hash}


class _Client:
    """Клиент Qdrant, отдающий заданные точки одной страницей."""

    def __init__(self, points: list[_Point], exists: bool = True):
        self._points = points
        self._exists = exists

    async def collection_exists(self, collection: str) -> bool:
        """Сообщает, существует ли коллекция.

        Args:
            collection: Имя коллекции.

        Returns:
            bool: Заданное при создании значение.
        """
        return self._exists

    async def scroll(self, **kwargs):
        """Отдаёт все точки одной страницей.

        Args:
            **kwargs: Параметры обхода, не используются.

        Returns:
            tuple: Точки и ``None`` как признак конца обхода.
        """
        return self._points, None


async def test_consistent_document_lands_in_state() -> None:
    """Документ с одним отпечатком попадает в состояние и будет пропущен."""
    client = _Client([_Point("88499", "aaa"), _Point("88499", "aaa")])
    state = await StateManager().load(client, "c")
    assert state == {"88499": "aaa"}


async def test_mixed_hashes_force_reindex() -> None:
    """Документ с разными отпечатками исключается из состояния.

    Несколько отпечатков означают оборванную прошлую загрузку: часть точек
    осталась от предыдущей версии. Раньше брался первый попавшийся, и если
    им оказывался новый, документ считался неизменившимся и пропускался
    навсегда. Теперь неоднозначность трактуется как «изменился».
    """
    client = _Client(
        [
            _Point("88499", "новый"),
            _Point("88499", "старый"),
            _Point("88551", "целый"),
        ]
    )
    state = await StateManager().load(client, "c")
    assert "88499" not in state
    assert state["88551"] == "целый"


async def test_order_does_not_matter() -> None:
    """Результат не зависит от порядка обхода.

    Порядок ``scroll`` ничем не задан, поэтому прежнее поведение было ещё и
    невоспроизводимым: пропустится документ или нет, решал случай.
    """
    forward = await StateManager().load(
        _Client([_Point("d", "a"), _Point("d", "b")]), "c"
    )
    backward = await StateManager().load(
        _Client([_Point("d", "b"), _Point("d", "a")]), "c"
    )
    assert forward == backward == {}


async def test_missing_collection_gives_empty_state() -> None:
    """Отсутствие коллекции означает полную индексацию."""
    assert await StateManager().load(_Client([], exists=False), "c") == {}


@pytest.mark.parametrize(
    "payload",
    [{"doc_id": "d"}, {"content_hash": "h"}, {"doc_id": 1, "content_hash": "h"}],
)
async def test_broken_payload_is_ignored(payload: dict) -> None:
    """Точка без пары «идентификатор — отпечаток» состояние не портит."""

    class _Bad:
        def __init__(self):
            self.payload = payload

    assert await StateManager().load(_Client([_Bad()]), "c") == {}
