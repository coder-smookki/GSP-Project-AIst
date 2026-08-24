"""Каталог базы знаний: дерево областей и сборка документа из индекса."""

from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from infrastructure.retrieval.catalog import Catalog, CatalogUnavailable


class _Point:
    """Точка Qdrant с одним лишь payload."""

    def __init__(self, payload: dict) -> None:
        """Запоминает payload точки.

        Args:
            payload: Полезная нагрузка точки индекса.
        """
        self.payload = payload


class _FakeClient:
    """Клиент Qdrant, отдающий заранее заданные точки порциями."""

    def __init__(self, points: list[dict], *, batch: int = 2) -> None:
        """Готовит клиент.

        Args:
            points: Полезные нагрузки точек коллекции.
            batch: Сколько точек отдавать за один вызов прокрутки.
        """
        self._points = points
        self._batch = batch
        self.calls = 0
        self.with_vectors: list[bool] = []

    async def scroll(
        self,
        collection_name: str,
        scroll_filter=None,
        limit: int = 0,
        offset=None,
        with_payload=None,
        with_vectors: bool = False,
    ):
        """Имитирует постраничную прокрутку коллекции.

        Args:
            collection_name: Имя коллекции.
            scroll_filter: Фильтр выборки; поддерживается отбор по doc_code.
            limit: Игнорируется, размер порции задан в конструкторе.
            offset: Смещение - индекс следующей точки.
            with_payload: Запрошенные поля payload.
            with_vectors: Нужны ли векторы.

        Returns:
            tuple: Список точек и смещение следующей порции.
        """
        self.calls += 1
        self.with_vectors.append(with_vectors)

        selected = self._points
        if scroll_filter is not None:
            wanted = scroll_filter.must[0].match.value
            selected = [p for p in self._points if p.get("doc_code") == wanted]

        start = int(offset or 0)
        chunk = selected[start : start + self._batch]
        nxt = start + self._batch
        next_offset = nxt if nxt < len(selected) else None
        return [_Point(p) for p in chunk], next_offset


def _payload(doc_code: str, area, section_type: str, order_no: int = 0) -> dict:
    """Собирает payload одного фрагмента.

    Args:
        doc_code: Код документа.
        area: Область или список областей.
        section_type: Тип раздела.
        order_no: Порядковый номер фрагмента внутри документа.

    Returns:
        dict: Полезная нагрузка точки.
    """
    return {
        "chunk_id": f"{doc_code}#c{order_no:03d}",
        "doc_code": doc_code,
        "doc_title": f"Документ {doc_code}",
        "area": area,
        "section_type": section_type,
        "order_no": order_no,
        "section_no": "4",
        "section_title": "Порядок проведения",
        "clause": "4.1.4",
        "breadcrumb": "Раздел 4",
        "text": f"текст {order_no}",
        "is_table_region": False,
    }


async def test_tree_groups_documents_by_area() -> None:
    """Дерево собирается по трём уровням: область, документ, тип раздела."""
    client = _FakeClient(
        [
            _payload("Р-У2-59", ["ohrana_truda"], "procedura"),
            _payload("Р-У2-59", ["ohrana_truda"], "terminy"),
            _payload("Р-У2-23", ["pozharnaya_bezopasnost"], "procedura"),
        ]
    )
    tree = await Catalog(client, collection="c").tree()

    areas = {item["area"]: item for item in tree}
    assert set(areas) == {"ohrana_truda", "pozharnaya_bezopasnost"}
    assert areas["ohrana_truda"]["title"] == "Охрана труда"

    documents = areas["ohrana_truda"]["documents"]
    assert [doc["doc_code"] for doc in documents] == ["Р-У2-59"]
    assert documents[0]["chunks"] == 2
    assert {item["key"] for item in documents[0]["section_types"]} == {
        "procedura",
        "terminy",
    }


async def test_document_belongs_to_every_area_it_is_tagged_with() -> None:
    """Разметка областей у чанка - список, поэтому документ попадает в обе.

    Иначе дерево «потеряло» бы документ в одной из областей, и выбор этой
    области молча исключал бы его из поиска.
    """
    client = _FakeClient(
        [_payload("Р-У2-80", ["ohrana_truda", "go_i_chs"], "procedura")]
    )
    tree = await Catalog(client, collection="c").tree()

    areas = {item["area"] for item in tree}
    assert areas == {"ohrana_truda", "go_i_chs"}


async def test_tree_is_cached_between_calls() -> None:
    """Состав базы меняется только при переиндексации - второй проход лишний."""
    client = _FakeClient([_payload("Р-У2-59", ["ohrana_truda"], "procedura")])
    catalog = Catalog(client, collection="c")

    await catalog.tree()
    calls_after_first = client.calls
    await catalog.tree()

    assert client.calls == calls_after_first


async def test_tree_never_requests_vectors() -> None:
    """Дереву нужны только метаданные: векторы утяжелили бы ответ вдесятеро."""
    client = _FakeClient([_payload("Р-У2-59", ["ohrana_truda"], "procedura")])
    await Catalog(client, collection="c").tree()

    assert client.with_vectors and not any(client.with_vectors)


async def test_document_is_restored_in_reading_order() -> None:
    """Фрагменты выстраиваются по order_no, а не по порядку хранения."""
    client = _FakeClient(
        [
            _payload("Р-У2-59", ["ohrana_truda"], "procedura", order_no=2),
            _payload("Р-У2-59", ["ohrana_truda"], "procedura", order_no=0),
            _payload("Р-У2-59", ["ohrana_truda"], "procedura", order_no=1),
            _payload("Р-У2-23", ["ohrana_truda"], "procedura", order_no=0),
        ]
    )
    document = await Catalog(client, collection="c").document("Р-У2-59")

    assert document["doc_code"] == "Р-У2-59"
    assert document["doc_title"] == "Документ Р-У2-59"
    assert [chunk["order_no"] for chunk in document["chunks"]] == [0, 1, 2]


async def test_unknown_document_returns_no_chunks() -> None:
    """Неизвестный код - пустой список фрагментов, а не исключение."""
    client = _FakeClient([_payload("Р-У2-59", ["ohrana_truda"], "procedura")])
    document = await Catalog(client, collection="c").document("Р-У2-99")

    assert document["chunks"] == []


async def test_blank_code_does_not_reach_the_index() -> None:
    """Пустой код отсекается до запроса: фильтр по нему вернул бы всю базу."""
    client = _FakeClient([_payload("Р-У2-59", ["ohrana_truda"], "procedura")])
    document = await Catalog(client, collection="c").document("   ")

    assert document["chunks"] == []
    assert client.calls == 0


async def test_index_failure_is_not_mistaken_for_empty_base() -> None:
    """Недоступный индекс поднимает ошибку, а не выдаёт пустое дерево.

    Пустое дерево означало бы «в базе ничего нет», и интерфейс показал бы
    пользователю пустой список областей вместо отказа.
    """

    class _Broken(_FakeClient):
        async def scroll(self, *args, **kwargs):
            """Имитирует недоступную коллекцию.

            Raises:
                UnexpectedResponse: Всегда.
            """
            raise UnexpectedResponse(404, "Not found", b"", None)

    with pytest.raises(CatalogUnavailable):
        await Catalog(_Broken([]), collection="c").tree()
