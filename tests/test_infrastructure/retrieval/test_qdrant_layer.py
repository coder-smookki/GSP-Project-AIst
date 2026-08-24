"""Тесты слоя Qdrant без живого сервера.

Проверяется то, что можно проверить, не поднимая контейнер: форма объектов,
уходящих на сервер, и поведение при ошибках. Именно здесь ловятся дефекты,
которые сервер не считает ошибкой и потому не сообщает о них.

Локальный режим ``:memory:`` для гибрида непригоден: встроенный BM25
считает сервер, а клиент в локальном режиме требует fastembed. Поэтому
работу самого поиска покрывает отдельный набор с маркером `qdrant`.
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from infrastructure.ingest.areas import UNCLASSIFIED, classify_areas
from infrastructure.ingest.models import IngestChunk, SectionType
from infrastructure.retrieval.bm25 import (
    BM25_MODEL,
    BM25_OPTIONS,
    _parse_version,
    bm25_document,
)
from infrastructure.retrieval.qdrant_search import QdrantSearch, SearchUnavailable
from infrastructure.retrieval.rerank import RerankerUnavailable
from infrastructure.retrieval.store import PAYLOAD_INDEXES, chunk_payload


def make_chunk(**overrides) -> IngestChunk:
    """Собирает чанк с разумными значениями по умолчанию.

    Args:
        **overrides: Поля, которые нужно переопределить.

    Returns:
        IngestChunk: Чанк для теста.
    """
    defaults = {
        "chunk_id": "ДОК-1#c0001",
        "doc_id": "88499",
        "doc_code": "ДОК-1",
        "doc_title": "Правила пожарной безопасности",
        "section_no": "3",
        "section_title": "Общие требования",
        "section_type": SectionType.PROCEDURA,
        "clause": "3.1",
        "breadcrumb": "ДОК-1 > 3. Общие требования > п. 3.1",
        "text": "ДОК-1 > 3. Общие требования > п. 3.1\nТекст пункта регламента.",
        "text_raw": "Текст пункта регламента.",
    }
    return IngestChunk(**{**defaults, **overrides})


# --------------------------------------------------------------------- BM25


def test_bm25_document_always_sets_russian() -> None:
    """Язык обязателен: без него сервер считает текст английским.

    Это самый дорогой из молчаливых дефектов: ошибки нет, а лексический
    канал на русских регламентах перестаёт находить словоформы.
    """
    document = bm25_document("средства индивидуальной защиты")
    dumped = document.model_dump()
    assert dumped["model"] == BM25_MODEL
    assert dumped["options"]["language"] == "russian"


def test_bm25_options_are_shared_between_write_and_query() -> None:
    """Опции записи и запроса — один и тот же объект.

    Документация Qdrant требует идентичности опций при индексации и поиске.
    Расхождение не даёт ошибки, поэтому единственная защита — общий источник.
    """
    # pydantic копирует словарь при валидации, поэтому сверяем по значению:
    # важно, что оба конца берут опции из одной константы.
    assert bm25_document("а").options == BM25_OPTIONS
    assert bm25_document("б").options == bm25_document("а").options


def test_bm25_options_do_not_cut_short_tokens() -> None:
    """Короткие аббревиатуры не отсекаются.

    На «ГО», «ЧС», «СИЗ» построена часть запросов по этому корпусу, а
    ``min_token_len`` выбросил бы их целиком.
    """
    assert "min_token_len" not in BM25_OPTIONS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1.19.0", (1, 19, 0)), ("1.16.0-rc", (1, 16, 0)), ("2.0", (2, 0, 0))],
)
def test_parse_version(raw: str, expected: tuple[int, int, int]) -> None:
    """Разбор версии сервера устойчив к суффиксам и коротким формам."""
    assert _parse_version(raw) == expected


def test_parse_version_survives_garbage() -> None:
    """Неразбираемая версия не роняет приложение."""
    assert _parse_version("не-версия") == (0, 0, 0)


# ------------------------------------------------------------------- payload


def test_payload_contains_both_filter_axes() -> None:
    """В payload есть обе оси фильтрации: предметная и структурная."""
    payload = chunk_payload(make_chunk(), ["pozharnaya_bezopasnost"], "run-1")
    assert payload["area"] == ["pozharnaya_bezopasnost"]
    assert payload["section_type"] == "procedura"


def test_payload_indexes_cover_filtered_fields() -> None:
    """Индексы созданы ровно для полей, по которым идёт фильтрация."""
    assert {"area", "section_type", "doc_id", "sync_tag"} <= set(PAYLOAD_INDEXES)
    # index_version намеренно без индекса: кардинальность 1-2.
    assert "index_version" not in PAYLOAD_INDEXES


def test_point_id_is_stable_and_order_independent() -> None:
    """Идентификатор точки зависит от содержимого, а не от порядка чанка.

    Это защита от молчаливой порчи индекса: при вставке абзаца в середину
    документа порядковые идентификаторы сдвигаются, и загрузка перезаписала
    бы чужие чанки, не сообщив об этом.
    """
    first = make_chunk(chunk_id="ДОК-1#c0001", order_no=1)
    same_text_other_position = make_chunk(chunk_id="ДОК-1#c0099", order_no=99)
    other_text = make_chunk(text_raw="Совсем другой текст пункта.")

    assert first.point_id() == same_text_other_position.point_id()
    assert first.point_id() != other_text.point_id()


def test_point_id_survives_document_code_change() -> None:
    """Смена кода документа не осиротит все точки разом."""
    before = make_chunk(doc_code="ДОК-1")
    after = make_chunk(doc_code="ДОК-01")
    assert before.point_id() == after.point_id()


# -------------------------------------------------------------------- фильтр


def test_filter_lives_in_prefetch_not_in_query_filter() -> None:
    """Фильтр собирается отдельно, чтобы уйти внутрь каждой ветки предвыборки.

    Ограничение выражается там, где отбираются кандидаты, и потому не зависит
    от того, проталкивает ли конкретная версия сервера верхний фильтр в ветки
    слияния (v1.19.0 проталкивает, см. живой набор тестов).
    """
    built = QdrantSearch._build_filter(["ohrana_truda"], ["procedura"])
    assert isinstance(built, models.Filter)
    keys = {condition.key for condition in built.must}
    assert keys == {"area", "section_type"}


def test_empty_selection_means_whole_base() -> None:
    """Пустой выбор означает поиск по всей базе, а не по пустому множеству."""
    assert QdrantSearch._build_filter(None, None) is None
    assert QdrantSearch._build_filter([], []) is None
    assert QdrantSearch._build_filter([], [], []) is None


def test_document_level_narrows_filter() -> None:
    """Третий уровень дерева областей знаний - конкретный документ.

    Поле ``doc_code`` проиндексировано как KEYWORD наравне с ``area`` и
    ``section_type``, поэтому отбор по документу идёт по индексу.
    """
    built = QdrantSearch._build_filter(None, None, ["Р-У2-59"])
    assert isinstance(built, models.Filter)
    assert {condition.key for condition in built.must} == {"doc_code"}


def test_three_levels_combine_as_conjunction() -> None:
    """Область, документ и тип раздела сужают выдачу одновременно."""
    built = QdrantSearch._build_filter(
        ["ohrana_truda"], ["procedura"], ["Р-У2-59", "Р-У2-23"]
    )
    assert isinstance(built, models.Filter)
    assert {condition.key for condition in built.must} == {
        "area",
        "section_type",
        "doc_code",
    }


# ----------------------------------------------------------------- поведение


class _BrokenEmbedder:
    """Эмбеддер, который всегда падает."""

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Имитирует недоступную модель.

        Args:
            texts: Тексты для кодирования.

        Raises:
            OSError: Всегда.
        """
        raise OSError("модель эмбеддингов недоступна")


async def test_unavailable_index_raises_instead_of_empty_result() -> None:
    """Авария поиска отличима от честного «ничего не найдено».

    Если вернуть пустой список, ассистент ответит «В базе знаний информации
    нет» — авария замаскируется под правильный ответ, и на защите жюри
    получит эту фразу на каждый вопрос.
    """
    search = QdrantSearch(client=None, embedder=_BrokenEmbedder(), collection="c")
    with pytest.raises(SearchUnavailable):
        await search.query("как часто проводится СОУТ")


async def test_empty_query_returns_empty_without_touching_index() -> None:
    """Пустой запрос не идёт ни в модель, ни в индекс."""
    search = QdrantSearch(client=None, embedder=_BrokenEmbedder(), collection="c")
    assert await search.query("   ") == []


def test_chunk_is_restored_from_payload() -> None:
    """Чанк собирается обратно из payload без потери полей цитаты."""
    payload = chunk_payload(make_chunk(), ["go_i_chs"], "run-1")
    chunk = QdrantSearch._to_chunk(payload)
    assert chunk.doc_code == "ДОК-1"
    assert chunk.clause == "3.1"
    assert chunk.section == "3"


# ------------------------------------------------------------------- области


def test_every_document_gets_at_least_one_area() -> None:
    """Документ без области невидим при любом выборе — так быть не должно."""
    assert classify_areas("Совершенно посторонний документ") == [UNCLASSIFIED]
    assert classify_areas("Правила пожарной безопасности")


def test_document_may_belong_to_several_areas() -> None:
    """Области не взаимоисключающие."""
    areas = classify_areas(
        "Регламент проверки знаний руководителей промышленной и пожарной безопасности"
    )
    assert "pozharnaya_bezopasnost" in areas
    assert "promyshlennaya_bezopasnost" in areas


class _Point:
    """Точка выдачи Qdrant."""

    def __init__(self, n: int) -> None:
        """Готовит точку.

        Args:
            n: Порядковый номер, попадает в идентификатор фрагмента.
        """
        self.score = 1.0 / n
        self.payload = {
            "chunk_id": f"R-U2-59#c{n:03d}",
            "doc_code": "Р-У2-59",
            "doc_title": "Регламент",
            "section_no": "4",
            "section_type": "procedura",
            "clause": f"4.{n}",
            "text": f"фрагмент {n}",
        }


class _Response:
    """Ответ Qdrant с готовыми точками."""

    def __init__(self, count: int) -> None:
        """Готовит ответ.

        Args:
            count: Сколько точек вернуть.
        """
        self.points = [_Point(n) for n in range(1, count + 1)]


class _FakeClient:
    """Клиент Qdrant, отдающий фиксированную выдачу."""

    async def query_points(self, **kwargs):
        """Возвращает четыре точки.

        Args:
            **kwargs: Параметры запроса, не используются.

        Returns:
            _Response: Ответ с точками.
        """
        return _Response(4)


class _FakeEmbedder:
    """Эмбеддер, отдающий фиксированный вектор."""

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Кодирует тексты нулевым вектором.

        Args:
            texts: Тексты для кодирования.

        Returns:
            list[list[float]]: По вектору на текст.
        """
        return [[0.0, 0.0, 0.0] for _ in texts]


class _FlakyReranker:
    """Переоценка, которая падает заданное число раз, затем работает."""

    def __init__(self, failures: int) -> None:
        """Готовит реранкер.

        Args:
            failures: Сколько первых вызовов должны упасть.
        """
        self._left = failures
        self.calls = 0
        self._candidates = 10

    def rerank(self, query, hits, limit):
        """Переоценивает кандидатов, роняя первые вызовы.

        Args:
            query: Текст запроса.
            hits: Кандидаты.
            limit: Сколько вернуть.

        Returns:
            list: Кандидаты в обратном порядке - видно, что переоценка была.

        Raises:
            RerankerUnavailable: Пока не исчерпан счётчик падений.
        """
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise RerankerUnavailable("сервис не ответил")
        return list(reversed(hits))[:limit]


async def test_transient_rerank_failure_does_not_disable_it_forever() -> None:
    """Разовый сбой переоценки не выключает её до перезапуска процесса.

    У реранкера своё остывание на 30 с, но поиск обнулял ссылку на него, и
    остывание не срабатывало никогда: одна сетевая заминка стоила +17 пунктов
    hit@1 на весь оставшийся сеанс, причём молча.
    """
    reranker = _FlakyReranker(failures=1)
    search = QdrantSearch(
        _FakeClient(),
        _FakeEmbedder(),
        collection="c",
        top_k=2,
        reranker=reranker,
    )

    first = await search.query("вопрос")
    assert first, "при сбое переоценки выдача остаётся, просто без неё"
    assert search._reranker is reranker, "реранкер не выброшен"

    await search.query("вопрос")
    assert reranker.calls == 2, "вторая попытка переоценки состоялась"
