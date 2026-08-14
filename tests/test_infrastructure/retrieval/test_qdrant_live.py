"""Тесты гибридного поиска на живом сервере Qdrant.

Эти тесты обязательны: встроенный BM25 считает сервер, а клиент в локальном
режиме ``:memory:`` его не поддерживает вовсе. Без живого сервера главный
компонент решения остаётся непроверенным, а набор при этом зелёный — то есть
создаёт ложную уверенность.

Запуск сервера:

    docker run -d --name aist-qdrant -p 6333:6333 \\
        -e QDRANT__TELEMETRY_DISABLED=true qdrant/qdrant:v1.19.0

Набор помечен маркером `qdrant` и пропускается, если сервер не отвечает.
Эмбеддинги не используются: дальше проверяется лексический канал и схема,
для которых модель не нужна. Так тесты идут секунды и не требуют Ollama.
"""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client import AsyncQdrantClient, models

from infrastructure.retrieval.bm25 import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    assert_server_supported,
    bm25_document,
)
from infrastructure.retrieval.store import QdrantStore

pytestmark = [pytest.mark.qdrant, pytest.mark.integration]

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

#: Размерность фиктивного dense-вектора. Настоящая модель здесь не нужна:
#: проверяется схема и лексический канал.
DIM = 8


def _vector(seed: int) -> list[float]:
    """Строит воспроизводимый dense-вектор.

    Args:
        seed: Число, определяющее вектор.

    Returns:
        list[float]: Вектор длины `DIM`.
    """
    return [((seed + i) % 7) / 7 + 0.01 for i in range(DIM)]


async def _server_available(client: AsyncQdrantClient) -> bool:
    """Проверяет, отвечает ли сервер.

    Args:
        client: Клиент Qdrant.

    Returns:
        bool: ``True``, если сервер доступен.
    """
    try:
        await client.get_collections()
    except Exception:
        return False
    return True


@pytest.fixture
async def store() -> tuple[QdrantStore, AsyncQdrantClient, str]:
    """Поднимает временную коллекцию и наполняет её тестовыми точками.

    Returns:
        tuple[QdrantStore, AsyncQdrantClient, str]: Хранилище, клиент и имя
        коллекции. Коллекция и алиас удаляются после теста.
    """
    client = AsyncQdrantClient(url=QDRANT_URL, timeout=20, check_compatibility=False)
    if not await _server_available(client):
        await client.close()
        pytest.skip(f"Qdrant не отвечает на {QDRANT_URL}")

    alias = f"aist_test_{uuid.uuid4().hex[:8]}"
    collection = f"{alias}_v1"
    qdrant_store = QdrantStore(client, alias=alias, batch_size=8)
    await qdrant_store.create_generation(collection, DIM)

    samples = [
        (
            "Работник обязан применять средства индивидуальной защиты "
            "при выполнении работ повышенной опасности.",
            ["ohrana_truda"],
            "procedura",
        ),
        (
            "Наряд-допуск оформляется до начала производства работ "
            "и подписывается уполномоченными лицами.",
            ["raboty_povyshennoy_opasnosti"],
            "procedura",
        ),
        (
            "Обучение по гражданской обороне проводится по ежегодному плану "
            "руководителя подразделения.",
            ["go_i_chs"],
            "obshchie",
        ),
    ]
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection}-{index}")),
            vector={
                DENSE_VECTOR: _vector(index),
                SPARSE_VECTOR: bm25_document(text),
            },
            payload={
                "chunk_id": f"TEST#{index:04d}",
                "doc_id": f"doc-{index}",
                "doc_code": f"ДОК-{index}",
                "text": text,
                "area": areas,
                "section_type": section_type,
                "is_table_region": False,
                "sync_tag": "run-test",
            },
        )
        for index, (text, areas, section_type) in enumerate(samples)
    ]
    await client.upsert(collection_name=collection, points=points, wait=True)

    try:
        yield qdrant_store, client, collection
    finally:
        try:
            await client.delete_collection(collection)
        finally:
            await client.close()


async def test_server_version_is_supported(store) -> None:
    """Сервер поддерживает встроенный BM25."""
    _, client, _ = store
    version = await assert_server_supported(client)
    assert version


async def test_point_carries_both_vectors(store) -> None:
    """У точки есть и dense, и sparse вектор.

    Upsert перезаписывает точку целиком, поэтому запись только dense молча
    обнулила бы лексический канал — этот тест такое поймает.
    """
    _, client, collection = store
    records, _ = await client.scroll(
        collection_name=collection, limit=1, with_vectors=True
    )
    assert set(records[0].vector) == {DENSE_VECTOR, SPARSE_VECTOR}


@pytest.mark.parametrize(
    "query",
    [
        "средства индивидуальной защиты",
        "средству индивидуальной защиты",
        "средствами индивидуальной защиты",
    ],
)
async def test_russian_morphology_is_enabled(store, query: str) -> None:
    """Запрос в любом падеже находит документ.

    Единственная проверка, ловящая пропущенный русский стеммер: без
    ``language="russian"`` сервер считает текст английским, словоформы не
    схлопываются, и лексический канал молча перестаёт находить — без ошибки.
    """
    _, client, collection = store
    response = await client.query_points(
        collection_name=collection,
        query=bm25_document(query),
        using=SPARSE_VECTOR,
        limit=3,
        with_payload=True,
    )
    assert response.points, f"BM25 ничего не нашёл по запросу {query!r}"
    assert "индивидуальной защиты" in response.points[0].payload["text"]


async def test_hybrid_query_merges_both_channels(store) -> None:
    """Гибридный запрос со слиянием RRF отрабатывает и отдаёт результаты."""
    _, client, collection = store
    response = await client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=_vector(0), using=DENSE_VECTOR, limit=10),
            models.Prefetch(
                query=bm25_document("наряд-допуск"), using=SPARSE_VECTOR, limit=10
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        limit=3,
        with_payload=True,
    )
    assert response.points
    assert all(point.score > 0 for point in response.points)


async def test_area_filter_works_in_both_placements(store) -> None:
    """Фильтр по области знаний отсекает чужие документы.

    Проверяются обе формы: внутри `Prefetch` (наш выбор) и верхним
    `query_filter`. На сервере v1.19.0 работают обе — распространённое
    утверждение, что при слиянии верхний фильтр игнорируется, для этой
    версии неверно, и проверено это здесь, а не взято на веру.

    Мы всё равно кладём фильтр в `Prefetch`: так ограничение выражено там,
    где отбираются кандидаты, и не зависит от поведения конкретной версии.
    Если поведение когда-нибудь разойдётся, тест это покажет.
    """
    _, client, collection = store
    only_go = models.Filter(
        must=[
            models.FieldCondition(key="area", match=models.MatchAny(any=["go_i_chs"]))
        ]
    )

    in_prefetch = await client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=_vector(2), using=DENSE_VECTOR, filter=only_go, limit=10
            ),
            models.Prefetch(
                query=bm25_document("обучение"),
                using=SPARSE_VECTOR,
                filter=only_go,
                limit=10,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        limit=10,
        with_payload=True,
    )
    assert in_prefetch.points, "фильтр в prefetch не дал ни одного результата"
    assert all("go_i_chs" in p.payload["area"] for p in in_prefetch.points)

    on_top = await client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=_vector(2), using=DENSE_VECTOR, limit=10),
            models.Prefetch(
                query=bm25_document("обучение"), using=SPARSE_VECTOR, limit=10
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        query_filter=only_go,
        limit=10,
        with_payload=True,
    )
    assert all("go_i_chs" in p.payload["area"] for p in on_top.points)


async def test_sweep_removes_stale_points(store) -> None:
    """Уборка удаляет точки документа, помеченные прошлым прогоном."""
    qdrant_store, client, collection = store
    before = (await client.count(collection_name=collection, exact=True)).count

    await qdrant_store.sweep_document(collection, "doc-0", "run-new")
    after = (await client.count(collection_name=collection, exact=True)).count
    assert after == before - 1

    await qdrant_store.sweep_document(collection, "doc-1", "run-test")
    assert (await client.count(collection_name=collection, exact=True)).count == after


async def test_alias_switch_is_atomic_and_reversible(store) -> None:
    """Алиас переключается на коллекцию и читается обратно."""
    qdrant_store, _, collection = store
    assert await qdrant_store.current_generation() is None

    previous = await qdrant_store.switch_generation(collection)
    assert previous is None
    assert await qdrant_store.current_generation() == collection
    assert await qdrant_store.count() > 0
