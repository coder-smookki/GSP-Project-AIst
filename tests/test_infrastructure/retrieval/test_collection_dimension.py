"""Тесты сверки размерности вектора с существующей коллекцией.

Размерность прошита в схему коллекции и на лету не меняется. Раньше
существующая коллекция принималась как есть, поэтому смена модели
эмбеддингов проходила незамеченной: документы пропускались как
неизменившиеся, запись не выполнялась, а падало это позже и в другом месте -
на первом же запросе с «Vector dimension error». Связать такое сообщение с
индексацией трудно, поэтому расхождение обязано вскрываться сразу.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infrastructure.retrieval.bm25 import DENSE_VECTOR
from infrastructure.retrieval.store import QdrantStore


class _Client:
    """Клиент Qdrant с коллекцией заданной размерности."""

    def __init__(self, dim: int | None, collections: tuple[str, ...] = ()):
        """Готовит клиента.

        Args:
            dim: Размерность dense-вектора; ``None`` - коллекции нет.
            collections: Имена НАСТОЯЩИХ коллекций. Алиасы сюда не входят:
                на них ``collection_exists`` отвечает утвердительно, и
                проверка коллизии имён обязана смотреть именно этот список.
        """
        self._dim = dim
        self._collections = collections
        self.created = False

    async def collection_exists(self, collection_name: str) -> bool:
        """Сообщает, существует ли коллекция.

        Args:
            collection_name: Имя коллекции.

        Returns:
            bool: ``True``, если размерность задана.
        """
        return self._dim is not None

    async def get_collection(self, collection_name: str):
        """Отдаёт описание коллекции.

        Args:
            collection_name: Имя коллекции.

        Returns:
            SimpleNamespace: Описание с размерностью dense-вектора.
        """
        vectors = {DENSE_VECTOR: SimpleNamespace(size=self._dim)}
        params = SimpleNamespace(vectors=vectors)
        return SimpleNamespace(config=SimpleNamespace(params=params))

    async def create_collection(self, **kwargs) -> None:
        """Отмечает создание коллекции.

        Args:
            **kwargs: Параметры создания.
        """
        self.created = True

    async def create_payload_index(self, **kwargs) -> None:
        """Заглушка создания индекса по payload.

        Args:
            **kwargs: Параметры индекса.
        """

    async def get_collections(self):
        """Отдаёт перечень НАСТОЯЩИХ коллекций.

        Алиасы сюда не входят намеренно: ``collection_exists`` отвечает на
        них утвердительно, и проверка коллизии имён обязана смотреть именно
        этот список, иначе она срабатывает на каждом законном переключении.

        Returns:
            SimpleNamespace: Поле ``collections`` со списком имён.
        """
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in self._collections]
        )


@pytest.fixture(autouse=True)
def _skip_version_check(monkeypatch):
    """Отключает проверку версии сервера: она ходит в сеть.

    Args:
        monkeypatch: Штатная фикстура подмены значений.
    """

    async def _noop(client):
        return None

    monkeypatch.setattr("infrastructure.retrieval.store.assert_server_supported", _noop)


async def test_mismatch_is_reported_with_instructions() -> None:
    """Расхождение размерности вскрывается сразу и подсказывает, что делать."""
    store = QdrantStore(_Client(2560), alias="aist_chunks")
    with pytest.raises(ValueError) as info:
        await store.create_generation("aist_chunks", 1024)

    message = str(info.value)
    assert "2560" in message and "1024" in message
    assert "--generation" in message and "--switch-alias" in message


async def test_matching_dimension_passes() -> None:
    """Совпадение размерности проходит без создания новой коллекции."""
    client = _Client(1024)
    store = QdrantStore(client, alias="aist_chunks")
    await store.create_generation("aist_chunks", 1024)
    assert client.created is False


async def test_missing_collection_is_created() -> None:
    """Отсутствующая коллекция создаётся заданной размерностью."""
    client = _Client(None)
    store = QdrantStore(client, alias="aist_chunks")
    await store.create_generation("aist_chunks", 1024)
    assert client.created is True


class TestAliasCollision:
    """Имя алиаса не должно совпадать с именем существующей коллекции.

    Qdrant держит алиасы и коллекции в одном пространстве имён и отвечает
    сырым «409 Conflict: Collection ... already exists». По такому
    сообщению причину не угадать, а она в настройке: `QDRANT_COLLECTION`
    задаёт имя алиаса, а не поколения.
    """

    async def test_collision_is_reported_with_instructions(self) -> None:
        """Совпадение имён вскрывается до запроса к серверу."""
        store = QdrantStore(
            _Client(1024, collections=("aist_chunks_v2",)),
            alias="aist_chunks_v2",
        )
        with pytest.raises(ValueError) as info:
            await store.switch_generation("aist_chunks_v2_v1")

        message = str(info.value)
        assert "QDRANT_COLLECTION" in message
        assert "aist_chunks" in message
        assert "--generation" in message

    async def test_free_alias_name_passes_the_check(self) -> None:
        """Свободное имя алиаса проверку проходит."""

        class _NoCollections(_Client):
            async def collection_exists(self, collection_name: str) -> bool:
                """Сообщает, что коллекции с таким именем нет.

                Args:
                    collection_name: Имя коллекции.

                Returns:
                    bool: Всегда ``False``.
                """
                return False

            async def get_aliases(self):
                """Отдаёт пустой перечень алиасов.

                Returns:
                    SimpleNamespace: Пустой перечень.
                """
                return SimpleNamespace(aliases=[])

            async def update_collection_aliases(self, **kwargs) -> None:
                """Отмечает переключение алиаса.

                Args:
                    **kwargs: Операции над алиасами.
                """
                self.switched = True

        client = _NoCollections(1024)
        store = QdrantStore(client, alias="aist_chunks")
        assert await store.switch_generation("aist_chunks_v1") is None
