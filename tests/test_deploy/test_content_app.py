"""Маршруты каталога, встроенные в Aegra.

Проверяются через ASGI-транспорт: сервер не поднимается, сеть не нужна.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

from infrastructure.retrieval.catalog import CatalogUnavailable

_APP_PATH = Path(__file__).resolve().parents[2] / "deploy" / "content_app.py"


def _load_app_module():
    """Загружает ``deploy/content_app.py`` как отдельный модуль.

    Returns:
        ModuleType: Модуль с приложением маршрутов каталога.
    """
    spec = importlib.util.spec_from_file_location("aist_content_under_test", _APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubCatalog:
    """Каталог с заранее заданными ответами."""

    def __init__(self, *, tree=None, document=None, error: Exception | None = None):
        """Готовит заглушку.

        Args:
            tree: Ответ метода ``tree``.
            document: Ответ метода ``document``.
            error: Исключение, которое надо поднять вместо ответа.
        """
        self._tree = tree or []
        self._document = document or {"doc_code": "", "doc_title": "", "chunks": []}
        self._error = error

    async def tree(self):
        """Отдаёт дерево или поднимает заданную ошибку.

        Returns:
            list: Дерево областей.

        Raises:
            Exception: Если задана в конструкторе.
        """
        if self._error:
            raise self._error
        return self._tree

    async def document(self, doc_code: str):
        """Отдаёт документ или поднимает заданную ошибку.

        Args:
            doc_code: Код документа.

        Returns:
            dict: Документ.

        Raises:
            Exception: Если задана в конструкторе.
        """
        if self._error:
            raise self._error
        return self._document


TOKEN = "t" * 32
HEADERS = {"Authorization": f"Bearer {TOKEN}", "X-Aist-User": "ivanov"}


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    """Даёт маршрутам настроенный токен доступа.

    Args:
        monkeypatch: Штатная фикстура подмены окружения.
    """
    monkeypatch.setenv("AEGRA_API_TOKEN", TOKEN)


@pytest.fixture
def module():
    """Отдаёт свежезагруженный модуль маршрутов.

    Returns:
        ModuleType: Модуль ``content_app``.
    """
    return _load_app_module()


def _client(module) -> httpx.AsyncClient:
    """Создаёт клиента поверх приложения без сетевого сервера.

    Args:
        module: Модуль с приложением.

    Returns:
        httpx.AsyncClient: Клиент ASGI-транспорта.
    """
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=module.app), base_url="http://test"
    )


async def test_tree_is_served(module, monkeypatch) -> None:
    """Дерево областей отдаётся под ключом ``areas``."""
    tree = [{"area": "ohrana_truda", "title": "Охрана труда", "documents": []}]
    monkeypatch.setattr(module, "_catalog", lambda: _StubCatalog(tree=tree))

    async with _client(module) as client:
        response = await client.get("/knowledge/tree", headers=HEADERS)

    assert response.status_code == 200
    assert response.json() == {"areas": tree}


async def test_document_is_served(module, monkeypatch) -> None:
    """Документ отдаётся вместе с фрагментами."""
    document = {
        "doc_code": "Р-У2-59",
        "doc_title": "Идентификация опасностей",
        "chunks": [{"order_no": 0, "text": "текст"}],
    }
    monkeypatch.setattr(module, "_catalog", lambda: _StubCatalog(document=document))

    async with _client(module) as client:
        response = await client.get("/knowledge/documents/Р-У2-59", headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["doc_code"] == "Р-У2-59"


async def test_unknown_document_is_404_not_empty_body(module, monkeypatch) -> None:
    """Отсутствующий документ - 404, а не пустой успешный ответ.

    Пустой успешный ответ интерфейс показал бы как «документ без содержания».
    """
    monkeypatch.setattr(module, "_catalog", lambda: _StubCatalog())

    async with _client(module) as client:
        response = await client.get("/knowledge/documents/Р-У2-99", headers=HEADERS)

    assert response.status_code == 404


async def test_unavailable_index_is_503_not_empty_tree(module, monkeypatch) -> None:
    """Недоступный индекс - 503, а не пустое дерево.

    Пустое дерево читается как «база пуста», и пользователь молча остался бы
    без выбора областей знаний.
    """
    monkeypatch.setattr(
        module,
        "_catalog",
        lambda: _StubCatalog(error=CatalogUnavailable("коллекции нет")),
    )

    async with _client(module) as client:
        response = await client.get("/knowledge/tree", headers=HEADERS)

    assert response.status_code == 503


async def test_overlong_document_code_is_refused(module, monkeypatch) -> None:
    """Слишком длинный код отсекается до обращения к индексу."""
    monkeypatch.setattr(module, "_catalog", lambda: _StubCatalog())

    async with _client(module) as client:
        response = await client.get("/knowledge/documents/" + "x" * 400, headers=HEADERS)

    assert response.status_code == 422


async def test_catalog_is_closed_without_a_token(module, monkeypatch) -> None:
    """Без токена каталог не отдаёт состав базы знаний.

    Флаг Aegra ``enable_custom_route_auth`` пишет в лог, что защиту применил,
    а маршрут при этом отвечал: весь перечень документов читался кем угодно,
    кто дотянулся до порта. Поэтому проверка своя.
    """
    monkeypatch.setattr(module, "_catalog", lambda: _StubCatalog(tree=[]))

    async with _client(module) as client:
        assert (await client.get("/knowledge/tree")).status_code == 401
        bad = {"Authorization": "Bearer wrong", "X-Aist-User": "ivanov"}
        assert (await client.get("/knowledge/tree", headers=bad)).status_code == 401
        no_user = {"Authorization": f"Bearer {TOKEN}"}
        assert (await client.get("/knowledge/tree", headers=no_user)).status_code == 401
