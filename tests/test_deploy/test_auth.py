"""Аутентификация Aegra: доверенный посредник плюс имя пользователя.

Модуль лежит в ``deploy/`` и грузится так же, как его грузит сама Aegra -
по пути из ``aegra.json``, а не импортом пакета.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from langgraph_sdk import Auth

_AUTH_PATH = Path(__file__).resolve().parents[2] / "deploy" / "auth.py"


def _load_auth_module():
    """Загружает ``deploy/auth.py`` как отдельный модуль.

    Returns:
        ModuleType: Загруженный модуль аутентификации.
    """
    spec = importlib.util.spec_from_file_location("aist_auth_under_test", _AUTH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def auth_module(monkeypatch):
    """Отдаёт модуль аутентификации с заданным токеном доступа.

    Args:
        monkeypatch: Штатная фикстура подмены окружения.

    Returns:
        ModuleType: Модуль с ``AEGRA_API_TOKEN=правильный-токен``.
    """
    monkeypatch.setenv("AEGRA_API_TOKEN", "правильный-токен")
    return _load_auth_module()


def _headers(token: str | None = "правильный-токен", user: str | None = "ivanov"):
    """Собирает заголовки запроса.

    Args:
        token: Токен для ``Authorization``; ``None`` - заголовка нет.
        user: Значение ``X-Aist-User``; ``None`` - заголовка нет.

    Returns:
        dict[str, str]: Заголовки в нижнем регистре, как их отдаёт Aegra.
    """
    headers: dict[str, str] = {}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    if user is not None:
        headers["x-aist-user"] = user
    return headers


async def test_valid_token_and_user_are_accepted(auth_module) -> None:
    """Верный токен и имя дают личность, под которой работает граф."""
    user = await auth_module.authenticate(_headers())

    assert user["identity"] == "ivanov"
    assert user["is_authenticated"] is True


async def test_wrong_token_is_rejected(auth_module) -> None:
    """Чужой токен не пускает."""
    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        await auth_module.authenticate(_headers(token="подобранный"))

    assert excinfo.value.status_code == 401


async def test_missing_authorization_header_is_rejected(auth_module) -> None:
    """Без заголовка доступа запрос не проходит."""
    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        await auth_module.authenticate(_headers(token=None))

    assert excinfo.value.status_code == 401


async def test_non_bearer_scheme_is_rejected(auth_module) -> None:
    """Схема, отличная от Bearer, не принимается."""
    with pytest.raises(Auth.exceptions.HTTPException):
        await auth_module.authenticate({"authorization": "Basic правильный-токен"})


async def test_user_header_is_required(auth_module) -> None:
    """Токен без имени пользователя не даёт доступа.

    Иначе все запросы снова слились бы в одну личность, а память стала бы
    общей - ровно то, ради чего этот модуль и появился.
    """
    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        await auth_module.authenticate(_headers(user=None))

    assert excinfo.value.status_code == 401


@pytest.mark.parametrize(
    "name",
    [
        "../../etc/passwd",
        "memories/admin",
        "имя с пробелом",
        "a" * 65,
        "",
    ],
)
async def test_dangerous_user_names_are_rejected(auth_module, name: str) -> None:
    """Имя становится частью пространства памяти, поэтому набор символов узкий.

    Args:
        auth_module: Модуль аутентификации.
        name: Проверяемое имя пользователя.
    """
    with pytest.raises(Auth.exceptions.HTTPException):
        await auth_module.authenticate(_headers(user=name))


async def test_service_without_token_refuses_everyone(monkeypatch) -> None:
    """Ненастроенный токен закрывает доступ, а не открывает его.

    Молчаливый откат к общей личности ``anonymous`` - именно то поведение,
    которое здесь и чинится, поэтому пустая настройка обязана отказывать.
    """
    monkeypatch.delenv("AEGRA_API_TOKEN", raising=False)
    module = _load_auth_module()

    with pytest.raises(Auth.exceptions.HTTPException) as excinfo:
        await module.authenticate(_headers())

    assert excinfo.value.status_code == 503


async def test_resources_are_scoped_to_their_owner(auth_module) -> None:
    """Создаваемый ресурс помечается владельцем, выборка им же ограничивается."""

    class _User:
        identity = "ivanov"

    class _Ctx:
        user = _User()

    value: dict = {}
    filters = await auth_module.scope_to_owner(_Ctx(), value)

    assert filters == {"owner": "ivanov"}
    assert value["metadata"]["owner"] == "ivanov"


def test_ownership_filter_covers_threads_only(auth_module) -> None:
    """Фильтр владельца висит на тредах, а не на всех ресурсах разом.

    Общий обработчик (``@auth.on``) захватывал и ассистента. Ассистент
    заводится автоматически и принадлежит ``system``, поэтому для любого
    живого пользователя список ассистентов оказывался пустым, интерфейс
    писал «Сервис агента недоступен», и ни один вопрос не доходил до графа.
    """
    resources = {resource for resource, _action in auth_module.auth._handlers}

    assert resources == {"threads"}
    assert not auth_module.auth._global_handlers
