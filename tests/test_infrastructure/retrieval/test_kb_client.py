"""Клиент базы знаний: вход, перечисление статей, загрузка, повторы.

Слой в 936 строк не был покрыт ни одним тестом, хотя именно через него
приходит весь корпус. Проверяется на подменённом транспорте: ни сети, ни
настоящих учётных данных не требуется, поэтому тесты идут в обычном прогоне.
"""

from __future__ import annotations

import json

import httpx
import pytest

from infrastructure.retrieval.kb_client import (
    AuthError,
    KnowledgeBaseClient,
    KnowledgeBaseConfig,
    TransportError,
)

CONFIG = KnowledgeBaseConfig(
    base_url="https://kb.example/",
    email="user@example",
    password="секрет",
    section_id=44030,
    token_name="aist",
    per_page=2,
    max_retries=2,
    backoff_base=0.0,
    backoff_max=0.0,
)


def _client(handler) -> KnowledgeBaseClient:
    """Собирает клиента поверх подменённого транспорта.

    Args:
        handler: Обработчик запросов ``httpx.MockTransport``.

    Returns:
        KnowledgeBaseClient: Клиент, не выходящий в сеть.
    """
    # Клиент принимает транспорт параметром - он для этого и задуман.
    return KnowledgeBaseClient(CONFIG, transport=httpx.MockTransport(handler))


def _ok(payload: dict) -> httpx.Response:
    """Успешный ответ с телом JSON.

    Args:
        payload: Тело ответа.

    Returns:
        httpx.Response: Ответ 200.
    """
    return httpx.Response(200, json=payload)


async def test_login_stores_the_access_token() -> None:
    """Вход запоминает токен доступа и токен обновления."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok({"access_token": "A1", "refresh_token": "R1", "expires_in": 3600})

    client = _client(handler)
    await client.login()

    assert seen[0] == {
        "email": "user@example",
        "password": "секрет",
        "token_name": "aist",
    }
    assert client._access_token == "A1"


async def test_login_without_a_token_is_an_auth_error() -> None:
    """Ответ без ``access_token`` не считается успешным входом.

    Иначе клиент пошёл бы дальше без токена и упал бы позже, на первом же
    запросе, с непонятным сообщением.
    """
    client = _client(lambda request: _ok({"detail": "ok"}))

    with pytest.raises(AuthError, match="access_token"):
        await client.login()


async def test_password_never_appears_in_the_error_text() -> None:
    """Пароль не попадает в текст ошибки.

    Сообщения об ошибках уходят в логи и в отчёты о прогоне.
    """
    client = _client(lambda request: httpx.Response(401, text="нет доступа"))

    with pytest.raises((AuthError, TransportError)) as info:
        await client.login()

    assert CONFIG.password not in str(info.value)


async def test_article_listing_walks_every_page() -> None:
    """Перечисление статей обходит страницы до пустой.

    Раздел заказчика не помещается на одну страницу, и обрыв на первой
    означал бы молча неполный корпус.
    """
    # Обход управляется полем meta.last_page - именно так отвечает сервер.
    pages = {
        "1": [{"id": 1, "name": "Первая"}, {"id": 2, "name": "Вторая"}],
        "2": [{"id": 3, "name": "Третья"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/logininng"):
            return _ok({"access_token": "A1"})
        page = request.url.params.get("page", "1")
        return _ok({"data": pages.get(page, []), "meta": {"last_page": 2}})

    client = _client(handler)
    await client.login()
    articles = await client.list_section_articles()

    assert [a["id"] for a in articles] == [1, 2, 3]


async def test_listing_uses_the_configured_section() -> None:
    """Раздел берётся из конфигурации, если не задан явно."""
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/logininng"):
            return _ok({"access_token": "A1"})
        return _ok({"data": [], "meta": {"last_page": 1}})

    client = _client(handler)
    await client.login()
    await client.list_section_articles()

    assert any("/section/44030/articles" in p for p in paths)


async def test_article_is_returned_from_the_data_field() -> None:
    """Статья достаётся из поля ``data`` ответа."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/logininng"):
            return _ok({"access_token": "A1"})
        return _ok({"data": {"id": 7, "name": "Регламент", "content": "<p>текст</p>"}})

    client = _client(handler)
    await client.login()
    article = await client.get_article(7)

    assert article["id"] == 7
    assert article["content"] == "<p>текст</p>"


async def test_article_without_data_is_a_transport_error() -> None:
    """Ответ без объекта статьи отличается от пустой статьи.

    Пустая статья - это данные, а отсутствие поля ``data`` - поломка
    протокола, и молчать о ней нельзя.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/logininng"):
            return _ok({"access_token": "A1"})
        return _ok({"detail": "not found"})

    client = _client(handler)
    await client.login()

    with pytest.raises(TransportError, match="data"):
        await client.get_article(7)


async def test_temporary_failure_is_retried() -> None:
    """Временная ошибка сервера повторяется, а не роняет прогон.

    Индексация идёт десятками минут: разовая пятисотка не должна стоить
    всего прогона.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/logininng"):
            return _ok({"access_token": "A1"})
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="перегрузка")
        return _ok({"data": {"id": 7, "content": "ок"}})

    client = _client(handler)
    await client.login()
    article = await client.get_article(7)

    assert calls["n"] == 2
    assert article["content"] == "ок"


async def test_retries_are_bounded() -> None:
    """Повторы не бесконечны: устойчивый отказ доходит до вызывающего."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/logininng"):
            return _ok({"access_token": "A1"})
        calls["n"] += 1
        return httpx.Response(503, text="перегрузка")

    client = _client(handler)
    await client.login()

    with pytest.raises(TransportError):
        await client.get_article(7)

    assert calls["n"] == CONFIG.max_retries + 1


async def test_expired_token_triggers_a_single_reauthentication() -> None:
    """На 401 клиент входит заново и повторяет запрос один раз.

    Токен живёт меньше, чем идёт полная индексация, поэтому обновление
    обязано происходить само.
    """
    logins = {"n": 0}
    articles = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/logininng"):
            logins["n"] += 1
            return _ok({"access_token": f"A{logins['n']}"})
        articles["n"] += 1
        if articles["n"] == 1:
            return httpx.Response(401, text="токен истёк")
        return _ok({"data": {"id": 7, "content": "ок"}})

    client = _client(handler)
    await client.login()
    article = await client.get_article(7)

    assert logins["n"] == 2, "вход выполнен повторно"
    assert article["content"] == "ок"


async def test_base_url_gets_a_trailing_slash() -> None:
    """Корень API приводится к виду со слэшем.

    Без него ``httpx`` отбрасывает последний сегмент пути, и запросы уходят
    не туда, куда написано в настройке.
    """
    config = KnowledgeBaseConfig(
        base_url="https://kb.example/api-root",
        email="u",
        password="p",
        section_id=1,
    )
    assert config.base_url.endswith("/")
