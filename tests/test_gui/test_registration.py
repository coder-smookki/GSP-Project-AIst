"""Самостоятельная регистрация: код приглашения и проверки ввода.

Регистрация закрыта кодом намеренно: заказчик просил, чтобы учётные записи
заводил администратор. Открытая страница означала бы, что доступ к
нормативной базе заводит себе любой, кто дотянулся до порта.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gui.register import _MIN_PASSWORD, _USERNAME, register_routes

CODE = "код-приглашения-1"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Отдаёт клиента приложения с включённой регистрацией.

    Args:
        monkeypatch: Штатная фикстура подмены окружения.

    Returns:
        TestClient: Клиент без сетевого сервера.
    """
    monkeypatch.setenv("REGISTRATION_CODE", CODE)
    app = FastAPI()
    register_routes(app)
    return TestClient(app, follow_redirects=False)


def test_page_is_hidden_without_a_code(monkeypatch) -> None:
    """Без кода в окружении страницы не существует.

    Именно 404, а не «регистрация выключена»: посторонний не должен узнавать,
    что такая страница вообще предусмотрена.
    """
    monkeypatch.delenv("REGISTRATION_CODE", raising=False)
    app = FastAPI()
    register_routes(app)

    assert TestClient(app).get("/register").status_code == 404


def test_page_is_served_with_a_code(client: TestClient) -> None:
    """С кодом страница отдаётся и содержит поле для него."""
    response = client.get("/register")

    assert response.status_code == 200
    assert "Код приглашения" in response.text


def test_wrong_code_is_refused(client: TestClient) -> None:
    """Неверный код не заводит учётную запись."""
    response = client.post(
        "/register",
        data={"username": "ivanov", "password": "длинный-пароль", "code": "чужой"},
    )

    assert response.status_code == 403


@pytest.mark.parametrize(
    "name",
    ["../etc/passwd", "ab", "имя-кириллицей", "a" * 65, "с пробелом"],
)
def test_dangerous_usernames_are_refused(name: str) -> None:
    """Логин ограничен узким набором символов.

    Он попадает в базу и в заголовки, поэтому разделители путей, пробелы и
    слишком короткие имена не проходят.

    Args:
        name: Проверяемое имя.
    """
    assert not _USERNAME.match(name)


def test_short_password_is_refused(client: TestClient) -> None:
    """Пароль короче предела не принимается."""
    response = client.post(
        "/register",
        data={"username": "ivanov", "password": "1234", "code": CODE},
    )

    assert response.status_code == 400
    assert str(_MIN_PASSWORD) in response.text


def test_rate_limit_stops_a_flood(client: TestClient) -> None:
    """Частые попытки упираются в предел.

    Без него код приглашения подбирается перебором за считанные минуты.
    """
    codes = [
        client.post(
            "/register",
            data={"username": "ivanov", "password": "длинный-пароль", "code": "нет"},
        ).status_code
        for _ in range(14)
    ]

    assert 429 in codes, "предел частоты не сработал"
