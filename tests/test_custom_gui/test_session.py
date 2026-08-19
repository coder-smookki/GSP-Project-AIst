"""Подписанная сессия интерфейса.

Cookie - единственное, что отделяет вошедшего человека от постороннего,
поэтому проверяется и подделка, и порча, и срок.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from custom_gui.server.session import (
    COOKIE_NAME,
    InvalidSession,
    issue,
    verify,
)

SECRET = "s" * 40
OTHER_SECRET = "o" * 40


def test_issued_session_verifies() -> None:
    """Своя же сессия проходит проверку и сохраняет имя и роль."""
    cookie = issue("ivanov", "user", SECRET, 3600)
    session = verify(cookie, SECRET)

    assert session.username == "ivanov"
    assert session.role == "user"
    assert session.is_admin is False


def test_admin_role_is_recognised() -> None:
    """Роль администратора видна из сессии."""
    assert verify(issue("admin", "admin", SECRET, 3600), SECRET).is_admin is True


def test_cookie_name_is_stable() -> None:
    """Имя cookie задано явно: от него зависит и выдача, и удаление."""
    assert COOKIE_NAME == "aist_session"


@pytest.mark.parametrize(
    "cookie",
    [None, "", "без-точки", "мусор.мусор", "a.b.c"],
)
def test_broken_cookie_is_refused(cookie: str | None) -> None:
    """Испорченная или отсутствующая cookie не пускает.

    Args:
        cookie: Проверяемое значение.
    """
    with pytest.raises(InvalidSession):
        verify(cookie, SECRET)


def test_session_signed_with_another_key_is_refused() -> None:
    """Сессия, подписанная чужим ключом, не принимается.

    Ровно этим и опасен ключ из шаблона: зная его, посторонний подписывает
    себе любую личность.
    """
    with pytest.raises(InvalidSession):
        verify(issue("admin", "admin", OTHER_SECRET, 3600), SECRET)


def test_tampered_payload_is_refused() -> None:
    """Подмена содержимого ломает подпись.

    Проверяется прямой сценарий повышения прав: обычный пользователь правит
    роль на ``admin``, оставив подпись прежней.
    """
    cookie = issue("ivanov", "user", SECRET, 3600)
    body, _, signature = cookie.partition(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["r"] = "admin"
    forged_body = (
        base64.urlsafe_b64encode(json.dumps(payload).encode())
        .decode()
        .rstrip("=")
    )

    with pytest.raises(InvalidSession):
        verify(f"{forged_body}.{signature}", SECRET)


def test_expired_session_is_refused() -> None:
    """Просроченная сессия не пускает, даже если подпись верна."""
    cookie = issue("ivanov", "user", SECRET, -1)
    with pytest.raises(InvalidSession):
        verify(cookie, SECRET)


def test_session_carries_its_expiry() -> None:
    """Срок годности лежит внутри и доступен вызывающей стороне."""
    session = verify(issue("ivanov", "user", SECRET, 120), SECRET)
    assert 0 < session.expires_at - int(time.time()) <= 120


def test_non_ascii_name_survives_a_round_trip() -> None:
    """Кириллическое имя не ломает подпись и возвращается как есть."""
    assert verify(issue("иванов", "user", SECRET, 60), SECRET).username == "иванов"
