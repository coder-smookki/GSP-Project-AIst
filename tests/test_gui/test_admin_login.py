"""Вход администратора из окружения.

Учётная запись задаётся переменными и работает в обход базы, поэтому её
проверка - самый короткий путь внутрь и разбирается отдельно.
"""

from __future__ import annotations

import importlib

import pytest


def _admin_check(monkeypatch, username: str, password: str):
    """Загружает модуль интерфейса с заданной учётной записью администратора.

    Значения читаются на уровне модуля, поэтому окружение подменяется до
    перезагрузки, а не после.

    Args:
        monkeypatch: Штатная фикстура подмены окружения.
        username: Значение ``ADMIN_USERNAME``.
        password: Значение ``ADMIN_PASSWORD``.

    Returns:
        Callable[[str, str], bool]: Функция проверки учётных данных.
    """
    monkeypatch.setenv("ADMIN_USERNAME", username)
    monkeypatch.setenv("ADMIN_PASSWORD", password)
    module = importlib.import_module("gui.chainlit_app")
    importlib.reload(module)
    return module._is_admin


def test_correct_credentials_are_accepted(monkeypatch) -> None:
    """Верная пара логин-пароль открывает вход."""
    is_admin = _admin_check(monkeypatch, "admin", "s3cret-пароль")
    assert is_admin("admin", "s3cret-пароль") is True


def test_wrong_password_is_rejected(monkeypatch) -> None:
    """Неверный пароль не пускает."""
    is_admin = _admin_check(monkeypatch, "admin", "s3cret")
    assert is_admin("admin", "s3crat") is False


def test_wrong_username_is_rejected(monkeypatch) -> None:
    """Чужое имя не пускает даже с верным паролем."""
    is_admin = _admin_check(monkeypatch, "admin", "s3cret")
    assert is_admin("root", "s3cret") is False


@pytest.mark.parametrize("supplied", ["", "что угодно"])
def test_unset_password_closes_the_door(monkeypatch, supplied: str) -> None:
    """Незаданный пароль закрывает вход, а не открывает его пустой строкой.

    Раньше при заданном имени и пустом ``ADMIN_PASSWORD`` пустой ввод
    совпадал сам с собой, и в систему пускало без пароля вовсе.

    Args:
        monkeypatch: Штатная фикстура подмены окружения.
        supplied: Введённый пароль.
    """
    is_admin = _admin_check(monkeypatch, "admin", "")
    assert is_admin("admin", supplied) is False


def test_non_ascii_password_does_not_crash_login(monkeypatch) -> None:
    """Кириллица в пароле даёт честный отказ, а не ошибку сервера.

    Сравнение постоянного времени на строках не принимает не-ASCII, поэтому
    сравниваются байты; иначе вход падал бы с ``TypeError``.
    """
    is_admin = _admin_check(monkeypatch, "admin", "пароль")
    assert is_admin("admin", "пароль") is True
    assert is_admin("admin", "другой") is False
