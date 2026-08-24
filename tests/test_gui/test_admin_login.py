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
    # Модуль не импортируется без пригодного ключа подписи сессий.
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "x" * 40)
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


class TestAuthSecretGuard:
    """Ключ подписи сессий проверяется на старте, а не при первом входе."""

    @staticmethod
    def _reload(monkeypatch, secret: str):
        """Перезагружает модуль интерфейса с заданным ключом подписи.

        Args:
            monkeypatch: Штатная фикстура подмены окружения.
            secret: Значение ``CHAINLIT_AUTH_SECRET``.

        Returns:
            ModuleType: Загруженный модуль.
        """
        monkeypatch.setenv("CHAINLIT_AUTH_SECRET", secret)
        module = importlib.import_module("gui.chainlit_app")
        return importlib.reload(module)

    def test_valid_secret_starts(self, monkeypatch) -> None:
        """Собственный длинный ключ пропускает запуск."""
        assert self._reload(monkeypatch, "x" * 40) is not None

    def test_placeholder_secret_refuses_to_start(self, monkeypatch) -> None:
        """Ключ из шаблона не запускает интерфейс.

        Он публично известен, и с ним кто угодно подписывает себе cookie
        администратора. Отказ на старте заметен, а тихая работа с чужим
        ключом - нет.
        """
        with pytest.raises(RuntimeError, match="из шаблона"):
            self._reload(monkeypatch, "change-me-to-a-random-secret")

    def test_empty_secret_refuses_to_start(self, monkeypatch) -> None:
        """Пустой ключ не запускает интерфейс."""
        with pytest.raises(RuntimeError, match="не задан"):
            self._reload(monkeypatch, "")

    def test_short_secret_refuses_to_start(self, monkeypatch) -> None:
        """Слишком короткий ключ не запускает интерфейс."""
        with pytest.raises(RuntimeError, match="короче"):
            self._reload(monkeypatch, "коротко")
