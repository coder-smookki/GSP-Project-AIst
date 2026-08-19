"""Настройки бэкенда интерфейса: непригодные значения останавливают запуск."""

from __future__ import annotations

import pytest

from custom_gui.server.config import Config, ConfigError

GOOD_SECRET = "s" * 40
GOOD_TOKEN = "t" * 32


def _env(monkeypatch, **overrides) -> None:
    """Готовит окружение с пригодными значениями и точечными подменами.

    Args:
        monkeypatch: Штатная фикстура подмены окружения.
        **overrides: Переменные, которые надо изменить; ``None`` - удалить.
    """
    base = {
        "CHAINLIT_AUTH_SECRET": GOOD_SECRET,
        "AEGRA_API_TOKEN": GOOD_TOKEN,
        "AEGRA_URL": "http://aegra:2026",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "s3cret",
    }
    base.update(overrides)
    for key, value in base.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_valid_environment_builds_config(monkeypatch) -> None:
    """Полное окружение собирается в конфигурацию."""
    _env(monkeypatch)
    config = Config.from_env()

    assert config.aegra_token == GOOD_TOKEN
    assert config.session_secret == GOOD_SECRET
    assert config.admin_username == "admin"


def test_trailing_slash_is_trimmed(monkeypatch) -> None:
    """Адрес агента приводится к виду без завершающего слэша.

    Иначе пути склеивались бы в двойной слэш.
    """
    _env(monkeypatch, AEGRA_URL="http://aegra:2026/")
    assert Config.from_env().aegra_url == "http://aegra:2026"


@pytest.mark.parametrize(
    ("secret", "message"),
    [
        (None, "не задан"),
        ("change-me-to-a-random-secret", "из шаблона"),
        ("коротко", "короче"),
    ],
)
def test_bad_session_secret_stops_startup(monkeypatch, secret, message) -> None:
    """Непригодный ключ подписи не даёт подняться.

    Ключ из шаблона опаснее пустого: сервис работает как ни в чём не бывало,
    но подписать себе сессию администратора может любой, кто видел репозиторий.

    Args:
        monkeypatch: Штатная фикстура подмены окружения.
        secret: Проверяемое значение ключа.
        message: Ожидаемый фрагмент сообщения.
    """
    _env(monkeypatch, CHAINLIT_AUTH_SECRET=secret)
    with pytest.raises(ConfigError, match=message):
        Config.from_env()


def test_missing_agent_token_stops_startup(monkeypatch) -> None:
    """Без токена агента сервис не поднимается.

    Агент всё равно отказал бы всем запросам, но отказ на старте заметен, а
    ошибка при первом вопросе - уже на демонстрации.
    """
    _env(monkeypatch, AEGRA_API_TOKEN=None)
    with pytest.raises(ConfigError, match="AEGRA_API_TOKEN"):
        Config.from_env()


def test_password_is_not_printed(monkeypatch) -> None:
    """Пароль администратора не попадает в представление объекта.

    Конфигурация мелькает в трассировках и логах.
    """
    _env(monkeypatch)
    assert "s3cret" not in repr(Config.from_env())
