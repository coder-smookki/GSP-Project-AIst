"""Настройки бэкенда интерфейса.

Читаются из окружения, как и у остального стека. Обязательные значения
проверяются на старте: сервис, поднявшийся с пустым ключом подписи или без
токена доступа к агенту, выглядит работающим, но не защищает ничего.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Значение из шаблона ``deploy/.env.example``. Его копируют как есть, и тогда
#: ключ подписи сессий становится публично известной строкой из репозитория.
PLACEHOLDER_SECRET = "change-me-to-a-random-secret"

#: Разумный минимум длины ключа подписи.
MIN_SECRET_LENGTH = 32

#: Сколько живёт cookie сессии, с. Двое суток: демонстрация идёт в один день,
#: а сотруднику не хочется входить заново после обеда.
SESSION_TTL_SECONDS = 172800


class ConfigError(RuntimeError):
    """Настройки непригодны для запуска."""


@dataclass(frozen=True, slots=True)
class Config:
    """Конфигурация бэкенда интерфейса."""

    aegra_url: str
    aegra_token: str
    session_secret: str
    admin_username: str
    admin_password: str = field(repr=False, default="")
    request_timeout: float = 300.0

    @staticmethod
    def from_env() -> Config:
        """Собирает конфигурацию из окружения.

        Returns:
            Config: Проверенная конфигурация.

        Raises:
            ConfigError: Если ключ подписи непригоден или нет токена агента.
        """
        secret = (os.getenv("CHAINLIT_AUTH_SECRET") or "").strip()
        hint = (
            'сгенерируйте: python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
        if not secret:
            raise ConfigError(f"CHAINLIT_AUTH_SECRET не задан - {hint}")
        if secret == PLACEHOLDER_SECRET:
            raise ConfigError(
                f"CHAINLIT_AUTH_SECRET взят из шаблона: этим ключом кто угодно "
                f"подпишет себе сессию администратора - {hint}"
            )
        if len(secret) < MIN_SECRET_LENGTH:
            raise ConfigError(
                f"CHAINLIT_AUTH_SECRET короче {MIN_SECRET_LENGTH} знаков - {hint}"
            )

        token = (os.getenv("AEGRA_API_TOKEN") or "").strip()
        if not token:
            raise ConfigError(
                "AEGRA_API_TOKEN не задан: агент откажет в доступе всем запросам. "
                f"{hint} и положите значение в deploy/.env"
            )

        return Config(
            aegra_url=(os.getenv("AEGRA_URL") or "http://aegra:2026").rstrip("/"),
            aegra_token=token,
            session_secret=secret,
            admin_username=(os.getenv("ADMIN_USERNAME") or "").strip(),
            admin_password=os.getenv("ADMIN_PASSWORD") or "",
            request_timeout=float(os.getenv("GUI_REQUEST_TIMEOUT") or 300.0),
        )
