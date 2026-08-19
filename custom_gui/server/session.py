"""Сессия пользователя: подписанная cookie без хранилища на сервере.

Внутри лежит только имя пользователя, его роль и срок годности. Подпись -
HMAC-SHA256 тем же ключом, что и раньше использовал Chainlit, поэтому на стенде
не появляется ещё одного секрета.

Почему не JWT. Нужен ровно один сценарий - «сервер сам себе выдал и сам же
проверил», - а у JWT под это идёт разбор заголовка с выбором алгоритма, и
именно там живёт классическая ошибка с ``alg: none``. Здесь алгоритм один и
задан кодом.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Имя cookie сессии.
COOKIE_NAME = "aist_session"


class InvalidSession(ValueError):
    """Cookie отсутствует, испорчена, подделана или просрочена."""


@dataclass(frozen=True, slots=True)
class Session:
    """Содержимое сессии."""

    username: str
    role: str
    expires_at: int

    @property
    def is_admin(self) -> bool:
        """Сообщает, администратор ли это.

        Returns:
            bool: ``True`` для роли ``admin``.
        """
        return self.role == "admin"


def _b64encode(raw: bytes) -> str:
    """Кодирует байты в url-безопасный base64 без выравнивания.

    Args:
        raw: Исходные байты.

    Returns:
        str: Строка base64url без символов ``=``.
    """
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    """Декодирует url-безопасный base64 без выравнивания.

    Args:
        value: Строка base64url.

    Returns:
        bytes: Исходные байты.

    Raises:
        InvalidSession: Если строка не декодируется.
    """
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    # Любая порча даёт один и тот же ответ: подробности подсказали бы,
    # какая именно часть cookie не сошлась.
    except Exception as exc:
        raise InvalidSession("cookie сессии испорчена") from exc


def issue(username: str, role: str, secret: str, ttl_seconds: int) -> str:
    """Выпускает подписанную сессию.

    Args:
        username: Имя пользователя.
        role: Роль (``user`` или ``admin``).
        secret: Ключ подписи.
        ttl_seconds: Сколько сессия действительна, с.

    Returns:
        str: Значение cookie вида ``<полезная нагрузка>.<подпись>``.
    """
    payload = {
        "u": username,
        "r": role,
        "e": int(time.time()) + ttl_seconds,
    }
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).digest()
    return f"{body}.{_b64encode(signature)}"


def verify(cookie: str | None, secret: str) -> Session:
    """Проверяет подпись и срок сессии.

    Args:
        cookie: Значение cookie; ``None`` - её нет.
        secret: Ключ подписи.

    Returns:
        Session: Разобранная сессия.

    Raises:
        InvalidSession: Если cookie нет, подпись не сходится, срок вышел или
            содержимое не разбирается.
    """
    if not cookie or "." not in cookie:
        raise InvalidSession("сессии нет")

    body, _, signature = cookie.partition(".")
    # Значение приходит из браузера, то есть от кого угодно. Кодируем в UTF-8,
    # а не в ASCII: кириллица в подделанной cookie роняла проверку исключением,
    # и посторонний получал ошибку сервера вместо честного отказа.
    expected = hmac.new(
        secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).digest()
    # Сравнение постоянного времени: иначе подпись подбирается побайтово по
    # времени ответа.
    if not hmac.compare_digest(_b64decode(signature), expected):
        raise InvalidSession("подпись сессии не сходится")

    try:
        payload = json.loads(_b64decode(body))
        username = str(payload["u"])
        role = str(payload["r"])
        expires_at = int(payload["e"])
    # Содержимое подписано нами же: если оно не разбирается, доверять
    # ему нельзя независимо от причины.
    except Exception as exc:
        raise InvalidSession("содержимое сессии не разбирается") from exc

    if expires_at <= int(time.time()):
        raise InvalidSession("срок сессии истёк")

    return Session(username=username, role=role, expires_at=expires_at)
