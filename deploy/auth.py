"""Аутентификация Aegra: доверенный посредник плюс имя пользователя.

Зачем. Без этого файла Aegra работает в режиме noop: она сама пишет в лог, что
все запросы получают личность ``anonymous`` и данные между пользователями не
разделены. Практически это означало три вещи. Личность бралась из тела запроса
(``config.configurable.user_id``), потому что сервер подставляет свою только
через ``setdefault`` - то есть любой клиент мог назваться кем угодно и прочитать
чужую долгосрочную память. ``DELETE /threads/{id}`` сверяет владельца с той же
``anonymous`` и потому разрешал удалить чужой диалог. И весь порт целиком был
открыт любому, кто до него дотянется.

Схема. Браузер в Aegra не ходит - ходит только доверенный посредник (сейчас это
Chainlit, дальше будет бэкенд своего интерфейса). Посредник уже проверил
человека по логину и паролю, поэтому он:

* доказывает, что он это он, заголовком ``Authorization: Bearer <токен>``;
* сообщает, за кого ручается, заголовком ``X-Aist-User``.

Без верного токена доступа нет вовсе, а имя пользователя без токена ничего не
значит. Личность из ``X-Aist-User`` дальше подставляется сервером в
``configurable.user_id``, и именно она задаёт пространство памяти
``("memories", user_id)`` в графе.

Закрыто по умолчанию: если ``AEGRA_API_TOKEN`` не задан, сервис не пускает
никого. Так тише не станет - в логе будет прямая инструкция, что сделать.
Молчаливое возвращение к общему ``anonymous`` было бы худшим из вариантов.

Проверки ``/health`` и ``/info`` остаются открытыми: аутентификация в Aegra
подключается зависимостью на конкретных маршрутах, а не общей прослойкой.
"""

from __future__ import annotations

import hmac
import logging
import os
import re

from langgraph_sdk import Auth

logger = logging.getLogger(__name__)

auth = Auth()

#: Допустимое имя пользователя. Оно становится частью пространства имён
#: хранилища памяти, поэтому набор символов задан жёстко: разделители путей и
#: пробелы внутрь не пропускаются.
_USER_PATTERN = re.compile(r"^[A-Za-z0-9._@-]{1,64}$")

#: Заголовок с именем пользователя, за которого ручается посредник.
_USER_HEADER = "x-aist-user"

_MISSING_TOKEN_HINT = (
    "AEGRA_API_TOKEN не задан - Aegra никого не пускает. "
    "Сгенерируйте токен командой `python -c \"import secrets;"
    "print(secrets.token_urlsafe(32))\"`, положите его в deploy/.env "
    "как AEGRA_API_TOKEN и перезапустите стек."
)


def _expected_token() -> str:
    """Возвращает ожидаемый токен доступа из окружения.

    Returns:
        str: Значение ``AEGRA_API_TOKEN`` без окружающих пробелов; пустая
        строка означает, что токен не настроен.
    """
    return (os.environ.get("AEGRA_API_TOKEN") or "").strip()


def _bearer(headers: dict[str, str]) -> str:
    """Достаёт токен из заголовка ``Authorization``.

    Args:
        headers: Заголовки запроса; имена уже приведены к нижнему регистру.

    Returns:
        str: Токен без схемы ``Bearer``; пустая строка, если заголовка нет
        или схема другая.
    """
    raw = (headers.get("authorization") or "").strip()
    if not raw:
        return ""
    scheme, _, value = raw.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return value.strip()


@auth.authenticate
async def authenticate(headers: dict[str, str]) -> Auth.types.MinimalUserDict:
    """Проверяет доверенного посредника и определяет пользователя.

    Args:
        headers: Заголовки запроса в виде словаря.

    Returns:
        Auth.types.MinimalUserDict: Личность пользователя для Aegra.

    Raises:
        Auth.exceptions.HTTPException: 503, если токен не настроен на сервере;
            401, если токен не совпал или имя пользователя не передано либо
            не прошло проверку.
    """
    expected = _expected_token()
    if not expected:
        logger.error(_MISSING_TOKEN_HINT)
        raise Auth.exceptions.HTTPException(
            status_code=503, detail="сервис не настроен: нет токена доступа"
        )

    supplied = _bearer(headers)
    # Сравнение постоянного времени: обычное сравнение строк выдаёт длину
    # совпавшего префикса по времени ответа. Сравниваются именно байты:
    # ``compare_digest`` на строках отказывается работать с не-ASCII, и
    # кириллица в токене превращала бы честный отказ в ошибку сервера.
    if not supplied or not hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="неверный токен доступа"
        )

    user = (headers.get(_USER_HEADER) or "").strip()
    if not user:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"не передан заголовок {_USER_HEADER}"
        )
    if not _USER_PATTERN.match(user):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="недопустимое имя пользователя"
        )

    return {"identity": user, "display_name": user, "is_authenticated": True}


@auth.on
async def scope_to_owner(ctx: Auth.types.AuthContext, value: dict) -> dict | None:
    """Привязывает создаваемые ресурсы к владельцу и фильтрует чужие.

    Помечает создаваемый ресурс владельцем и тем же условием ограничивает
    выборку, поэтому чужие треды не видны в списке и недоступны по прямому
    идентификатору.

    Ассистенты — глобальный ресурс (создаются системой с ``user_id="system"``,
    см. ``aegra_api.services.langgraph_service._ensure_default_assistants``).
    Для них фильтр владельца не применяется: иначе системный ассистент
    ``agent`` с ``metadata={"created_by": "system"}`` не проходит проверку
    ``metadata @> {"owner": "<user>"}`` и ``/assistants/search`` возвращает
    пустой список — интерфейс показывает «не найден ни один ассистент».

    Args:
        ctx: Контекст запроса; из него берётся личность пользователя.
        value: Тело операции; в него дописываются метаданные владельца.

    Returns:
        dict | None: Фильтр по владельцу для выборок Aegra или ``None``,
            если ресурс не требует изоляции по владельцу.
    """
    # Системные ассистенты должны быть видны всем аутентифицированным
    # пользователям. Фильтр по владельцу здесь ломает их видимость.
    if ctx.resource == "assistants":
        return None
    owner = ctx.user.identity
    if isinstance(value, dict):
        metadata = value.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("owner", owner)
    return {"owner": owner}
