"""Хранилище диалогов собственного интерфейса.

Свои таблицы, а не таблицы слоя данных Chainlit: их формат принадлежит
Chainlit, от которого мы уходим, и подстраиваться под чужую схему значило бы
тащить её ограничения дальше.

Идентификатор диалога он же идентификатор треда агента - соответствий нигде не
хранится, и после перезагрузки страницы разговор продолжается с того же места.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import asyncpg

from gui.db import get_pool, init_db

logger = logging.getLogger(__name__)

#: Схема хранилища. Диалог принадлежит пользователю, сообщение - диалогу.
#: Каскад по владельцу: удаление пользователя не должно оставлять сирот.
SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS aist_conversations (
        id UUID PRIMARY KEY,
        owner VARCHAR(255) NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS aist_messages (
        id BIGSERIAL PRIMARY KEY,
        conversation_id UUID NOT NULL
            REFERENCES aist_conversations(id) ON DELETE CASCADE,
        role VARCHAR(16) NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # Список диалогов открывается на каждом входе и сортируется по свежести.
    """
    CREATE INDEX IF NOT EXISTS aist_conversations_owner_idx
        ON aist_conversations (owner, updated_at DESC)
    """,
    # Сообщения всегда читаются целиком по одному диалогу и по порядку.
    """
    CREATE INDEX IF NOT EXISTS aist_messages_conversation_idx
        ON aist_messages (conversation_id, id)
    """,
)

_ready = False


async def setup() -> None:
    """Создаёт таблицы хранилища, если их ещё нет."""
    global _ready
    if _ready:
        return
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as connection:
        for statement in SCHEMA:
            await connection.execute(statement)
    _ready = True
    logger.info("хранилище диалогов готово")


async def list_conversations(owner: str) -> list[dict[str, Any]]:
    """Возвращает диалоги пользователя от свежих к старым.

    Args:
        owner: Имя владельца.

    Returns:
        list[dict[str, Any]]: Записи с идентификатором, названием, временем
        обновления и числом сообщений.
    """
    pool = await get_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT c.id, c.title, c.updated_at,
                   (SELECT count(*) FROM aist_messages m
                     WHERE m.conversation_id = c.id) AS messages
            FROM aist_conversations c
            WHERE c.owner = $1
            ORDER BY c.updated_at DESC
            """,
            owner,
        )
    return [
        {
            "id": str(row["id"]),
            "title": row["title"],
            "updated_at": row["updated_at"].isoformat(),
            "messages": int(row["messages"]),
        }
        for row in rows
    ]


async def create_conversation(owner: str, title: str = "") -> dict[str, Any]:
    """Заводит новый диалог.

    Args:
        owner: Имя владельца.
        title: Название; пусто - подставится по первому вопросу.

    Returns:
        dict[str, Any]: Созданная запись.
    """
    pool = await get_pool()
    conversation_id = uuid.uuid4()
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            INSERT INTO aist_conversations (id, owner, title)
            VALUES ($1, $2, $3)
            RETURNING id, title, updated_at
            """,
            conversation_id,
            owner,
            title,
        )
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "updated_at": row["updated_at"].isoformat(),
        "messages": 0,
    }


async def owns(owner: str, conversation_id: str) -> bool:
    """Проверяет, принадлежит ли диалог пользователю.

    Вызывается перед каждым обращением к диалогу: идентификатор приходит из
    браузера, и без проверки чужой диалог открывался бы по прямой ссылке.

    Args:
        owner: Имя пользователя.
        conversation_id: Идентификатор диалога.

    Returns:
        bool: ``True``, если диалог существует и принадлежит пользователю.
    """
    try:
        key = uuid.UUID(conversation_id)
    except (ValueError, AttributeError, TypeError):
        return False
    pool = await get_pool()
    async with pool.acquire() as connection:
        found = await connection.fetchval(
            "SELECT 1 FROM aist_conversations WHERE id = $1 AND owner = $2",
            key,
            owner,
        )
    return found is not None


async def load_messages(conversation_id: str) -> list[dict[str, Any]]:
    """Читает сообщения диалога по порядку.

    Args:
        conversation_id: Идентификатор диалога.

    Returns:
        list[dict[str, Any]]: Сообщения с ролью, текстом и полезной нагрузкой
        (источники, подсказки, этапы).
    """
    pool = await get_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT role, content, payload, created_at
            FROM aist_messages
            WHERE conversation_id = $1
            ORDER BY id
            """,
            uuid.UUID(conversation_id),
        )
    return [
        {
            "role": row["role"],
            "content": row["content"],
            **json.loads(row["payload"]),
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


async def append_message(
    conversation_id: str,
    role: str,
    content: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Дописывает сообщение и освежает диалог в списке.

    Args:
        conversation_id: Идентификатор диалога.
        role: ``user`` или ``assistant``.
        content: Текст сообщения.
        payload: Источники, подсказки и этапы; для вопроса пусто.
    """
    pool = await get_pool()
    key = uuid.UUID(conversation_id)
    async with pool.acquire() as connection, connection.transaction():
        await connection.execute(
            """
                INSERT INTO aist_messages (conversation_id, role, content, payload)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
            key,
            role,
            content,
            json.dumps(payload or {}, ensure_ascii=False),
        )
        await connection.execute(
            "UPDATE aist_conversations SET updated_at = now() WHERE id = $1",
            key,
        )


async def set_title_if_empty(conversation_id: str, title: str) -> str:
    """Задаёт название диалога, если его ещё нет.

    Название берётся из первого вопроса: обращаться ради заголовка к модели
    значит гонять полный цикл поиска и писать лишнюю запись в память.

    Args:
        conversation_id: Идентификатор диалога.
        title: Предлагаемое название.

    Returns:
        str: Итоговое название диалога.
    """
    clean = " ".join(title.split())[:120]
    pool = await get_pool()
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            UPDATE aist_conversations
            SET title = CASE WHEN title = '' THEN $2 ELSE title END
            WHERE id = $1
            RETURNING title
            """,
            uuid.UUID(conversation_id),
            clean,
        )
    return row["title"] if row else clean


async def rename(conversation_id: str, title: str) -> str:
    """Переименовывает диалог.

    Args:
        conversation_id: Идентификатор диалога.
        title: Новое название.

    Returns:
        str: Сохранённое название.
    """
    clean = " ".join(title.split())[:120]
    pool = await get_pool()
    async with pool.acquire() as connection:
        await connection.execute(
            "UPDATE aist_conversations SET title = $2 WHERE id = $1",
            uuid.UUID(conversation_id),
            clean,
        )
    return clean


async def delete(conversation_id: str) -> None:
    """Удаляет диалог вместе с сообщениями.

    Args:
        conversation_id: Идентификатор диалога.
    """
    pool = await get_pool()
    async with pool.acquire() as connection:
        await connection.execute(
            "DELETE FROM aist_conversations WHERE id = $1",
            uuid.UUID(conversation_id),
        )


async def healthy() -> bool:
    """Проверяет, отвечает ли база.

    Returns:
        bool: ``True``, если простой запрос выполняется.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
    except (asyncpg.PostgresError, OSError) as exc:
        logger.error("база недоступна: %s", exc)
        return False
    return True
