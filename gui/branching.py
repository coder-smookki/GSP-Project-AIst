"""Ветвление диалога: развилка от выбранного ответа и переключение веток.

Зачем. Разговор с базой знаний часто идёт вилкой: «а если работы не
огневые, а газоопасные?». Продолжать одну ленту значит затирать прошлый
разбор; начинать пустой диалог значит терять весь контекст до развилки.
Ветка сохраняет и то и другое: исходная лента остаётся нетронутой, а
продолжение идёт со своей историей.

Как устроено. Ветка это отдельный тред агента. Состояние в него не
копируется: Aegra отклоняет ``update_state`` для треда, за которым ещё не
закреплён граф. Вместо копирования первый вопрос ветки уходит вместе с
историей до точки развилки - редьюсер ``add_messages`` складывает её в
состояние нового треда, и дальше ветка живёт сама по себе.

Дерево веток хранится рядом с историей Chainlit, в той же базе.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from gui.db import get_pool

logger = logging.getLogger(__name__)

#: Ветка, с которой начинается любой диалог.
ROOT_LABEL = "Основная"

#: Роли, которыми граф отмечает служебные сообщения. Обрывать историю на
#: них нельзя: ответ инструмента без вызова ломает разбор у модели.
_SERVICE_TYPES = frozenset({"tool", "function"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS gui_branches (
    branch_id UUID PRIMARY KEY,
    chat_id TEXT NOT NULL,
    agent_thread_id TEXT NOT NULL,
    label TEXT NOT NULL,
    parent_branch_id UUID,
    fork_after INT NOT NULL DEFAULT 0,
    seeded BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    owner TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

INDEX = """
CREATE INDEX IF NOT EXISTS gui_branches_chat_idx ON gui_branches (chat_id)
"""


@dataclass(frozen=True, slots=True)
class Branch:
    """Ветка диалога.

    Attributes:
        branch_id: Идентификатор ветки.
        chat_id: Диалог Chainlit, которому принадлежит ветка.
        agent_thread_id: Тред агента, где живёт продолжение.
        label: Подпись для кнопки переключения.
        parent_branch_id: Ветка, от которой отделились; ``None`` у корня.
        fork_after: Сколько сообщений взято из родителя.
        seeded: История уже отправлена в тред агента.
        is_active: Ветка, в которую уходят новые вопросы.
    """

    branch_id: str
    chat_id: str
    agent_thread_id: str
    label: str
    parent_branch_id: str | None = None
    fork_after: int = 0
    seeded: bool = False
    is_active: bool = False
    created_at: datetime | None = None


def _is_service(message: dict) -> bool:
    """Отличает служебное сообщение инструмента от обычного.

    Args:
        message: Сообщение истории.

    Returns:
        bool: ``True`` для ответа инструмента.
    """
    return str(message.get("type") or message.get("role") or "") in _SERVICE_TYPES


def _has_call(messages: list[dict], call_id: str | None) -> bool:
    """Проверяет, есть ли в истории вызов инструмента с таким номером.

    Args:
        messages: Сообщения, которые остаются в срезе.
        call_id: Номер вызова из ответа инструмента.

    Returns:
        bool: ``True``, если вызов на месте.
    """
    if not call_id:
        return False
    return any(
        call.get("id") == call_id
        for message in messages
        for call in (message.get("tool_calls") or [])
    )


def prefix_messages(messages: list[dict], fork_after: int) -> list[dict]:
    """Отрезает историю по точке развилки.

    Обрыв делается не ровно по счётчику: если на границе оказался ответ
    инструмента, он утащит за собой вызов, которого в срезе уже нет, и
    модель получит рассогласованную историю. Такие хвосты отбрасываются.

    Args:
        messages: Сообщения треда агента по порядку.
        fork_after: Сколько первых сообщений оставить.

    Returns:
        list[dict]: История до развилки включительно.
    """
    cut = messages[: max(0, fork_after)]
    while cut:
        last = cut[-1]
        if _is_service(last):
            # Ответ инструмента без своего вызова модель разобрать не может.
            if _has_call(cut[:-1], last.get("tool_call_id")):
                break
        elif last.get("tool_calls"):
            # Вызов без ответа - та же рассогласованность, только наоборот.
            answered = {m.get("tool_call_id") for m in cut if _is_service(m)}
            if all(call.get("id") in answered for call in last["tool_calls"]):
                break
        else:
            break
        cut = cut[:-1]
    return cut


def next_label(existing: list[Branch]) -> str:
    """Подбирает подпись для новой ветки.

    Args:
        existing: Уже заведённые ветки диалога.

    Returns:
        str: Подпись вида «Ветка 2»; нумерация продолжает наибольшую, а не
        считает ветки, иначе после удаления подписи начали бы повторяться.
    """
    used = 1
    for branch in existing:
        head, _, tail = branch.label.partition(" ")
        if head == "Ветка" and tail.isdigit():
            used = max(used, int(tail))
    return f"Ветка {used + 1}"


def describe(branches: list[Branch]) -> str:
    """Собирает читаемое описание дерева веток.

    Args:
        branches: Ветки диалога.

    Returns:
        str: Строки вида «• Ветка 2 - от 4-го сообщения», активная отмечена.
    """
    lines = []
    for branch in branches:
        mark = "▸" if branch.is_active else "•"
        where = (
            f" - от {branch.fork_after}-го сообщения"
            if branch.parent_branch_id
            else " - с начала диалога"
        )
        lines.append(f"{mark} {branch.label}{where}")
    return "\n".join(lines)


async def ensure_schema() -> None:
    """Создаёт таблицу веток, если её ещё нет.

    Returns:
        None
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA)
        await conn.execute(INDEX)


async def list_branches(chat_id: str) -> list[Branch]:
    """Отдаёт ветки диалога в порядке появления.

    Args:
        chat_id: Диалог Chainlit.

    Returns:
        list[Branch]: Ветки диалога; пустой список, если развилок не было.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT branch_id, chat_id, agent_thread_id, label, parent_branch_id,"
            " fork_after, seeded, is_active, created_at"
            " FROM gui_branches WHERE chat_id = $1 ORDER BY created_at",
            chat_id,
        )
    return [
        Branch(
            branch_id=str(row["branch_id"]),
            chat_id=row["chat_id"],
            agent_thread_id=row["agent_thread_id"],
            label=row["label"],
            parent_branch_id=(
                str(row["parent_branch_id"]) if row["parent_branch_id"] else None
            ),
            fork_after=row["fork_after"],
            seeded=row["seeded"],
            is_active=row["is_active"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


async def get_branch(branch_id: str) -> Branch | None:
    """Находит ветку по идентификатору.

    Args:
        branch_id: Идентификатор ветки.

    Returns:
        Branch | None: Ветка либо ``None``, если её нет.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT branch_id, chat_id, agent_thread_id, label, parent_branch_id,"
            " fork_after, seeded, is_active, created_at"
            " FROM gui_branches WHERE branch_id = $1",
            uuid.UUID(branch_id),
        )
    if row is None:
        return None
    return Branch(
        branch_id=str(row["branch_id"]),
        chat_id=row["chat_id"],
        agent_thread_id=row["agent_thread_id"],
        label=row["label"],
        parent_branch_id=(
            str(row["parent_branch_id"]) if row["parent_branch_id"] else None
        ),
        fork_after=row["fork_after"],
        seeded=row["seeded"],
        is_active=row["is_active"],
        created_at=row["created_at"],
    )


async def save_branch(branch: Branch) -> None:
    """Записывает ветку.

    Args:
        branch: Ветка диалога.

    Returns:
        None
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO gui_branches (branch_id, chat_id, agent_thread_id, label,"
            " parent_branch_id, fork_after, seeded, is_active, owner)"
            " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)"
            " ON CONFLICT (branch_id) DO NOTHING",
            uuid.UUID(branch.branch_id),
            branch.chat_id,
            branch.agent_thread_id,
            branch.label,
            uuid.UUID(branch.parent_branch_id) if branch.parent_branch_id else None,
            branch.fork_after,
            branch.seeded,
            branch.is_active,
            branch.chat_id,
        )


async def mark_seeded(branch_id: str) -> None:
    """Отмечает, что история ветки уже ушла в тред агента.

    Args:
        branch_id: Идентификатор ветки.

    Returns:
        None
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE gui_branches SET seeded = TRUE WHERE branch_id = $1",
            uuid.UUID(branch_id),
        )


async def activate(chat_id: str, branch_id: str) -> None:
    """Делает ветку активной, снимая отметку с остальных.

    Args:
        chat_id: Диалог Chainlit.
        branch_id: Ветка, в которую уходят новые вопросы.

    Returns:
        None
    """
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "UPDATE gui_branches SET is_active = FALSE WHERE chat_id = $1",
            chat_id,
        )
        await conn.execute(
            "UPDATE gui_branches SET is_active = TRUE WHERE branch_id = $1",
            uuid.UUID(branch_id),
        )


async def active_branch(chat_id: str) -> Branch | None:
    """Возвращает активную ветку диалога.

    Args:
        chat_id: Диалог Chainlit.

    Returns:
        Branch | None: Активная ветка либо ``None``, если развилок не было.
    """
    for branch in await list_branches(chat_id):
        if branch.is_active:
            return branch
    return None


async def ensure_root(chat_id: str, agent_thread_id: str) -> Branch:
    """Заводит корневую ветку диалога, если её ещё нет.

    Корень нужен, чтобы после развилки было куда вернуться: без записи о
    нём исходная лента осталась бы без кнопки переключения.

    Args:
        chat_id: Диалог Chainlit.
        agent_thread_id: Тред агента исходной ленты.

    Returns:
        Branch: Корневая ветка.
    """
    await ensure_schema()
    for branch in await list_branches(chat_id):
        if branch.parent_branch_id is None:
            return branch
    root = Branch(
        branch_id=str(uuid.uuid4()),
        chat_id=chat_id,
        agent_thread_id=agent_thread_id,
        label=ROOT_LABEL,
        seeded=True,
        is_active=True,
    )
    await save_branch(root)
    await activate(chat_id, root.branch_id)
    return root


async def count_messages(client, agent_thread_id: str) -> int:
    """Считает сообщения в треде агента.

    Число нужно как точка развилки: оно попадает в кнопку под ответом и
    потом задаёт длину среза истории.

    Args:
        client: Клиент Aegra.
        agent_thread_id: Тред агента.

    Returns:
        int: Сколько сообщений в состоянии треда; ``0``, если состояние
        недоступно - тогда развилка просто не предлагается.
    """
    try:
        state = await client.threads.get_state(agent_thread_id)
    except Exception as exc:
        logger.warning("состояние треда %s не прочитано: %s", agent_thread_id, exc)
        return 0
    return len((state.get("values") or {}).get("messages") or [])


async def snapshot(client, agent_thread_id: str, fork_after: int) -> list[dict]:
    """Забирает историю треда агента до точки развилки.

    Args:
        client: Клиент Aegra.
        agent_thread_id: Тред агента, от которого ответвляются.
        fork_after: Сколько сообщений взять.

    Returns:
        list[dict]: Сообщения до развилки; пустой список, если состояние
        недоступно.
    """
    try:
        state = await client.threads.get_state(agent_thread_id)
    except Exception as exc:
        logger.warning("история треда %s не прочитана: %s", agent_thread_id, exc)
        return []
    messages = (state.get("values") or {}).get("messages") or []
    return prefix_messages(messages, fork_after)


async def fork(client, chat_id: str, parent: Branch, fork_after: int) -> Branch:
    """Создаёт ветку от указанного места родительской ленты.

    Args:
        client: Клиент Aegra.
        chat_id: Диалог Chainlit.
        parent: Ветка, от которой ответвляемся.
        fork_after: Сколько сообщений родителя войдёт в ветку.

    Returns:
        Branch: Новая ветка; она сразу становится активной.
    """
    await ensure_schema()
    existing = await list_branches(chat_id)
    thread_id = str(uuid.uuid4())
    await client.threads.create(thread_id=thread_id, if_exists="do_nothing")
    branch = Branch(
        branch_id=str(uuid.uuid4()),
        chat_id=chat_id,
        agent_thread_id=thread_id,
        label=next_label(existing),
        parent_branch_id=parent.branch_id,
        fork_after=fork_after,
        seeded=False,
        is_active=True,
    )
    await save_branch(branch)
    await activate(chat_id, branch.branch_id)
    logger.info(
        "диалог %s: ветка %s от %s после %d сообщений",
        chat_id,
        branch.label,
        parent.label,
        fork_after,
    )
    return branch
