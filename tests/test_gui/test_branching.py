"""Ветвление диалога: срез истории, подписи веток и создание развилки."""

from __future__ import annotations

import pytest

from gui import branching
from gui.branching import Branch
from gui.contract import build_input


def _branch(label: str, parent: str | None = None, **kw) -> Branch:
    """Собирает ветку для проверок.

    Args:
        label: Подпись ветки.
        parent: Идентификатор родительской ветки.
        **kw: Остальные поля ветки.

    Returns:
        Branch: Ветка.
    """
    return Branch(
        branch_id=kw.pop("branch_id", label),
        chat_id=kw.pop("chat_id", "chat"),
        agent_thread_id=kw.pop("agent_thread_id", f"thread-{label}"),
        label=label,
        parent_branch_id=parent,
        **kw,
    )


def test_history_is_cut_at_the_fork_point() -> None:
    """В ветку уходит история ровно до выбранного места."""
    messages = [{"type": "human", "content": str(n)} for n in range(6)]

    assert branching.prefix_messages(messages, 4) == messages[:4]


def test_paired_call_and_reply_survive_the_cut() -> None:
    """Вызов инструмента вместе со своим ответом остаётся в срезе."""
    messages = [
        {"type": "human", "content": "вопрос"},
        {"type": "ai", "content": "", "tool_calls": [{"id": "1"}]},
        {"type": "tool", "content": "результат", "tool_call_id": "1"},
    ]

    assert branching.prefix_messages(messages, 3)[-1]["type"] == "tool"


def test_call_without_its_reply_is_dropped() -> None:
    """Вызов инструмента без ответа отбрасывается.

    Рассогласованная история ломает следующий запрос ещё до поиска.
    """
    messages = [
        {"type": "human", "content": "вопрос"},
        {"type": "ai", "content": "", "tool_calls": [{"id": "1"}]},
        {"type": "tool", "content": "результат", "tool_call_id": "1"},
    ]

    assert len(branching.prefix_messages(messages, 2)) == 1


def test_reply_without_its_call_is_dropped() -> None:
    """Ответ инструмента без своего вызова отбрасывается."""
    messages = [
        {"type": "human", "content": "вопрос"},
        {"type": "tool", "content": "результат", "tool_call_id": "чужой"},
    ]

    assert branching.prefix_messages(messages, 2) == messages[:1]


def test_fork_point_beyond_history_is_harmless() -> None:
    """Точка развилки за пределами истории не ломает срез."""
    messages = [{"type": "human", "content": "вопрос"}]

    assert branching.prefix_messages(messages, 99) == messages
    assert branching.prefix_messages(messages, 0) == []


def test_labels_do_not_repeat_after_a_branch_is_gone() -> None:
    """Нумерация продолжает наибольшую, а не считает ветки.

    Если считать количество, то после удаления средней ветки новая
    получила бы уже занятую подпись.
    """
    existing = [_branch("Основная"), _branch("Ветка 3", parent="Основная")]

    assert branching.next_label(existing) == "Ветка 4"


def test_first_label_after_the_root_is_the_second() -> None:
    """Первая развилка называется «Ветка 2»: корень уже первый."""
    assert branching.next_label([_branch("Основная")]) == "Ветка 2"


def test_tree_marks_the_active_branch() -> None:
    """В описании дерева видно, где идёт разговор."""
    text = branching.describe(
        [
            _branch("Основная"),
            _branch("Ветка 2", parent="Основная", fork_after=4, is_active=True),
        ]
    )

    assert "▸ Ветка 2 - от 4-го сообщения" in text
    assert "• Основная - с начала диалога" in text


def test_history_goes_before_the_question() -> None:
    """История ветки уходит перед вопросом, а не после."""
    payload = build_input(
        "новый вопрос", history=[{"type": "human", "content": "прежний"}]
    )

    assert [m["content"] for m in payload["messages"]] == ["прежний", "новый вопрос"]


def test_without_history_the_payload_is_unchanged() -> None:
    """Обычный вопрос собирается ровно как раньше."""
    assert build_input("вопрос") == {
        "messages": [{"type": "human", "content": "вопрос"}]
    }


class _Threads:
    """Треды Aegra, достаточные для проверки развилки."""

    def __init__(self, state: dict | None = None) -> None:
        """Готовит фальшивые треды.

        Args:
            state: Состояние, которое вернёт ``get_state``.
        """
        self.state = state or {}
        self.created: list[str] = []

    async def create(self, thread_id: str, if_exists: str = "raise") -> dict:
        """Заводит тред.

        Args:
            thread_id: Идентификатор треда.
            if_exists: Поведение при повторе.

        Returns:
            dict: Созданный тред.
        """
        self.created.append(thread_id)
        return {"thread_id": thread_id}

    async def get_state(self, thread_id: str) -> dict:
        """Возвращает состояние треда.

        Args:
            thread_id: Идентификатор треда.

        Returns:
            dict: Состояние.
        """
        return self.state


class _Client:
    """Клиент Aegra с одними тредами."""

    def __init__(self, state: dict | None = None) -> None:
        """Готовит клиента.

        Args:
            state: Состояние треда.
        """
        self.threads = _Threads(state)


@pytest.fixture
def storage(monkeypatch):
    """Подменяет хранилище веток словарём в памяти.

    Args:
        monkeypatch: Штатная фикстура подмены.

    Returns:
        dict: Записанные ветки по идентификатору.
    """
    saved: dict[str, Branch] = {}
    active: dict[str, str] = {}

    async def _ensure_schema() -> None:
        return None

    async def _list(chat_id: str) -> list[Branch]:
        return [b for b in saved.values() if b.chat_id == chat_id]

    async def _save(branch: Branch) -> None:
        saved[branch.branch_id] = branch

    async def _activate(chat_id: str, branch_id: str) -> None:
        active[chat_id] = branch_id

    monkeypatch.setattr(branching, "ensure_schema", _ensure_schema)
    monkeypatch.setattr(branching, "list_branches", _list)
    monkeypatch.setattr(branching, "save_branch", _save)
    monkeypatch.setattr(branching, "activate", _activate)
    return saved


@pytest.mark.asyncio
async def test_fork_creates_its_own_agent_thread(storage) -> None:
    """Развилка заводит отдельный тред агента.

    Иначе продолжение писалось бы в исходное состояние и затирало прошлый
    разбор - ровно то, ради чего ветка и заводится.
    """
    client = _Client()
    parent = _branch("Основная", branch_id="root")
    await branching.save_branch(parent)

    branch = await branching.fork(client, "chat", parent, 4)

    assert client.threads.created == [branch.agent_thread_id]
    assert branch.agent_thread_id != parent.agent_thread_id
    assert branch.parent_branch_id == "root"
    assert branch.fork_after == 4
    assert branch.is_active


@pytest.mark.asyncio
async def test_new_branch_is_not_seeded(storage) -> None:
    """Новая ветка помечена как незасеянная.

    По этой отметке первый вопрос уходит вместе с историей до развилки.
    """
    branch = await branching.fork(_Client(), "chat", _branch("Основная"), 2)

    assert not branch.seeded


@pytest.mark.asyncio
async def test_snapshot_returns_history_up_to_the_fork() -> None:
    """Снимок отдаёт историю родителя до точки развилки."""
    messages = [{"type": "human", "content": str(n)} for n in range(5)]
    client = _Client({"values": {"messages": messages}})

    assert await branching.snapshot(client, "thread", 3) == messages[:3]


@pytest.mark.asyncio
async def test_unreachable_state_does_not_break_the_answer() -> None:
    """Недоступное состояние не роняет ответ, а гасит развилку.

    Кнопка развилки рисуется после каждого ответа; сбой чтения состояния
    не должен стоить пользователю самого ответа.
    """

    class _Broken(_Client):
        def __init__(self) -> None:
            super().__init__()

            async def _boom(thread_id: str) -> dict:
                raise RuntimeError("Aegra недоступна")

            self.threads.get_state = _boom

    assert await branching.count_messages(_Broken(), "thread") == 0
    assert await branching.snapshot(_Broken(), "thread", 3) == []


@pytest.mark.asyncio
async def test_message_count_is_the_fork_point() -> None:
    """Точка развилки берётся из числа сообщений треда."""
    client = _Client({"values": {"messages": [{}, {}, {}]}})

    assert await branching.count_messages(client, "thread") == 3
