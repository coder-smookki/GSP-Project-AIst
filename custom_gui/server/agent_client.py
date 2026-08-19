"""Разговор с агентом от имени вошедшего человека.

Браузер к Aegra не ходит. Сюда приходит вопрос, отсюда уходит запрос с токеном
доверенного посредника и заголовком ``X-Aist-User``; личность подставляет
сервер, поэтому подменить её клиент не может (см. ``deploy/auth.py``).

Наружу отдаётся уже разобранный поток событий, а не сырые куски LangGraph:
разбор конверта ``aist`` живёт в одном месте (``gui/contract.py``), он покрыт
тестами, и по сети уходит на порядок меньше данных.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from gui.contract import build_input, decode_aist_envelope, parse_stage

logger = logging.getLogger(__name__)

#: Узлы, по завершении которых можно пускать токены в пузырь ответа.
#:
#: ``updates`` в LangGraph приходит по ЗАВЕРШЕНИИ узла, поэтому открытие на
#: ``plan`` включает стриминг и для режима беседы, где ``finalize`` не будет
#: вовсе. Держится на том, что ``subgraphs=True`` не передаётся: иначе в поток
#: полились бы служебные токены веточных агентов.
STREAM_GATE_NODES = frozenset({"plan", "finalize"})


class AgentUnavailable(RuntimeError):
    """Агент не отвечает или отказал в доступе."""


@dataclass(slots=True)
class TurnResult:
    """Итог одного хода, который надо сохранить в диалоге."""

    text: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    failed: bool = False


class AgentClient:
    """Клиент агента с одним переиспользуемым пулом соединений."""

    def __init__(self, base_url: str, token: str, timeout: float = 300.0) -> None:
        """Готовит клиента.

        Args:
            base_url: Адрес Aegra без завершающего слэша.
            token: Токен доверенного посредника.
            timeout: Предел ожидания ответа, с.
        """
        self._base_url = base_url
        self._token = token
        self._assistant_id: str | None = None
        # Один клиент на процесс: создавать соединение на каждый запрос значит
        # платить рукопожатием за каждый вопрос и течь дескрипторами.
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {token}"},
        )

    async def aclose(self) -> None:
        """Закрывает пул соединений."""
        await self._http.aclose()

    def _headers(self, username: str) -> dict[str, str]:
        """Собирает заголовки запроса от имени пользователя.

        Args:
            username: Имя вошедшего пользователя.

        Returns:
            dict[str, str]: Заголовок личности; токен уже в клиенте.
        """
        return {"X-Aist-User": username}

    async def healthy(self) -> bool:
        """Проверяет, отвечает ли агент.

        Returns:
            bool: ``True``, если ``/info`` отвечает успехом.
        """
        try:
            response = await self._http.get("/info", timeout=5.0)
        except httpx.HTTPError as exc:
            logger.error("агент недоступен: %s", exc)
            return False
        return response.is_success

    async def get_json(self, username: str, path: str) -> httpx.Response:
        """Выполняет GET к агенту от имени пользователя.

        Args:
            username: Имя вошедшего пользователя.
            path: Путь внутри агента.

        Returns:
            httpx.Response: Ответ как есть - код разбирает вызывающая сторона.

        Raises:
            AgentUnavailable: Если связи нет.
        """
        try:
            return await self._http.get(path, headers=self._headers(username))
        except httpx.HTTPError as exc:
            raise AgentUnavailable(f"каталог недоступен: {exc}") from exc

    async def assistant_id(self, username: str) -> str:
        """Возвращает идентификатор ассистента, запрашивая его один раз.

        Args:
            username: Имя пользователя для заголовка личности.

        Returns:
            str: Идентификатор ассистента.

        Raises:
            AgentUnavailable: Если ассистентов нет или сервис не отвечает.
        """
        if self._assistant_id:
            return self._assistant_id
        try:
            response = await self._http.post(
                "/assistants/search",
                json={"limit": 10, "offset": 0},
                headers=self._headers(username),
            )
            response.raise_for_status()
            assistants = response.json()
        except httpx.HTTPError as exc:
            raise AgentUnavailable(f"агент не отвечает: {exc}") from exc
        if not assistants:
            raise AgentUnavailable("у агента не зарегистрировано ни одного ассистента")
        self._assistant_id = str(assistants[0]["assistant_id"])
        return self._assistant_id

    async def ensure_thread(self, username: str, thread_id: str) -> None:
        """Заводит тред агента под диалог, если его ещё нет.

        Идентификатор треда совпадает с идентификатором диалога, поэтому
        соответствий хранить не нужно, а вызов идемпотентен.

        Args:
            username: Имя пользователя.
            thread_id: Идентификатор диалога.

        Raises:
            AgentUnavailable: Если сервис не отвечает.
        """
        try:
            response = await self._http.post(
                "/threads",
                json={"thread_id": thread_id, "if_exists": "do_nothing"},
                headers=self._headers(username),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AgentUnavailable(f"не удалось открыть диалог: {exc}") from exc

    async def delete_thread(self, username: str, thread_id: str) -> None:
        """Удаляет тред агента. Отсутствие треда ошибкой не считается.

        Args:
            username: Имя пользователя.
            thread_id: Идентификатор диалога.
        """
        try:
            await self._http.delete(
                f"/threads/{thread_id}", headers=self._headers(username)
            )
        except httpx.HTTPError as exc:
            # Диалог из своего хранилища уже удалён, и повторить не выйдет.
            # Осиротевший тред безвреден: без записи о нём до него не дойти.
            logger.warning("тред %s не удалён у агента: %s", thread_id, exc)

    async def ask(
        self,
        username: str,
        thread_id: str,
        question: str,
        *,
        areas: list[str] | None = None,
        docs: list[str] | None = None,
        sections: list[str] | None = None,
        always_search: bool = False,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Задаёт вопрос и отдаёт разобранный поток событий.

        Args:
            username: Имя вошедшего пользователя.
            thread_id: Идентификатор диалога.
            question: Текст вопроса.
            areas: Выбранные области знаний.
            docs: Выбранные документы.
            sections: Выбранные типы разделов.
            always_search: Искать в базе даже для простых реплик.

        Yields:
            tuple[str, dict[str, Any]]: Пары «вид события, полезная нагрузка».
            Виды: ``stage``, ``token``, ``sources``, ``followups``, ``error``.

        Raises:
            AgentUnavailable: Если агент не отвечает или отказал в доступе.
        """
        assistant = await self.assistant_id(username)
        payload = build_input(
            question,
            sections=sections,
            areas=areas,
            always_answer_with_db=always_search,
        )
        if docs:
            payload["docs"] = docs

        body = {
            "assistant_id": assistant,
            "input": payload,
            "stream_mode": ["messages", "updates", "custom"],
            # subgraphs НЕ включаем: иначе токены веточных агентов попадут в
            # пузырь ответа, потому что гейт стриминга открывается раньше них.
            "stream_subgraphs": False,
        }

        streaming = False
        # Накопленный текст по идентификатору сообщения. Поток шлёт не приросты,
        # а весь текст целиком на каждом шаге («Пр» -> «Прив» -> «Привет»),
        # поэтому наружу уходит разница - иначе ответ задваивался бы.
        partials: dict[str, str] = {}
        try:
            async with self._http.stream(
                "POST",
                f"/threads/{thread_id}/runs/stream",
                json=body,
                headers={**self._headers(username), "Accept": "text/event-stream"},
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise AgentUnavailable(
                        f"агент отказал ({response.status_code}): {detail}"
                    )
                async for event, data in _iter_sse(response):
                    if event == "messages/partial":
                        if not streaming:
                            continue
                        for delta in _deltas(data, partials):
                            yield "token", {"text": delta}
                        continue
                    for kind, item in _translate(event, data):
                        if kind == "gate":
                            streaming = True
                            continue
                        yield kind, item
        except httpx.HTTPError as exc:
            raise AgentUnavailable(f"связь с агентом прервалась: {exc}") from exc


async def _iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str, Any]]:
    """Разбирает поток server-sent events в пары «событие, данные».

    Args:
        response: Открытый потоковый ответ.

    Yields:
        tuple[str, Any]: Имя события и разобранный JSON.
    """
    event = "message"
    async for line in response.aiter_lines():
        if not line:
            event = "message"
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.lstrip()
        if field == "event":
            event = value
        elif field == "data":
            try:
                yield event, json.loads(value)
            except json.JSONDecodeError:
                logger.debug("пропущено событие с неразбираемыми данными")


def _translate(event: str, data: Any) -> list[tuple[str, dict[str, Any]]]:
    """Переводит событие агента в события интерфейса.

    Args:
        event: Имя события потока LangGraph.
        data: Полезная нагрузка события.

    Returns:
        list[tuple[str, dict[str, Any]]]: Пары «вид, нагрузка» для браузера.
    """
    if event == "custom" and isinstance(data, dict):
        envelope = decode_aist_envelope(data)
        if envelope is None:
            return []
        kind = envelope.get("kind")
        payload = envelope.get("payload") or {}
        if kind == "trace":
            stage = parse_stage(payload)
            if stage is None:
                return []
            return [
                (
                    "stage",
                    {
                        "key": stage.key,
                        "title": stage.title,
                        "running": stage.running,
                        "failed": stage.failed,
                        "detail": stage.detail,
                    },
                )
            ]
        if kind == "sources":
            return [("sources", {"items": list(payload.get("items") or [])})]
        if kind == "followups":
            return [("followups", {"items": list(payload.get("items") or [])})]
        return []

    if event == "updates" and isinstance(data, dict):
        if STREAM_GATE_NODES & set(data):
            return [("gate", {})]
        return []

    return []


def _deltas(data: Any, partials: dict[str, str]) -> list[str]:
    """Вычисляет прирост текста для каждого сообщения потока.

    Args:
        data: Полезная нагрузка события ``messages/partial`` - список сообщений.
        partials: Накопленный текст по идентификатору сообщения; изменяется.

    Returns:
        list[str]: Приросты в порядке появления. Пустой список, если текст не
        вырос: поток повторяет уже виденное на каждом шаге.
    """
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if not isinstance(item, dict) or item.get("type") != "ai":
            continue
        message_id = item.get("id")
        content = item.get("content") or ""
        if not message_id or not isinstance(content, str):
            continue
        seen = partials.get(message_id, "")
        if len(content) > len(seen):
            partials[message_id] = content
            out.append(content[len(seen) :])
    return out
