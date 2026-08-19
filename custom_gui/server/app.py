"""Маршруты интерфейса.

Всё, что видит браузер, проходит через этот слой: он держит сессию, знает
владельца каждого диалога и один разговаривает с агентом. Наружу отдаётся
только то, что нужно нарисовать.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from custom_gui.server import session as session_module
from custom_gui.server import store
from custom_gui.server.agent_client import AgentClient, AgentUnavailable
from custom_gui.server.config import SESSION_TTL_SECONDS, Config
from gui.db import get_user, seed_admin

logger = logging.getLogger(__name__)

#: Каталог собранной статики. Появляется после `npm run build`.
STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "dist"

#: Сколько ждать между служебными комментариями в потоке, с.
#:
#: Ответ рождается за 9-12 секунд, а посредники и прокси рвут соединение,
#: если по нему долго ничего не идёт. Комментарий SSE ничего не значит для
#: клиента и держит канал живым.
HEARTBEAT_SECONDS = 15.0


class LoginRequest(BaseModel):
    """Тело запроса на вход."""

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AskRequest(BaseModel):
    """Тело запроса вопроса к агенту."""

    question: str = Field(min_length=1, max_length=4000)
    areas: list[str] = Field(default_factory=list, max_length=32)
    docs: list[str] = Field(default_factory=list, max_length=200)
    sections: list[str] = Field(default_factory=list, max_length=32)
    always_search: bool = False


class RenameRequest(BaseModel):
    """Тело запроса на переименование диалога."""

    title: str = Field(min_length=1, max_length=120)


def create_app(config: Config | None = None) -> FastAPI:
    """Собирает приложение интерфейса.

    Args:
        config: Готовая конфигурация; пусто - читается из окружения.

    Returns:
        FastAPI: Приложение с маршрутами и статикой.
    """
    settings = config or Config.from_env()
    agent = AgentClient(
        settings.aegra_url, settings.aegra_token, settings.request_timeout
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Готовит хранилище на старте и закрывает соединения на выходе.

        Yields:
            None: Время жизни приложения.
        """
        await store.setup()
        await seed_admin()
        try:
            yield
        finally:
            await agent.aclose()

    app = FastAPI(title="АИст", docs_url=None, redoc_url=None, lifespan=lifespan)

    # ------------------------------------------------------------- сессия

    def current(request: Request) -> session_module.Session:
        """Достаёт сессию из cookie.

        Args:
            request: Входящий запрос.

        Returns:
            session_module.Session: Сессия вошедшего пользователя.

        Raises:
            HTTPException: 401, если сессии нет или она непригодна.
        """
        cookie = request.cookies.get(session_module.COOKIE_NAME)
        try:
            return session_module.verify(cookie, settings.session_secret)
        except session_module.InvalidSession as exc:
            raise HTTPException(status_code=401, detail="требуется вход") from exc

    #: Вошедший пользователь как зависимость маршрута.
    CurrentUser = Annotated[session_module.Session, Depends(current)]

    async def owned(conversation_id: str, user: session_module.Session) -> str:
        """Убеждается, что диалог принадлежит пользователю.

        Args:
            conversation_id: Идентификатор диалога из запроса.
            user: Сессия пользователя.

        Returns:
            str: Тот же идентификатор.

        Raises:
            HTTPException: 404, если диалога нет или он чужой. Именно 404, а не
                403: разные коды подсказали бы, какие идентификаторы существуют.
        """
        if not await store.owns(user.username, conversation_id):
            raise HTTPException(status_code=404, detail="диалог не найден")
        return conversation_id

    # -------------------------------------------------------------- вход

    @app.post("/api/login")
    async def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
        """Проверяет учётные данные и выдаёт сессию.

        Args:
            payload: Имя пользователя и пароль.
            response: Ответ, в который ставится cookie.

        Returns:
            dict[str, Any]: Имя и роль вошедшего.

        Raises:
            HTTPException: 401, если пара не подошла.
        """
        role = await _authenticate(settings, payload.username, payload.password)
        if role is None:
            # Одинаковый ответ на «нет такого» и «неверный пароль»: разные
            # ответы позволяют перебирать существующие имена.
            raise HTTPException(status_code=401, detail="неверный логин или пароль")

        cookie = session_module.issue(
            payload.username, role, settings.session_secret, SESSION_TTL_SECONDS
        )
        response.set_cookie(
            session_module.COOKIE_NAME,
            cookie,
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return {"username": payload.username, "role": role}

    @app.post("/api/logout")
    async def logout(response: Response) -> dict[str, bool]:
        """Гасит сессию.

        Args:
            response: Ответ, из которого убирается cookie.

        Returns:
            dict[str, bool]: Признак выполнения.
        """
        response.delete_cookie(session_module.COOKIE_NAME, path="/")
        return {"ok": True}

    @app.get("/api/me")
    async def me(user: CurrentUser) -> dict[str, Any]:
        """Сообщает, кто вошёл.

        Args:
            user: Сессия пользователя.

        Returns:
            dict[str, Any]: Имя и роль.
        """
        return {"username": user.username, "role": user.role}

    # ---------------------------------------------------------- диалоги

    @app.get("/api/conversations")
    async def conversations(
        user: CurrentUser,
    ) -> dict[str, Any]:
        """Отдаёт список диалогов пользователя.

        Args:
            user: Сессия пользователя.

        Returns:
            dict[str, Any]: Ключ ``items`` со списком диалогов.
        """
        return {"items": await store.list_conversations(user.username)}

    @app.post("/api/conversations")
    async def new_conversation(
        user: CurrentUser,
    ) -> dict[str, Any]:
        """Заводит диалог и открывает под него тред агента.

        Args:
            user: Сессия пользователя.

        Returns:
            dict[str, Any]: Созданный диалог.

        Raises:
            HTTPException: 503, если агент недоступен.
        """
        created = await store.create_conversation(user.username)
        try:
            await agent.ensure_thread(user.username, created["id"])
        except AgentUnavailable as exc:
            await store.delete(created["id"])
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return created

    @app.get("/api/conversations/{conversation_id}")
    async def conversation(
        conversation_id: str, user: CurrentUser
    ) -> dict[str, Any]:
        """Отдаёт сообщения диалога.

        Args:
            conversation_id: Идентификатор диалога.
            user: Сессия пользователя.

        Returns:
            dict[str, Any]: Идентификатор и список сообщений.
        """
        await owned(conversation_id, user)
        return {
            "id": conversation_id,
            "messages": await store.load_messages(conversation_id),
        }

    @app.patch("/api/conversations/{conversation_id}")
    async def rename_conversation(
        conversation_id: str,
        payload: RenameRequest,
        user: CurrentUser,
    ) -> dict[str, str]:
        """Переименовывает диалог.

        Args:
            conversation_id: Идентификатор диалога.
            payload: Новое название.
            user: Сессия пользователя.

        Returns:
            dict[str, str]: Сохранённое название.
        """
        await owned(conversation_id, user)
        return {"title": await store.rename(conversation_id, payload.title)}

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str, user: CurrentUser
    ) -> dict[str, bool]:
        """Удаляет диалог и его тред у агента.

        Args:
            conversation_id: Идентификатор диалога.
            user: Сессия пользователя.

        Returns:
            dict[str, bool]: Признак выполнения.
        """
        await owned(conversation_id, user)
        await store.delete(conversation_id)
        await agent.delete_thread(user.username, conversation_id)
        return {"ok": True}

    # ------------------------------------------------------------ вопрос

    @app.post("/api/conversations/{conversation_id}/ask")
    async def ask(
        conversation_id: str,
        payload: AskRequest,
        user: CurrentUser,
    ) -> StreamingResponse:
        """Задаёт вопрос и стримит ответ событиями.

        Args:
            conversation_id: Идентификатор диалога.
            payload: Вопрос и ограничения поиска.
            user: Сессия пользователя.

        Returns:
            StreamingResponse: Поток server-sent events.
        """
        await owned(conversation_id, user)
        question = payload.question.strip()
        await store.append_message(conversation_id, "user", question)
        title = await store.set_title_if_empty(conversation_id, question)

        async def events() -> AsyncIterator[bytes]:
            """Собирает ответ, отдавая события по мере поступления.

            Yields:
                bytes: Строки server-sent events.
            """
            yield _sse("title", {"title": title})
            answer: list[str] = []
            sources: list[dict[str, Any]] = []
            followups: list[str] = []
            stages: list[dict[str, Any]] = []
            try:
                async for kind, item in agent.ask(
                    user.username,
                    conversation_id,
                    question,
                    areas=payload.areas,
                    docs=payload.docs,
                    sections=payload.sections,
                    always_search=payload.always_search,
                ):
                    if kind == "token":
                        answer.append(item["text"])
                    elif kind == "sources":
                        sources = item["items"]
                    elif kind == "followups":
                        followups = item["items"]
                    elif kind == "stage":
                        stages.append(item)
                    yield _sse(kind, item)
            except AgentUnavailable as exc:
                logger.error("ход не завершён: %s", exc)
                yield _sse("error", {"message": str(exc)})
            finally:
                text = "".join(answer)
                if text or sources:
                    await store.append_message(
                        conversation_id,
                        "assistant",
                        text,
                        {
                            "sources": sources,
                            "followups": followups,
                            "stages": stages,
                        },
                    )
                yield _sse("done", {"sources": sources, "followups": followups})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                # Иначе обратный прокси копит ответ целиком и время до первого
                # символа перестаёт быть временем до первого символа.
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------------------------------------------------- каталог

    @app.get("/api/knowledge/tree")
    async def knowledge_tree(
        user: CurrentUser,
    ) -> dict[str, Any]:
        """Отдаёт дерево областей знаний из индекса.

        Args:
            user: Сессия пользователя.

        Returns:
            dict[str, Any]: Ключ ``areas`` со списком областей.

        Raises:
            HTTPException: 503, если каталог недоступен.
        """
        return await _proxy_get(agent, user.username, "/knowledge/tree")

    @app.get("/api/knowledge/documents/{doc_code}")
    async def knowledge_document(
        doc_code: str, user: CurrentUser
    ) -> dict[str, Any]:
        """Отдаёт документ целиком для режима чтения.

        Args:
            doc_code: Код документа.
            user: Сессия пользователя.

        Returns:
            dict[str, Any]: Документ с фрагментами по порядку чтения.

        Raises:
            HTTPException: 404 или 503 от каталога.
        """
        return await _proxy_get(
            agent, user.username, f"/knowledge/documents/{doc_code}"
        )

    # ---------------------------------------------------------- здоровье

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Сообщает готовность интерфейса и его зависимостей.

        Returns:
            dict[str, Any]: Состояние базы и агента.
        """
        database, agent_ok = await asyncio.gather(store.healthy(), agent.healthy())
        return {
            "ok": bool(database and agent_ok),
            "database": bool(database),
            "agent": bool(agent_ok),
        }

    _mount_static(app)
    return app


async def _authenticate(config: Config, username: str, password: str) -> str | None:
    """Проверяет учётные данные по окружению и базе.

    Args:
        config: Конфигурация с учётной записью администратора.
        username: Введённое имя.
        password: Введённый пароль.

    Returns:
        str | None: Роль при успехе, иначе ``None``.
    """
    if config.admin_username and config.admin_password:
        # Сравнение постоянного времени по байтам: на строках оно не принимает
        # не-ASCII и уронило бы вход с кириллическим паролем.
        same_user = hmac.compare_digest(
            username.encode("utf-8"), config.admin_username.encode("utf-8")
        )
        same_pass = hmac.compare_digest(
            password.encode("utf-8"), config.admin_password.encode("utf-8")
        )
        if same_user and same_pass:
            return "admin"

    try:
        user = await get_user(username)
    except Exception:
        logger.exception("не удалось прочитать пользователя %s", username)
        return None
    if user is None:
        return None
    try:
        # bcrypt считается заметное время и блокировал бы событийный цикл.
        ok = await asyncio.to_thread(
            bcrypt.checkpw,
            password.encode("utf-8"),
            user["password_hash"].encode("utf-8"),
        )
    except ValueError:
        # bcrypt отвергает пароли длиннее 72 байт исключением. Без этой ветки
        # длинный пароль давал бы 500 вместо честного отказа.
        return None
    return str(user["role"]) if ok else None


async def _proxy_get(
    agent: AgentClient, username: str, path: str
) -> dict[str, Any]:
    """Передаёт запрос каталога агенту и возвращает ответ.

    Args:
        agent: Клиент агента.
        username: Имя пользователя для заголовка личности.
        path: Путь внутри агента.

    Returns:
        dict[str, Any]: Разобранный ответ.

    Raises:
        HTTPException: С кодом и сообщением от каталога.
    """
    try:
        response = await agent.get_json(username, path)
    except AgentUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if response.status_code >= 400:
        detail = "каталог недоступен"
        with contextlib.suppress(ValueError):
            detail = str(response.json().get("detail") or detail)
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    """Собирает одно событие потока.

    Args:
        event: Имя события.
        payload: Полезная нагрузка.

    Returns:
        bytes: Готовая запись server-sent events.
    """
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode()


def _mount_static(app: FastAPI) -> None:
    """Подключает собранную статику, если она есть.

    Args:
        app: Приложение, к которому подключается статика.
    """
    if not STATIC_DIR.is_dir():
        logger.warning(
            "статика не собрана (%s): запустите `npm run build` в custom_gui/web",
            STATIC_DIR,
        )
        return

    app.mount(
        "/assets",
        StaticFiles(directory=STATIC_DIR / "assets"),
        name="assets",
    )

    @app.get("/{path:path}")
    async def spa(path: str) -> FileResponse:
        """Отдаёт файл статики, а на неизвестный путь - страницу приложения.

        Маршрутизация живёт в браузере, поэтому обновление страницы на любом
        адресе должно возвращать то же приложение, а не 404.

        Args:
            path: Запрошенный путь.

        Returns:
            FileResponse: Файл или ``index.html``.
        """
        candidate = (STATIC_DIR / path).resolve()
        if (
            path
            and candidate.is_file()
            and candidate.is_relative_to(STATIC_DIR.resolve())
        ):
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
