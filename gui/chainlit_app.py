import logging
import os
import time
import uuid

import bcrypt
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import InputWidget, MultiSelect
from chainlit.server import app as _chainlit_app
from chainlit.types import ThreadDict
from fastapi import APIRouter, HTTPException
from langgraph_sdk import get_client
from pydantic import BaseModel

from gui.contract import (
    build_input,
    decode_aist_envelope,
    format_sources,
    pick_final_text,
)
from gui.db import (
    async_database_url,
    create_user,
    get_user,
    init_db,
    seed_admin,
    update_password,
)
from gui.stages import StageView

logger = logging.getLogger(__name__)

AEGRA_URL = os.getenv("AEGRA_URL", "http://aegra:2026")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

AREA_CHOICES = {
    "Пожарная безопасность": "pozharnaya_bezopasnost",
    "Охрана труда": "ohrana_truda",
    "Гражданская оборона и ЧС": "go_i_chs",
    "Промышленная безопасность": "promyshlennaya_bezopasnost",
    "Производственная безопасность": "proizvodstvennaya_bezopasnost",
    "Работы повышенной опасности": "raboty_povyshennoy_opasnosti",
    "Прочее": "prochee",
}

# Пустой список = поиск по всей базе (см. docs/GUI.md).
DEFAULT_AREAS: list[str] = []

# По умолчанию выключена: доступ по первому вводу неизвестного логина
# получали бы все (см. docs/GUI.md).
ALLOW_SELF_REGISTRATION = os.getenv("ALLOW_SELF_REGISTRATION", "0") == "1"


@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    return SQLAlchemyDataLayer(conninfo=async_database_url())


async def _ensure_persisted(user: cl.User) -> cl.User:
    try:
        layer = cl.data.get_data_layer()
        if layer is not None:
            await layer.create_user(user)
    except Exception:
        logger.exception(
            "не удалось завести пользователя %s в слое данных", user.identifier
        )
    return user


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    if ADMIN_USERNAME and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return await _ensure_persisted(
            cl.User(
                identifier=username,
                metadata={"role": "admin", "provider": "admin_env"},
            )
        )

    try:
        user = await get_user(username)
    except Exception:
        logger.exception("DB error for %s", username)
        return None
    if user is None:
        if not ALLOW_SELF_REGISTRATION:
            logger.warning(
                "вход отклонён: пользователь %s не заведён, "
                "самостоятельная регистрация выключена",
                username,
            )
            return None
        ok = await create_user(username, password)
        if ok:
            logger.info("заведён пользователь %s (разрешена саморегистрация)", username)
            return await _ensure_persisted(
                cl.User(
                    identifier=username,
                    metadata={"role": "user", "provider": "database"},
                )
            )
        logger.info("не удалось завести пользователя %s", username)
        return None

    if user and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return await _ensure_persisted(
            cl.User(
                identifier=username,
                metadata={"role": user["role"], "provider": "database"},
            )
        )
    logger.info("Login failed for %s", username)
    return None


@cl.set_starters
async def set_starters(
    user: cl.User | None, language_code: str | None
) -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Действия при пожаре",
            message="Что должен сделать работник при обнаружении пожара?",
        ),
        cl.Starter(
            label="Наряд-допуск",
            message="Кто выдаёт наряд-допуск на работы повышенной опасности?",
        ),
        cl.Starter(
            label="Средства защиты",
            message=(
                "Какие средства индивидуальной защиты обязательны "
                "на объекте строительства?"
            ),
        ),
        cl.Starter(
            label="Обучение по ГО и ЧС",
            message="Как организовано обучение работников по гражданской обороне?",
        ),
        cl.Starter(
            label="Расследование происшествий",
            message=("В каком порядке расследуется несчастный случай на производстве?"),
        ),
        cl.Starter(
            label="Оценка условий труда",
            message=(
                "Как проводится специальная оценка условий труда на рабочих местах?"
            ),
        ),
    ]


@cl.on_app_startup
async def on_app_startup():
    await init_db()


@cl.on_chat_start
async def on_chat_start():
    await init_db()
    await seed_admin()

    app_user = cl.user_session.get("user")
    if app_user is None:
        await cl.Message(content="Требуется вход в систему.").send()
        return
    user_id = app_user.identifier

    client = get_client(url=AEGRA_URL)
    assistants = await client.assistants.search()
    if not assistants:
        await cl.Message(
            content="Сервис агента недоступен: не найден ни один ассистент."
        ).send()
        return

    # Идентификатор треда aegra совпадает с идентификатором диалога
    # Chainlit (см. docs/GUI.md); `if_exists="do_nothing"` делает вызов
    # идемпотентным при повторном входе.
    thread_id = cl.context.session.thread_id
    thread = await client.threads.create(thread_id=thread_id, if_exists="do_nothing")

    cl.user_session.set("client", client)
    cl.user_session.set("assistant_id", assistants[0]["assistant_id"])
    cl.user_session.set("thread_id", thread["thread_id"])
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("areas", DEFAULT_AREAS)

    widgets: list[InputWidget] = [
        MultiSelect(
            id="areas",
            label="Области знаний",
            items=AREA_CHOICES,
            initial=DEFAULT_AREAS,
        ),
        cl.input_widget.Checkbox(
            id="always_answer_with_db",
            label="Всегда искать в базе знаний",
            initial=False,
            description=(
                "Принудительно использовать поиск по базе даже для общих вопросов"
            ),
        ),
    ]
    chat_settings = cl.ChatSettings(inputs=widgets)
    await chat_settings.send()

    await cl.Message(
        content=(
            f"Здравствуйте, **{user_id}**. Я АИст — помощник по корпоративной базе "
            "знаний: охрана труда, промышленная и пожарная безопасность, "
            "гражданская оборона и ЧС.\n\n"
            "Задайте вопрос своими словами. В ответе будут ссылки на документ "
            "и пункт регламента, а если сведений в базе нет — я так и скажу."
        )
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    app_user = cl.user_session.get("user")
    if app_user is None:
        await cl.Message(content="Требуется вход в систему.").send()
        return

    client = get_client(url=AEGRA_URL)
    assistants = await client.assistants.search()
    if not assistants:
        await cl.Message(
            content="Сервис агента недоступен: не найден ни один ассистент."
        ).send()
        return

    thread_id = str(thread.get("id") or cl.context.session.thread_id)
    await client.threads.create(thread_id=thread_id, if_exists="do_nothing")

    cl.user_session.set("client", client)
    cl.user_session.set("assistant_id", assistants[0]["assistant_id"])
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("user_id", app_user.identifier)
    cl.user_session.set("areas", DEFAULT_AREAS)
    logger.info("диалог %s восстановлен для %s", thread_id, app_user.identifier)


@cl.on_settings_update
async def on_settings_update(settings: dict):
    raw = settings.get("areas")
    if isinstance(raw, list):
        areas = raw
    elif raw is None or raw == "":
        areas = []
    else:
        areas = [raw]
    cl.user_session.set("areas", areas)

    # чекбокс «Всегда искать в базе знаний»
    always_db = settings.get("always_answer_with_db")
    if isinstance(always_db, bool):
        cl.user_session.set("always_answer_with_db", always_db)


@cl.on_message
async def on_message(message: cl.Message):
    client = cl.user_session.get("client")
    assistant_id: str | None = cl.user_session.get("assistant_id")
    thread_id: str | None = cl.user_session.get("thread_id")
    user_id: str | None = cl.user_session.get("user_id")

    if client is None or assistant_id is None or thread_id is None:
        await cl.Message(
            content="Сессия не инициализирована. Обновите страницу."
        ).send()
        return

    msg = cl.Message(content="")
    msg_sent = False

    async def ensure_message_sent() -> None:
        """Отправляет пустое сообщение ответа не раньше, чем оно нужно.

        Порядок в ленте задаётся моментом отправки. Если отправить сообщение
        сразу, шаги этапов встанут ПОД пустым пузырём ответа, и ход работы
        читался бы после самого ответа (см. docs/GUI.md).

        Returns:
            None
        """
        nonlocal msg_sent
        if not msg_sent:
            msg_sent = True
            await msg.send()

    stages = StageView()

    # Ключ панели источников создаётся ДО цикла: панель рисуется по событию
    # sources, а поток завершается только после узла write_memory — то есть
    # заметно позже (см. docs/GUI.md).
    sidebar_key = f"sources-{msg.id or uuid.uuid4()}"
    sidebar_drawn = False

    areas: list[str] | None = cl.user_session.get("areas")
    always_db: bool = cl.user_session.get("always_answer_with_db", False)

    sidebar_key = f"sources-{msg.id or uuid.uuid4()}"
    sidebar_drawn = False

    sources_items: list[dict] = []
    followups: list[str] = []
    partials: dict[str, str] = {}
    assistant_final: list[str] = []
    run_error: Exception | None = None

    # Гейт: стримим токены ТОЛЬКО от узла assistant
    should_stream = False

    try:
        async for chunk in client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=build_input(
                message.content, areas=areas, always_answer_with_db=always_db
            ),
            config={"configurable": {"user_id": user_id or "anonymous"}},
            stream_mode=["messages", "updates", "custom"],
            version="v1",
        ):
            # --- messages/partial: стриминг токенов, но только если should_stream ---
            if chunk.event == "messages/partial" and should_stream:
                for item in chunk.data:
                    if item.get("type") != "ai":
                        continue
                    mid = item.get("id")
                    if not mid:
                        continue
                    content = item.get("content") or ""
                    prev = partials.get(mid, "")
                    if len(content) > len(prev):
                        partials[mid] = content
                        await ensure_message_sent()
                        await msg.stream_token(content[len(prev) :])
                continue

            # --- custom: панель источников и подсказки ---
            #
            # Блок обязан стоять ДО отсева не-updates. Раньше он был под
            # ним, поэтому события custom до него не доходили вовсе:
            # панель «Источники» и подсказки не рисовались никогда, хотя
            # агент их исправно слал (agent/graph.py, узел finalize).
            if chunk.event == "custom":
                env = decode_aist_envelope(chunk.data or {})
                if env:
                    kind = env.get("kind")
                    payload = env.get("payload") or {}
                    if kind == "sources":
                        sources_items = payload.get("items") or []
                        if sources_items:
                            await cl.ElementSidebar.set_title(
                                f"Источники ({len(sources_items)})"
                            )
                            await cl.ElementSidebar.set_elements(
                                [
                                    cl.Text(
                                        content=format_sources(sources_items),
                                        display="inline",
                                    )
                                ],
                                key=sidebar_key,
                            )
                            sidebar_drawn = True
                    elif kind == "followups":
                        followups = payload.get("items") or []
                    elif kind == "trace":
                        await stages.handle(payload)
                continue

            if chunk.event != "updates":
                continue

            # --- updates: управляем should_stream по узлам графа ---
            for node_name, data in chunk.data.items():
                if node_name in ("plan", "finalize") and data:
                    # Включаем стрим после планировщика, а не только после
                    # finalize. В режиме «chat» (приветствие, вопрос о
                    # возможностях, вопрос о самом пользователе) граф идёт
                    # plan -> assistant МИМО finalize, поэтому прежнее условие
                    # там не срабатывало и ответ не стримился вовсе.
                    #
                    # После plan это безопасно в обоих режимах: ветви поиска
                    # живут в подграфе, а он не стримится (в runs.stream нет
                    # subgraphs=True), finalize модель не зовёт, а запись
                    # памяти идёт уже после assistant, когда стрим выключен.
                    should_stream = True
                elif node_name == "assistant" and data:
                    # assistant завершился — выключаем стрим и забираем финальный ответ
                    should_stream = False
                    for m in data.get("messages") or []:
                        if m.get("type") == "ai" and not m.get("tool_calls"):
                            content = str(m.get("content") or "")
                            partials[m.get("id", "")] = content
                            assistant_final.append(content)

    except Exception as exc:
        run_error = exc
        logger.exception("поток запуска графа оборван")
    finally:
        # Незакрытый шаг остался бы «выполняется» навсегда.
        await stages.close()

    msg.content = pick_final_text(partials, assistant_final, error=run_error)

    # Когда источников не было — убрать панель от предыдущего вопроса.
    if not sidebar_drawn:
        await cl.ElementSidebar.set_elements([], key=sidebar_key)

    if followups:
        msg.actions = [
            cl.Action(
                name="followup",
                label=label,
                payload={"text": label},
            )
            for label in followups[:4]
        ]

    await ensure_message_sent()
    await msg.update()


@cl.action_callback("followup")
async def on_followup(action: cl.Action):
    text = str(action.payload.get("text") or "").strip()
    if not text:
        return

    # Вопрос отправляется сообщением, чтобы в переписке был ответ с вопросом
    # (см. docs/GUI.md).
    question = cl.Message(content=text, type="user_message")
    await question.send()
    await on_message(question)


_password_router = APIRouter()


class _ChangePasswordBody(BaseModel):
    username: str
    current_password: str
    new_password: str


#: Неудачные попытки смены пароля: логин -> отметки времени.
_password_attempts: dict[str, list[float]] = {}

#: Попыток в окне и длина окна ограничения, с (см. docs/GUI.md).
_MAX_ATTEMPTS = 5
_ATTEMPT_WINDOW = 300.0


def _too_many_attempts(username: str) -> bool:
    now = time.monotonic()
    history = _password_attempts.get(username, [])
    attempts = [t for t in history if now - t < _ATTEMPT_WINDOW]
    _password_attempts[username] = attempts
    return len(attempts) >= _MAX_ATTEMPTS


@_password_router.post("/change-password")
async def change_password(body: _ChangePasswordBody):
    if _too_many_attempts(body.username):
        logger.warning("превышен лимит попыток смены пароля для %s", body.username)
        raise HTTPException(
            status_code=429, detail="Слишком много попыток, повторите позже"
        )

    ok, message = await update_password(
        body.username, body.current_password, body.new_password
    )
    if not ok:
        _password_attempts.setdefault(body.username, []).append(time.monotonic())
        logger.info("смена пароля отклонена для %s: %s", body.username, message)
        raise HTTPException(
            status_code=400, detail="Не удалось изменить пароль: проверьте данные"
        )

    _password_attempts.pop(body.username, None)
    logger.info("пароль изменён для %s", body.username)
    return {"success": True, "message": message}


_chainlit_app.include_router(_password_router)
