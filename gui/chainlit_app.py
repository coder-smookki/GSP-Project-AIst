import logging
import os
import time
import uuid

import bcrypt
import chainlit as cl
from chainlit.input_widget import MultiSelect
from chainlit.server import app as _chainlit_app
from fastapi import APIRouter, HTTPException
from langgraph_sdk import get_client
from pydantic import BaseModel

from gui.contract import (
    build_input,
    decode_aist_envelope,
    format_sources,
    pick_final_text,
)
from gui.db import create_user, get_user, init_db, seed_admin, update_password

logger = logging.getLogger(__name__)

AEGRA_URL = os.getenv("AEGRA_URL", "http://aegra:2026")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# Области знаний — предметная ось, которую выбирает пользователь.
# Раньше здесь были типы разделов документа (procedura, sokrashcheniya):
# это другая, служебная ось, и она не отвечала требованию кейса «выбор
# областей знаний». Хуже того, выбор по умолчанию отсекал 71 % корпуса,
# а значения sokrashcheniya не существует вовсе — оно не находило ничего.
AREA_CHOICES = {
    "Пожарная безопасность": "pozharnaya_bezopasnost",
    "Охрана труда": "ohrana_truda",
    "Гражданская оборона и ЧС": "go_i_chs",
    "Промышленная безопасность": "promyshlennaya_bezopasnost",
    "Производственная безопасность": "proizvodstvennaya_bezopasnost",
    "Работы повышенной опасности": "raboty_povyshennoy_opasnosti",
    "Прочее": "prochee",
}

# Пустой список = искать по всей базе. Непустой список по умолчанию
# молча сузил бы поиск, и пользователь не понял бы, почему ответа нет.
DEFAULT_AREAS: list[str] = []

# Заводить ли учётную запись при первом входе неизвестного логина.
#
# По умолчанию ВЫКЛЮЧЕНО: раньше любой незнакомый логин автоматически
# получал доступ к базе знаний, задав пароль первой попыткой входа.
# Включать только для локальной отладки.
ALLOW_SELF_REGISTRATION = os.getenv("ALLOW_SELF_REGISTRATION", "0") == "1"


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    if ADMIN_USERNAME and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return cl.User(
            identifier=username,
            metadata={"role": "admin", "provider": "admin_env"},
        )

    try:
        user = await get_user(username)
    except Exception:
        logger.exception("DB error for %s", username)
        return None
    if user is None:
        if not ALLOW_SELF_REGISTRATION:
            # Любой незнакомый логин заводил учётную запись автоматически:
            # пароль задавался первой же попыткой входа, то есть доступ к
            # базе знаний получал кто угодно. Для корпоративного контура
            # это дыра, а на защите — прямой вопрос от ИБ.
            logger.warning(
                "вход отклонён: пользователь %s не заведён, "
                "самостоятельная регистрация выключена",
                username,
            )
            return None
        ok = await create_user(username, password)
        if ok:
            logger.info("заведён пользователь %s (разрешена саморегистрация)", username)
            return cl.User(
                identifier=username,
                metadata={"role": "user", "provider": "database"},
            )
        logger.info("не удалось завести пользователя %s", username)
        return None

    if user and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return cl.User(
            identifier=username,
            metadata={"role": user["role"], "provider": "database"},
        )
    logger.info("Login failed for %s", username)
    return None


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    """Возвращает подсказки, с которых можно начать диалог.

    Нужны не для красоты: на защите пустое поле ввода заставляет думать, что
    спросить, а неудачная формулировка первого вопроса портит впечатление о
    поиске. Формулировки взяты из тем реального корпуса раздела 44030.

    Returns:
        list[cl.Starter]: Подсказки для стартового экрана.
    """
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

    thread = await client.threads.create()

    cl.user_session.set("client", client)
    cl.user_session.set("assistant_id", assistants[0]["assistant_id"])
    cl.user_session.set("thread_id", thread["thread_id"])
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("areas", DEFAULT_AREAS)

    chat_settings = cl.ChatSettings(
        inputs=[
            MultiSelect(
                id="areas",
                label="Области знаний",
                items=AREA_CHOICES,
                initial=DEFAULT_AREAS,
            )
        ]
    )
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
    await msg.send()

    # Ключ панели источников должен существовать ДО цикла: панель рисуется
    # сразу по приходу события sources, а не после конца потока. Раньше
    # отрисовка стояла после цикла, а поток завершается только когда
    # отработает узел write_memory — то есть спустя ещё один полный вызов
    # модели. Источники появлялись на несколько секунд позже ответа, хотя
    # событие приходило до первого токена.
    sidebar_key = f"sources-{msg.id or uuid.uuid4()}"
    sidebar_drawn = False

    areas: list[str] | None = cl.user_session.get("areas")
    sources_items: list[dict] = []
    followups: list[str] = []
    partials: dict[str, str] = {}
    assistant_final: list[str] = []
    run_error: Exception | None = None

    try:
        async for chunk in client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=build_input(message.content, areas=areas),
            config={"configurable": {"user_id": user_id or "anonymous"}},
            stream_mode=["messages", "updates", "custom"],
            version="v1",
        ):
            mode = (chunk.event or "").split("|", 1)[0]

            if mode == "messages/partial":
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
                        await msg.stream_token(content[len(prev) :])
                continue

            if mode == "custom":
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
                continue

            if chunk.event != "updates":
                continue

            for node_name, data in chunk.data.items():
                if node_name == "assistant" and data:
                    for m in data.get("messages") or []:
                        if m.get("type") == "ai" and not m.get("tool_calls"):
                            assistant_final.append(str(m.get("content") or ""))
    except Exception as exc:
        run_error = exc
        logger.exception("run stream failed")

    msg.content = pick_final_text(partials, assistant_final, error=run_error)

    # Панель уже отрисована внутри цикла. Здесь остаётся только случай,
    # когда источников не было вовсе: старую панель надо убрать, иначе она
    # покажет источники предыдущего вопроса.
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

    await msg.update()


@cl.action_callback("followup")
async def on_followup(action: cl.Action):
    """Обрабатывает клик по подсказке следующего вопроса.

    Args:
        action: Действие с текстом вопроса в поле ``payload["text"]``.
    """
    text = str(action.payload.get("text") or "").strip()
    if not text:
        return

    # Сообщение надо ОТПРАВИТЬ, а не просто передать обработчику: иначе в
    # переписке появляется ответ без вопроса, и её нельзя перечитать —
    # непонятно, на что именно система отвечала.
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

#: Сколько неудачных попыток допускается в окне.
_MAX_ATTEMPTS = 5

#: Длина окна ограничения, с.
_ATTEMPT_WINDOW = 300.0


def _too_many_attempts(username: str) -> bool:
    """Проверяет, не исчерпан ли лимит попыток смены пароля.

    Эндпоинт не требует сессии, поэтому без ограничения он превращается в
    оракул для подбора пароля и перечисления учётных записей.

    Args:
        username: Логин, по которому идёт попытка.

    Returns:
        bool: ``True``, если лимит исчерпан.
    """
    now = time.monotonic()
    history = _password_attempts.get(username, [])
    attempts = [t for t in history if now - t < _ATTEMPT_WINDOW]
    _password_attempts[username] = attempts
    return len(attempts) >= _MAX_ATTEMPTS


@_password_router.post("/change-password")
async def change_password(body: _ChangePasswordBody):
    """Меняет пароль пользователя.

    Args:
        body: Логин, текущий и новый пароль.

    Returns:
        dict: Признак успеха и сообщение.

    Raises:
        HTTPException: 429 при исчерпании лимита попыток, 400 при неверных
            данных. Текст ошибки одинаков для «нет такого пользователя» и
            «неверный пароль»: разные ответы позволяли бы перечислять
            существующие учётные записи.
    """
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
