import logging
import os
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

SECTION_CHOICES = {
    "Процедуры": "procedura",
    "Сокращения": "sokrashcheniya",
}
DEFAULT_SECTIONS = list(SECTION_CHOICES.values())


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
        ok = await create_user(username, password)
        if ok:
            logger.info("Auto-registered user: %s", username)
            return cl.User(
                identifier=username,
                metadata={"role": "user", "provider": "database"},
            )
        logger.info("Create user failed (race?) for %s", username)
        return None

    if user and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return cl.User(
            identifier=username,
            metadata={"role": user["role"], "provider": "database"},
        )
    logger.info("Login failed for %s", username)
    return None


@cl.on_chat_start
async def on_chat_start():
    await init_db()
    await seed_admin()

    app_user = cl.user_session.get("user")
    if app_user is None:
        await cl.Message(content="Authentication required. Please log in.").send()
        return
    user_id = app_user.identifier

    client = get_client(url=AEGRA_URL)
    assistants = await client.assistants.search()
    if not assistants:
        await cl.Message(content="Error: no assistants found on Aegra server").send()
        return

    thread = await client.threads.create()

    cl.user_session.set("client", client)
    cl.user_session.set("assistant_id", assistants[0]["assistant_id"])
    cl.user_session.set("thread_id", thread["thread_id"])
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("sections", DEFAULT_SECTIONS)

    chat_settings = cl.ChatSettings(
        inputs=[
            MultiSelect(
                id="sections",
                label="Разделы знаний",
                items=SECTION_CHOICES,
                initial=DEFAULT_SECTIONS,
            )
        ]
    )
    await chat_settings.send()

    await cl.Message(
        content=f"Hello **{user_id}**! I'm your AI assistant. Ask me anything."
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    raw = settings.get("sections")
    if isinstance(raw, list):
        sections = raw
    elif raw is None or raw == "":
        sections = []
    else:
        sections = [raw]
    cl.user_session.set("sections", sections)


@cl.on_message
async def on_message(message: cl.Message):
    client = cl.user_session.get("client")
    assistant_id: str | None = cl.user_session.get("assistant_id")
    thread_id: str | None = cl.user_session.get("thread_id")
    user_id: str | None = cl.user_session.get("user_id")

    if client is None or assistant_id is None or thread_id is None:
        await cl.Message(content="Session error: not initialized").send()
        return

    msg = cl.Message(content="")
    await msg.send()

    sections: list[str] | None = cl.user_session.get("sections")
    sources_items: list[dict] = []
    followups: list[str] = []
    partials: dict[str, str] = {}
    assistant_final: list[str] = []
    run_error: Exception | None = None

    try:
        async for chunk in client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=build_input(message.content, sections),
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

    sidebar_key = f"sources-{msg.id or uuid.uuid4()}"

    if sources_items:
        marker = f"Источники ({len(sources_items)})"
        await cl.ElementSidebar.set_title(marker)
        await cl.ElementSidebar.set_elements(
            [
                cl.Text(
                    content=format_sources(sources_items),
                    display="inline",
                )
            ],
            key=sidebar_key,
        )
    else:
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
    text = str(action.payload.get("text") or "").strip()
    if not text:
        return
    await on_message(cl.Message(content=text))


_password_router = APIRouter()


class _ChangePasswordBody(BaseModel):
    username: str
    current_password: str
    new_password: str


@_password_router.post("/change-password")
async def change_password(body: _ChangePasswordBody):
    ok, message = await update_password(
        body.username, body.current_password, body.new_password
    )
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"success": True, "message": message}


_chainlit_app.include_router(_password_router)
