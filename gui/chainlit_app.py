import logging
import os

import bcrypt
import chainlit as cl
from fastapi import APIRouter, HTTPException
from langgraph_sdk import get_client
from pydantic import BaseModel

from gui.db import create_user, get_user, init_db, seed_admin, update_password

logger = logging.getLogger(__name__)

AEGRA_URL = os.getenv("AEGRA_URL", "http://aegra:2026")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


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

    await cl.Message(
        content=f"Hello **{user_id}**! I'm your AI assistant. Ask me anything."
    ).send()


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

    should_stream = False
    content_by_id = {}

    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"type": "human", "content": message.content}]},
        config={"configurable": {"user_id": user_id or "anonymous"}},
        stream_mode=["messages", "updates"],
    ):
        if chunk.event == "messages/partial" and should_stream:
            for item in chunk.data:
                if item.get("type") != "ai":
                    continue
                mid = item.get("id")
                if not mid:
                    continue
                content = item.get("content") or ""
                prev = content_by_id.get(mid, "")
                if len(content) > len(prev):
                    await msg.stream_token(content[len(prev) :])
                    content_by_id[mid] = content
            continue

        if chunk.event != "updates":
            continue

        for node_name, data in chunk.data.items():
            if node_name == "condense_context":
                should_stream = True
            elif node_name == "assistant" and data:
                for m in data.get("messages") or []:
                    if m.get("type") == "ai" and not m.get("tool_calls"):
                        should_stream = False

    await msg.update()


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


from chainlit.server import app as _chainlit_app

_chainlit_app.include_router(_password_router)
