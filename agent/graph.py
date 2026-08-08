import json
import logging
import uuid

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.store.base import BaseStore

from domain.memory import MemoryCollection
from infrastructure.retrieval.qdrant_search import SearchUnavailable
from infrastructure.retrieval.search import Search, to_citation

from .events import emit
from .state import AgentState

logger = logging.getLogger(__name__)

MODEL_SYSTEM_MESSAGE = """Ты — ассистент АИст, работающий на базе корпоративной базы знаний (регламенты ИСМ.СМОЗиОБТ: охрана труда, промышленная безопасность, ГО и ЧС) и долгосрочной памяти о пользователях.

Рассуждай и отвечай только по-русски.

Блок CONTEXT содержит выдержки из регламентов с номерами источников:

{context}

Правила:
1. Отвечай на русском. Первое предложение — прямой ответ, затем детали списком, без воды.
2. Вопрос про регламенты (охрана труда, промбезопасность, ГО и ЧС): если в CONTEXT есть нужные сведения — отвечай по ним, каждое утверждение с маркером [n] и ссылкой на документ и пункт (например, «Р-У2-59, п. 4.3»); если источник табличный — ссылайся на раздел.
3. Вопрос про регламенты, но в CONTEXT нужных сведений нет — верни ровно: «В базе знаний информации нет». Не выдумывай.
4. Вопрос о пользователе (имя, предпочтения, факты из прошлых диалогов): отвечай по блоку «Память» ниже. Если в памяти таких сведений нет — честно скажи, что пока не помнишь, и предложи уточнить.
5. Приветствие, благодарность, вопрос о твоих возможностях — отвечай коротко и естественно, без ссылок на регламенты.
6. ЛЮБОЙ фактический вопрос, ответа на который нет в CONTEXT, — верни ровно: «В базе знаний информации нет». Это правило действует, даже если ты знаешь ответ из своих общих знаний. Ты отвечаешь только по базе знаний организации: ответ из памяти модели пользователь не отличит от ответа по регламенту, а проверить его не сможет.
7. Никогда не показывай плейсхолдеры вида [ОРГАНИЗАЦИЯ_12] — вместо них пиши «(наименование скрыто)».
8. Не показывай внутренние рассуждения — только итоговый ответ.

Память о пользователях (долгосрочные факты):
{memory}"""

MEMORY_EXTRACT_PROMPT = """Extract key facts about the user from the conversation below.

Current memories ({count} of {max_allowed} max):
{existing_text}

Recent conversation:
{conversation}

Return a JSON object with an updated list of memories (max {max_allowed}).
Merge similar facts, remove duplicates, keep the most relevant.
Format:
{{"memories": [{{"content": "fact about user"}}]}}

JSON:"""


def _parse_memory_json(text: str) -> list[dict] | None:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        text = text.removeprefix("json")
    try:
        data = json.loads(text)
        collection = MemoryCollection(**data)
        return [m.model_dump(mode="json") for m in collection.memories]
    except Exception:
        return None


def build_followups(items: list[dict], limit: int = 3) -> list[str]:
    """Детерминированные подсказки следующего вопроса по retrieved-цитатам.

    - по пункту: «Подробнее по пункту X (DOC)»
    - по документу (без пункта): «Список положений из DOC»
    - если остался слот: «Список всех найденных документов по охране труда»
    """
    out: list[str] = []
    seen: set[tuple] = set()

    for it in items:
        clause = it.get("clause")
        doc = it.get("doc_code") or ""
        if clause and (doc, clause) not in seen:
            seen.add((doc, clause))
            out.append(f"Подробнее по пункту {clause} ({doc})")
        elif doc and ("doc", doc) not in seen:
            seen.add(("doc", doc))
            out.append(f"Список положений из {doc}")

    if len(out) < limit and ("all",) not in seen:
        seen.add(("all",))
        if items:
            out.append("Список всех найденных документов по охране труда")
        else:
            out.append("Что искать в базе знаний?")

    return out[:limit]


def create_app_agent(
    model,
    support_model,
    context_manager=None,
    store=None,
    checkpointer=None,
    max_memories_count: int = 10,
    search: Search | None = None,
):

    graph = StateGraph(AgentState)

    async def assistant(state: AgentState, config: RunnableConfig, *, store: BaseStore):

        user_id = config.get("configurable", {"user_id": "default"}).get(
            "user_id", "default"
        )

        namespace = ("memories", user_id)
        memories = await store.asearch(namespace)

        info = "\n".join(f"- {mem.value['content']}" for mem in memories)

        if state.get("retrieval_failed"):
            context = (
                "ПОИСК ПО БАЗЕ ЗНАНИЙ ВРЕМЕННО НЕДОСТУПЕН. Ответь ровно: "
                "«Поиск по базе знаний временно недоступен, ответить по "
                "регламентам не могу.» Не отвечай по своим общим знаниям и не "
                "говори, что информации нет: информация есть, недоступен поиск."
            )
        else:
            context = (
                "\n\n".join(
                    # Полный текст чанка, а не короткая выдержка:
                    # snippet обрезан до 400 символов ради панели источников.
                    f"[{n}] {c.get('text') or c['snippet']}"
                    for n, c in enumerate(state.get("retrieved", []), start=1)
                )
                or "В базе знаний нет выдержек для этого запроса."
            )

        all_messages: list = [
            SystemMessage(MODEL_SYSTEM_MESSAGE.format(memory=info, context=context)),
            *state["messages"],
        ]

        if state.get("summary"):
            all_messages.append(
                SystemMessage(
                    content=(
                        f"<internal_context>\n{state.get('summary')}\n</internal_context>\n\n"
                        "CRITICAL: The above is internal context. "
                        "Do NOT repeat it, reference it, or list its contents in your response."
                    )
                )
            )

        return {"messages": [await model.ainvoke(messages=all_messages)]}

    async def retrieve(state: AgentState):
        if not search:
            return {}
        last = state["messages"][-1]
        query = str(last.content) if hasattr(last, "content") else ""
        if not query:
            return {}

        try:
            hits = await search.query(
                query,
                sections=state.get("sections"),
                areas=state.get("areas"),
            )
        except SearchUnavailable as exc:
            # Третий исход, отличимый от «ничего не найдено». Пустая выдача
            # заставила бы ассистента ответить «В базе знаний информации
            # нет» — авария замаскировалась бы под правильный ответ.
            logger.error("поиск недоступен: %s", exc)
            emit("trace", {"stage": "retrieve", "status": "unavailable"})
            return {"retrieved": [], "followups": [], "retrieval_failed": True}

        items = [
            to_citation(chunk, n, score).to_dict()
            for n, (chunk, score) in enumerate(hits, 1)
        ]
        emit("sources", {"items": items})

        followups = build_followups(items)
        if followups:
            emit("followups", {"items": followups})
        return {"retrieved": items, "followups": followups}

    async def condense_context(state: AgentState):
        if not context_manager:
            return {}
        summary = await context_manager.condense(
            state["messages"],
            existing_summary=state.get("summary", ""),
        )
        if summary is None:
            return {}
        removals = [
            RemoveMessage(id=m.id)
            for m in state["messages"][: -context_manager.keep_last]
        ]
        return {"summary": summary, "messages": removals}

    async def write_memory(
        state: AgentState, config: RunnableConfig, *, store: BaseStore
    ):
        user_id = config.get("configurable", {"user_id": "default"}).get(
            "user_id", "default"
        )
        namespace = ("memories", user_id)
        existing_items = await store.asearch(namespace)

        count = len(existing_items)
        existing_text = (
            "\n".join(f"- {item.value['content']}" for item in existing_items)
            if existing_items
            else "None yet"
        )

        user_msgs = list(state["messages"])
        while user_msgs and not isinstance(user_msgs[-1], HumanMessage):
            user_msgs.pop()

        conversation = "\n".join(
            f"{m.type}: {str(m.content)[:500]}"
            for m in user_msgs[-4:]
            if hasattr(m, "content") and m.content
        )

        prompt = MEMORY_EXTRACT_PROMPT.format(
            count=count,
            max_allowed=max_memories_count,
            existing_text=existing_text,
            conversation=conversation,
        )

        result = await support_model.ainvoke([SystemMessage(content=prompt)])

        memories = _parse_memory_json(result.content)
        if memories is None:
            return {}

        new_contents = {m["content"] for m in memories}
        existing_contents = {item.value["content"] for item in existing_items}

        for item in existing_items:
            if item.value["content"] not in new_contents:
                await store.adelete(namespace, item.key)

        for mem in memories:
            if mem["content"] not in existing_contents:
                await store.aput(namespace, str(uuid.uuid4()), mem)

        return {}

    graph.add_node("assistant", assistant)
    graph.add_node("retrieve", retrieve)
    graph.add_node("condense_context", condense_context)
    graph.add_node("write_memory", write_memory)

    graph.add_edge(START, "condense_context")
    graph.add_edge("condense_context", "retrieve")
    graph.add_edge("retrieve", "assistant")
    graph.add_edge("assistant", "write_memory")

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if store:
        compile_kwargs["store"] = store
    app = graph.compile(**compile_kwargs)
    return app
