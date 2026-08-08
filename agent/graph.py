import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.store.base import BaseStore

from domain.memory import MemoryCollection
from infrastructure.retrieval.qdrant_search import SearchUnavailable
from infrastructure.retrieval.search import Search, to_citation

from .correction import rewrite_query
from .events import emit
from .grading import grade_chunks
from .memory_gate import worth_extracting
from .numbers import annotate as annotate_numbers
from .state import AgentState

logger = logging.getLogger(__name__)


#: Сбои обращения к модели, которые не должны ронять запрос.
#:
#: `httpx` бросает `ReadTimeout` на долгой генерации и `ConnectError`, когда
#: Ollama не поднята; `ollama` — `ResponseError` на выгруженной модели.
#: Импорт мягкий: в тестах и в сборке без Ollama этих пакетов может не быть,
#: а падать из-за отсутствия класса исключения бессмысленно.
def _model_errors() -> tuple[type[BaseException], ...]:
    """Собирает классы исключений клиентов модели.

    Returns:
        tuple[type[BaseException], ...]: Классы, которые ловятся как
        «модель недоступна». Отсутствующие пакеты просто пропускаются.
    """
    errors: list[type[BaseException]] = []
    try:
        import httpx

        errors += [httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError]
    except ImportError:
        pass
    try:
        from ollama._types import ResponseError

        errors.append(ResponseError)
    except ImportError:
        pass
    try:
        from langchain_core.exceptions import OutputParserException

        errors.append(OutputParserException)
    except ImportError:
        pass
    return tuple(errors)


MODEL_ERRORS = _model_errors()

#: Что показать пользователю, когда модель не ответила.
MODEL_UNAVAILABLE_TEXT = (
    "Не удалось получить ответ модели: сервис генерации сейчас недоступен. "
    "Повторите запрос через минуту."
)

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
6.1. ТЕКСТ ПО ТЕМЕ — ЭТО ЕЩЁ НЕ ОТВЕТ. Поиск подбирает выдержки по совпадению слов, поэтому в CONTEXT почти всегда есть что-то на тему вопроса. Прежде чем отвечать, найди в CONTEXT ту самую строку, где сказано именно то, о чём спросили. Не нашёл — правило 3, отказ. Соседний по смыслу пункт, определение термина или ссылка на другой документ ответом не являются.
6.2. КАЖДОЕ ЧИСЛО — ИЗ CONTEXT, ДОСЛОВНО. Срок, периодичность, количество, размер: перед тем как написать число, убедись, что видишь его в тексте CONTEXT. Если числа там нет — не подставляй правдоподобное и не бери из общих знаний, а откажись. Выдуманное число со ссылкой на источник — худшая из возможных ошибок: пользователь примет его за требование регламента.
6.3. Если CONTEXT ссылается на документ, которого в нём самом нет («порядок установлен положением N»), то ответа у тебя нет. Назови документ, в котором следует искать, и не пересказывай его содержание по памяти.
7. Никогда не показывай плейсхолдеры вида [ОРГАНИЗАЦИЯ_12] — вместо них пиши «(наименование скрыто)».
8. Не показывай внутренние рассуждения — только итоговый ответ.

Сегодняшняя дата: {today}. Опирайся на неё в вопросах о сроках и периодичности; не выдумывай даты, которых нет в CONTEXT.

Память о пользователях (долгосрочные факты):
{memory}"""

MEMORY_EXTRACT_PROMPT = """Ты ведёшь досье фактов о пользователе для корпоративного ассистента.

Запоминай ТОЛЬКО устойчивые сведения о самом человеке:
- имя и как к нему обращаться;
- должность, подразделение, участок, объект;
- обязанности и допуски (например, ответственный за пожарную безопасность);
- явно высказанные предпочтения в общении.

НЕ ЗАПОМИНАЙ, о чём он спрашивал. Записи вида «Пользователь интересуется \
сроком расследования» или «Пользователь спрашивал про огнетушители» — это \
журнал вопросов, а не факты о человеке. Они бесполезны при ответе и \
вытесняют настоящие сведения, потому что список ограничен. Если в разговоре \
нет ничего о самом пользователе, верни список без изменений.

Текущая память ({count} из {max_allowed}):
{existing_text}

Последние сообщения:
{conversation}

Верни JSON с обновлённым списком (не больше {max_allowed}). Объединяй \
близкое, убирай повторы.
Формат:
{{"memories": [{{"content": "факт о пользователе"}}]}}

JSON:"""


#: Часовой пояс, в котором работает заказчик.
MSK = timezone(timedelta(hours=3))


def today_msk(config: RunnableConfig | None = None) -> str:
    """Возвращает сегодняшнюю дату в московском времени.

    Дату можно передать снаружи через ``config["configurable"]["today"]`` —
    это нужно тестам: без фиксированной даты ответы на вопросы о сроках
    невоспроизводимы, и тест на них написать нельзя.

    Args:
        config: Конфигурация запуска графа.

    Returns:
        str: Дата в формате ``ДД.ММ.ГГГГ``.
    """
    if config:
        supplied = (config.get("configurable") or {}).get("today")
        if supplied:
            return str(supplied)
    return datetime.now(MSK).strftime("%d.%m.%Y")


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
    grading: bool = False,
    grading_keep_partial: bool = False,
    verify_numbers: bool = True,
):
    """Собирает граф ассистента.

    Args:
        model: Модель, пишущая ответ пользователю.
        support_model: Служебная модель для сжатия истории, переписывания
            запроса, извлечения памяти и грейдинга.
        context_manager: Сжатие истории; ``None`` — без сжатия.
        store: Хранилище долгосрочной памяти.
        checkpointer: Хранилище состояния диалога.
        max_memories_count: Предел числа запоминаемых фактов о пользователе.
        search: Поиск по базе знаний; ``None`` — узел поиска не работает.
        grading: Отсеивать ли найденные фрагменты моделью перед генерацией.
            Подробности и замеры — в `agent.grading`.
        grading_keep_partial: Оставлять ли фрагменты с вердиктом «частично».
        verify_numbers: Помечать ли в ответе числа, которых нет в выдержках.

    Returns:
        CompiledStateGraph: Скомпилированный граф.
    """

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
            SystemMessage(
                MODEL_SYSTEM_MESSAGE.format(
                    memory=info, context=context, today=today_msk(config)
                )
            ),
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

        try:
            answer = await model.ainvoke(messages=all_messages)
        except MODEL_ERRORS as exc:
            # Отказ модели — это авария сервиса, а не отсутствие сведений.
            # Пропускать исключение наверх нельзя: пользователь увидел бы
            # трассировку вместо ответа, а диалог остался бы без реплики
            # ассистента и не восстановился при перезагрузке.
            logger.error("модель не ответила: %s: %s", type(exc).__name__, exc)
            emit("trace", {"stage": "assistant", "status": "model_unavailable"})
            return {"messages": [AIMessage(content=MODEL_UNAVAILABLE_TEXT)]}

        if verify_numbers and not state.get("retrieval_failed"):
            # Детерминированная проверка поверх модели: срок или количество,
            # которого нет в выдержках, помечается прямо в ответе. Замер
            # показал, что модель выдаёт такие числа со ссылкой на источник,
            # то есть неотличимо от настоящего требования регламента.
            text = str(getattr(answer, "content", "") or "")
            marked = annotate_numbers(text, context)
            if marked != text:
                answer = AIMessage(content=marked)

        return {"messages": [answer]}

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

        if not hits and support_model is not None:
            # Коррекция запроса — только при пустой выдаче. На обычных
            # вопросах этот путь не выполняется вовсе, поэтому средняя цена
            # по всем запросам близка к нулю, а спасает он ровно тот случай,
            # ради которого задуман: вопрос задан житейскими словами, а
            # регламент написан канцелярскими. Круг ровно один: второй стоит
            # ещё вызова модели и ещё поиска при малых шансах на успех.
            emit("trace", {"stage": "retrieve", "status": "rewriting"})
            rewritten = await rewrite_query(support_model, query)
            if rewritten:
                try:
                    hits = await search.query(
                        rewritten,
                        sections=state.get("sections"),
                        areas=state.get("areas"),
                    )
                except SearchUnavailable as exc:
                    logger.error("поиск недоступен после коррекции: %s", exc)
                    return {
                        "retrieved": [],
                        "followups": [],
                        "retrieval_failed": True,
                    }
                logger.info("коррекция запроса дала %d источников", len(hits))

        items = [
            to_citation(chunk, n, score).to_dict()
            for n, (chunk, score) in enumerate(hits, 1)
        ]

        if grading and items and support_model is not None:
            # Отсев фрагментов моделью. Смысл — не скорость (грейдинг её
            # ухудшает), а честный отказ на вопросах, которые проходят порог
            # близости, но ответа в базе не имеют. Сбой грейдера возвращает
            # None и означает «оставить выдачу как есть»: молча урезать
            # источники из-за неудачи служебной модели нельзя.
            emit("trace", {"stage": "retrieve", "status": "grading"})
            graded = await grade_chunks(
                support_model,
                query,
                items,
                keep_partial=grading_keep_partial,
            )
            if graded is not None:
                items = [dict(item, n=n) for n, item in enumerate(graded, 1)]

        emit("sources", {"items": items})

        followups = build_followups(items)
        if followups:
            emit("followups", {"items": followups})
        return {"retrieved": items, "followups": followups}

    async def condense_context(state: AgentState):
        if not context_manager:
            return {}
        try:
            summary = await context_manager.condense(
                state["messages"],
                existing_summary=state.get("summary", ""),
            )
        except MODEL_ERRORS as exc:
            # Без сжатия история просто останется длинной — это хуже, чем
            # сжатая, но лучше, чем упавший запрос.
            logger.warning("история не сжата: %s: %s", type(exc).__name__, exc)
            return {}
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

        if not worth_extracting(state["messages"]):
            # Вопрос по регламенту сведений о человеке не содержит, и вызов
            # модели здесь только тратит время после уже готового ответа.
            logger.debug("извлечение памяти пропущено: личных сведений нет")
            return {}

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

        try:
            result = await support_model.ainvoke([SystemMessage(content=prompt)])
        except MODEL_ERRORS as exc:
            # Память — улучшение, а не условие работоспособности. Ответ
            # пользователю уже отдан, и ронять запрос из-за неудачи записи
            # памяти значит терять готовый ответ.
            logger.warning("память не обновлена: %s: %s", type(exc).__name__, exc)
            return {}

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
