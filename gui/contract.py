from dataclasses import dataclass

#: Названия этапов графа для интерфейса (см. docs/GUI.md).
STAGE_TITLES: dict[str, str] = {
    "condense": "Сжатие истории диалога",
    "plan": "Планирование поиска",
    "search_agent": "Поиск в базе знаний",
    "finalize": "Отбор источников",
    "assistant": "Подготовка ответа",
}

#: Статусы события trace, при которых этап считается неудавшимся.
STAGE_ERROR_STATUSES = frozenset({"error", "model_unavailable", "unavailable"})

#: Итог неудавшегося этапа по статусу события.
STAGE_ERROR_DETAILS: dict[str, str] = {
    "model_unavailable": "Модель недоступна",
    "unavailable": "Сервис недоступен",
    "error": "Этап завершён с ошибкой",
}

#: Предельная длина подзапроса в названии шага.
STAGE_QUERY_LIMIT = 60


@dataclass(frozen=True)
class StageEvent:
    """Этап работы графа в том виде, в каком его показывают пользователю.

    Attributes:
        key: Ключ этапа; у веток поиска включает подзапрос, поэтому
            параллельные ветки не перетирают шаги друг друга.
        title: Заголовок шага.
        running: True, если этап только начался и ещё выполняется.
        failed: True, если этап завершён неудачей.
        detail: Короткий итог этапа; у начатого этапа пустой.
    """

    key: str
    title: str
    running: bool
    failed: bool
    detail: str


def _as_int(value: object) -> int:
    """Приводит значение полезной нагрузки к целому.

    Args:
        value: Значение произвольного типа из события.

    Returns:
        int: Число либо 0, если значение нечисловое.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _shorten(text: str, limit: int = STAGE_QUERY_LIMIT) -> str:
    """Сжимает пробелы и обрезает строку до предела.

    Args:
        text: Исходная строка.
        limit: Предельная длина результата.

    Returns:
        str: Строка не длиннее предела; обрезанная оканчивается многоточием.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _stage_detail(stage: str, status: str, payload: dict) -> str:
    """Собирает короткий итог завершённого этапа.

    Args:
        stage: Имя этапа из события.
        status: Статус события.
        payload: Полезная нагрузка события trace.

    Returns:
        str: Текст итога; для только что начатого этапа пустая строка.
    """
    if status == "start":
        return ""
    if status in STAGE_ERROR_STATUSES:
        return STAGE_ERROR_DETAILS.get(status, STAGE_ERROR_DETAILS["error"])
    if stage == "condense":
        return "История свёрнута" if payload.get("condensed") else "История не менялась"
    if stage == "plan":
        if payload.get("mode") == "chat":
            return "Поиск по базе не требуется"
        return f"Подзапросов: {_as_int(payload.get('subqueries'))}"
    if stage == "search_agent":
        found = _as_int(payload.get("retrieved"))
        if found:
            return f"Найдено фрагментов: {found}"
        if payload.get("failed"):
            return "Поиск недоступен"
        return "Ничего не найдено"
    if stage == "finalize":
        if payload.get("retrieval_failed"):
            return "Поиск по базе знаний недоступен"
        return f"Источников: {_as_int(payload.get('retrieved'))}"
    if stage == "assistant":
        return "Ответ готов"
    return ""


def parse_stage(payload: dict) -> StageEvent | None:
    """Разбирает полезную нагрузку события trace в описание этапа.

    Событие без поля stage или с незнакомым этапом показывать нечем:
    интерфейс такие события молча пропускает, а не выдумывает заголовок.

    Args:
        payload: Поле payload конверта aist при kind="trace".

    Returns:
        StageEvent | None: Описание этапа либо None, если показывать нечего.
    """
    if not isinstance(payload, dict):
        return None

    stage = str(payload.get("stage") or "").strip()
    title = STAGE_TITLES.get(stage)
    if title is None:
        return None

    status = str(payload.get("status") or "done").strip() or "done"
    key = stage
    if stage == "search_agent":
        query = _shorten(str(payload.get("query") or ""))
        key = f"{stage}:{query}"
        if query:
            title = f"{title}: {query}"

    failed = status in STAGE_ERROR_STATUSES
    if stage == "search_agent" and status != "start":
        failed = failed or (
            bool(payload.get("failed")) and _as_int(payload.get("retrieved")) == 0
        )
    if stage == "finalize" and payload.get("retrieval_failed"):
        failed = True

    return StageEvent(
        key=key,
        title=title,
        running=status == "start",
        failed=failed,
        detail=_stage_detail(stage, status, payload),
    )


def build_input(
    content: str,
    sections: list[str] | None = None,
    areas: list[str] | None = None,
    always_answer_with_db: bool = False,
) -> dict:
    """Собирает полезную нагрузку `client.runs.stream` (см. docs/GUI.md)."""
    payload: dict = {"messages": [{"type": "human", "content": content}]}
    if sections:
        payload["sections"] = sections
    if areas:
        payload["areas"] = areas
    if always_answer_with_db:
        payload["always_answer_with_db"] = always_answer_with_db
    return payload


def decode_aist_envelope(chunk: dict) -> dict | None:
    env = chunk.get("aist")
    if env is None:
        params = (chunk.get("params") or {}).get("data") or {}
        env = params.get("aist")
    return env if isinstance(env, dict) else None


def format_sources(items: list[dict]) -> str:
    lines = []
    for item in items:
        n = item.get("n")
        ref = item.get("doc_code") or ""
        clause = item.get("clause")
        if clause:
            ref = f"{ref} · п. {clause}"
        title = item.get("doc_title") or ""
        snippet = (item.get("snippet") or "").strip()

        head = f"[{n}] {ref} — {title}" if (n is not None and ref) else (title or "—")
        fields = [f"**{head}**"]
        if snippet:
            fields.append(snippet)
        lines.append("\n\n".join(fields))
    return "\n\n---\n".join(lines)


def pick_final_text(
    partials: dict[str, str],
    assistant_final: list[str] | None = None,
    error: Exception | None = None,
) -> str:
    """Выбор финального текста ответа (см. порядок в docs/GUI.md)."""
    if error is not None:
        return (
            "Ошибка обработки запроса. Если проблема повторится — "
            "проверьте доступность сервиса (aegra) и попробуйте ещё раз."
        )

    if assistant_final:
        for text in reversed(assistant_final):
            if text.strip():
                return text

    if partials:
        return max(partials.values(), key=len)

    return ""
