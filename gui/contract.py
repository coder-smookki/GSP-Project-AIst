def build_input(
    content: str,
    sections: list[str] | None = None,
    areas: list[str] | None = None,
) -> dict:
    """Собирает полезную нагрузку для ``client.runs.stream``.

    Args:
        content: Текст сообщения пользователя.
        sections: Типы разделов документа — служебная ось (назначение,
            термины, процедура). Из интерфейса не задаётся, оставлена для
            отладки и тестов.
        areas: Предметные области знаний, выбранные пользователем. Пустой
            список означает поиск по всей базе, а не по пустому множеству.

    Returns:
        dict: Нагрузка запуска графа.
    """
    payload: dict = {"messages": [{"type": "human", "content": content}]}
    if sections:
        payload["sections"] = sections
    if areas:
        payload["areas"] = areas
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
    """Выбор финального текста ответа агента.

    Порядок: сообщение об ошибке -> последний завершённый AI-чанк узла
    assistant -> самый длинный partial-chunk (fallback на огрызок при прерывании).
    """
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
