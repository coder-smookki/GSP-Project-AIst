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
