from __future__ import annotations

import os

from infrastructure.orchestration.ingest_flow import ingest_flow


def _deployment_description() -> str:
    """Человекочитаемая сводка конфигурации раннера для страницы deployment в UI."""
    query = os.environ.get("EMBEDDING_QUERY_INSTRUCTION")
    lines = [
        "**Конфигурация раннера**",
        "",
        f"- Источник: `{os.environ.get('AIST_SOURCE_KIND', 'files')}`"
        f" (каталог `{os.environ.get('AIST_CORPUS_DIR') or 'данные из настроек'}`)",
        f"- Поколение коллекции: `v{os.environ.get('AIST_GENERATION', '1')}`",
        f"- Эмбеддер: `{os.environ.get('EMBEDDING_MODEL') or 'из настроек'}`"
        f" (`{os.environ.get('EMBEDDING_PROVIDER', 'ollama-local')}`)",
        f"- Query-инструкция: `{query}`" if query else "- Query-инструкция: пусто",
        f"- Dry-run: `{os.environ.get('AIST_DRY_RUN', '0')}`;"
        f" полная переиндексация: `{os.environ.get('AIST_FULL', '0')}`;"
        f" переключение алиаса: `{os.environ.get('AIST_SWITCH_ALIAS', '0')}`",
        f"- QA-индексация: `{os.environ.get('AIST_QA_INDEX', '0')}`"
        f" (модель `{os.environ.get('LLM_MODEL') or 'из настроек'}`)",
        f"- Cron: `{os.environ.get('AIST_CRON') or 'нет'}`;"
        f" интервал: `{os.environ.get('AIST_INTERVAL') or 'нет'}`",
    ]
    return "\n".join(lines)


def _deployment_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {"description": _deployment_description()}
    cron = os.environ.get("AIST_CRON")
    if cron:
        kwargs["cron"] = cron
    interval = os.environ.get("AIST_INTERVAL")
    if interval:
        kwargs["interval"] = int(interval)
    section = os.environ.get("AIST_SECTION_ID")
    limit = os.environ.get("AIST_LIMIT")
    kwargs["parameters"] = {
        "source_kind": os.environ.get("AIST_SOURCE_KIND", "files"),
        "corpus_dir": os.environ.get("AIST_CORPUS_DIR") or None,
        "section_id": int(section) if section else None,
        "generation": int(os.environ.get("AIST_GENERATION", "1")),
        "limit": int(limit) if limit else None,
        "dry_run": os.environ.get("AIST_DRY_RUN", "0") == "1",
        "full": os.environ.get("AIST_FULL", "0") == "1",
        "qa_index": os.environ.get("AIST_QA_INDEX", "0") == "1",
        "switch_alias": os.environ.get("AIST_SWITCH_ALIAS", "0") == "1",
        "sync_tag": os.environ.get("AIST_SYNC_TAG") or None,
        # Эмбеддинги: берём уже существующие EMBEDDING_* (они же в .env).
        # Значения попадают в deployment, в UI их видно и можно переопределить.
        "embedding_model": os.environ.get("EMBEDDING_MODEL") or None,
        "embedding_provider": os.environ.get("EMBEDDING_PROVIDER") or None,
        "embedding_query_instruction": os.environ.get(
            "EMBEDDING_QUERY_INSTRUCTION"
        ) or None,
        "llm_model": os.environ.get("LLM_MODEL") or None,
        "llm_provider": os.environ.get("LLM_PROVIDER") or None,
    }
    return kwargs


def main() -> None:
    serve = getattr(ingest_flow, "serve", None)
    if serve is None:
        raise RuntimeError(
            "Prefect не установлен: serve недоступен; "
            "установите группу orchestration (uv sync --extra orchestration)"
        )
    serve(
        name=os.environ.get("AIST_DEPLOYMENT_NAME", "aist-ingest"),
        tags=["aist", "ingest"],
        **_deployment_kwargs(),
    )


if __name__ == "__main__":
    main()