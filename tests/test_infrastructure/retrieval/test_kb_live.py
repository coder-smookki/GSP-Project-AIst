"""Живая загрузка по API базы знаний.

Требует настроенного доступа (`KB_EMAIL`, `KB_PASSWORD`) и сети, поэтому
помечена маркером ``integration`` и в обычный прогон не входит. Без учётных
данных пропускается, а не падает: у большинства разработчиков доступа нет.

Запуск::

    uv run pytest -m integration tests/test_infrastructure/retrieval/test_kb_live.py
"""

from __future__ import annotations

import pytest

from infrastructure.config.settings import get_settings
from infrastructure.retrieval.kb_client import (
    KnowledgeBaseClient,
    KnowledgeBaseConfig,
)

pytestmark = pytest.mark.integration


def _config() -> KnowledgeBaseConfig:
    """Собирает конфигурацию доступа из настроек приложения.

    Returns:
        KnowledgeBaseConfig: Параметры подключения.
    """
    settings = get_settings()
    if not settings.kb_email or not settings.kb_password:
        pytest.skip("доступ к базе знаний не настроен: KB_EMAIL / KB_PASSWORD")
    return KnowledgeBaseConfig(
        base_url=settings.kb_base_url,
        email=settings.kb_email,
        password=settings.kb_password,
        section_id=settings.kb_section_id,
        token_name=settings.kb_token_name,
        per_page=settings.kb_per_page,
        max_retries=settings.kb_max_retries,
    )


async def test_login_and_list_articles() -> None:
    """Вход проходит, раздел кейса отдаёт непустой перечень статей."""
    async with KnowledgeBaseClient(_config()) as client:
        articles = await client.list_section_articles()

    assert articles, "раздел кейса не должен быть пустым"
    assert all("id" in item for item in articles)


async def test_article_content_is_html() -> None:
    """Статья приходит с непустым содержимым в HTML.

    Разбор рассчитан на разметку: пустое или неразмеченное содержимое
    означает, что дальше по конвейеру пойдёт документ без структуры.
    """
    config = _config()
    async with KnowledgeBaseClient(config) as client:
        articles = await client.list_section_articles()
        article = await client.get_article(articles[0]["id"])

    content = str(article.get("content") or "")
    assert content.strip(), "содержимое статьи пустое"
    assert "<" in content, "содержимое не похоже на HTML"


async def test_pipeline_parses_a_live_article() -> None:
    """Живая статья проходит разбор и даёт фрагменты.

    Это и есть настоящая проверка связки: доступ, формат ответа и разбор
    сходятся только вместе.
    """
    from infrastructure.ingest.chunker import chunk_document
    from infrastructure.ingest.pipeline import IngestPipeline
    from infrastructure.retrieval.source import RawDocument

    config = _config()
    async with KnowledgeBaseClient(config) as client:
        articles = await client.list_section_articles()
        article = await client.get_article(articles[0]["id"])

    raw = RawDocument(
        doc_id=str(article["id"]),
        title=str(article.get("name") or ""),
        kind="html",
        content=str(article.get("content") or ""),
        section_id=str(article.get("section_id") or ""),
        updated_at=str(article.get("updated_at") or "") or None,
    )
    document = IngestPipeline().parse_raw(raw)
    chunks = chunk_document(document)

    assert document.sections, "разбор не нашёл ни одного раздела"
    assert chunks, "нарезка не дала ни одного фрагмента"
