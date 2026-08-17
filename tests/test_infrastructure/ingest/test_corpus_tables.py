"""Тесты табличного корпуса и золотого набора для A/B-замера.

Каждая цитата золотого набора должна находиться в чанках соответствующего
документа в обеих реализациях (Markdown и уплощённой) — иначе A/B-сравнение
сравнивало бы разную форму текста, а не ранг поиска.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.golden import chunk_contains
from infrastructure.ingest.chunker import chunk_document
from infrastructure.ingest.config import ParsingConfig
from infrastructure.ingest.html_parse import HtmlParser

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
CORPUS_DIR = FIXTURES / "corpus_tables"
GOLDEN_PATH = FIXTURES / "golden" / "tables_questions.json"


def _corpus() -> dict[str, dict]:
    articles: dict[str, dict] = {}
    for path in sorted(CORPUS_DIR.glob("*.json")):
        article = json.loads(path.read_text(encoding="utf-8"))["data"]
        articles[str(article["id"])] = article
    return articles


def _chunks_by_doc(render_markdown: bool) -> dict[str, list]:
    from infrastructure.ingest.models import IngestChunk

    by_doc: dict[str, list[IngestChunk]] = {}
    config = ParsingConfig(render_tables_as_markdown=render_markdown)
    for doc_id, article in _corpus().items():
        document = HtmlParser(config).parse_article(article)
        by_doc[doc_id] = chunk_document(document)
    return by_doc


def test_golden_quotes_present_in_both_variants() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    markdown = _chunks_by_doc(True)
    flat = _chunks_by_doc(False)

    for item in golden["answerable"]:
        assert chunk_contains(
            item["quote"], " ".join(c.text for c in markdown[item["doc_id"]])
        ), f"цитата не найдена в Markdown-чанках: {item['question']}"
        assert chunk_contains(
            item["quote"], " ".join(c.text for c in flat[item["doc_id"]])
        ), f"цитата не найдена в уплощённых чанках: {item['question']}"


def test_markdown_variant_keeps_table_own_chunk() -> None:
    article = _corpus()["90101"]
    markdown = chunk_document(HtmlParser(ParsingConfig()).parse_article(article))
    flat = chunk_document(
        HtmlParser(ParsingConfig(render_tables_as_markdown=False)).parse_article(article)
    )

    md_table = [c for c in markdown if "| Объект защиты" in c.text]
    assert md_table, "нет отдельного Markdown-чанка таблицы"
    assert md_table[0].is_table_region

    flat_table = [c for c in flat if "Открытый склад горючих материалов" in c.text]
    assert flat_table
    assert "Расстояния принимаются в зависимости" in flat_table[0].text