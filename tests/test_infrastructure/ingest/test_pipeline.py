"""Тесты офлайн-конвейера: разбор HTML, деанонимизация, чанкинг.

Фикстура `tests/fixtures/kb_articles.json` — **синтетическая**: она
воспроизводит разметку и все известные дефекты источника (срезанная первая
цифра пункта, склейка двух пунктов в один абзац, разрыв пункта на несколько
абзацев, строки словаря в обоих порядках, одинаковая метка с разным смыслом
в разных документах), но не содержит документов заказчика — им не место в
репозитории.

Тесты работают полностью офлайн и не требуют ни сети, ни моделей.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infrastructure.ingest.chunker import MAX_CHARS, chunk_document
from infrastructure.ingest.deanon import MASK, build_glossary, deanonymize, restore_text
from infrastructure.ingest.html_parse import extract_blocks, parse_article
from infrastructure.ingest.models import SectionType
from infrastructure.ingest.patterns import (
    CLAUSE_HEAD,
    PLACEHOLDER,
    is_false_clause,
    restore_clause,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "kb_articles.json"


@pytest.fixture(scope="module")
def articles() -> dict[str, dict]:
    """Загружает синтетические статьи, повторяющие разметку источника.

    Returns:
        dict[str, dict]: Отображение ``идентификатор статьи -> объект статьи``.
    """
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --------------------------------------------------------------- регулярки


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4.1.2. Совершенствование навыков", "4.1.2"),
        ("1.2. Требования настоящего Регламента", "1.2"),
        ("10. Общие требования", "10"),
    ],
)
def test_clause_head_matches_real_numbering(text: str, expected: str) -> None:
    """Якорная регулярка находит номер пункта в начале абзаца."""
    match = CLAUSE_HEAD.match(text)
    assert match is not None
    assert match.group("clause") == expected


def test_clause_head_ignores_dates() -> None:
    """Дата в начале абзаца не принимается за номер пункта.

    Это главная ловушка корпуса: «12.02.1998 № 28-ФЗ» в тексте о законах.
    """
    text = "12.02.1998 № 28-ФЗ «О гражданской обороне»"
    match = CLAUSE_HEAD.match(text)
    assert match is None or is_false_clause(text, match)


def test_restore_clause_uses_section_number() -> None:
    """Срезанный конвертером номер достраивается номером раздела."""
    assert restore_clause("1", truncated=True, section_no="4") == "4.1"
    assert restore_clause("2.1", truncated=True, section_no="4") == "4.2.1"
    assert restore_clause("4.2", truncated=False, section_no="4") == "4.2"


def test_restore_clause_without_section_keeps_value() -> None:
    """Без известного раздела номер не выдумывается."""
    assert restore_clause("1", truncated=True, section_no="") == "1"


# ------------------------------------------------------------------ разбор


def test_extract_blocks_drops_navigation() -> None:
    """Оглавление внутри ``<nav>`` не попадает в текст документа."""
    html = (
        "<main><nav><ol><li><a href='#s1'>1. Назначение</a></li></ol></nav>"
        "<h2>1. Назначение</h2><p>1.1. Текст пункта.</p></main>"
    )
    blocks = extract_blocks(html)
    assert ("h2", "1. Назначение") in blocks
    assert all("Назначение" not in text for tag, text in blocks if tag == "li")


def test_parse_article_builds_sections_and_clauses(articles: dict[str, dict]) -> None:
    """Разбор статьи даёт разделы с восстановленными номерами пунктов."""
    document = parse_article(articles["1002"])

    assert document.doc_id == "1002"
    assert document.sections, "разделы не найдены"

    titles = [section.title for section in document.sections]
    assert any("Назначение" in title for title in titles)
    assert any("Термины" in title for title in titles)

    clauses = {atom.clause for atom in document.atoms() if atom.clause}
    # Первый пункт первого раздела приходит как «.1.» и должен стать «1.1».
    assert "1.1" in clauses


def test_parse_article_marks_terms_section(articles: dict[str, dict]) -> None:
    """Раздел терминов классифицируется — на нём строится словарь."""
    document = parse_article(articles["1002"])
    assert any(s.type is SectionType.TERMINY for s in document.sections)


def test_parse_article_rejects_empty_content() -> None:
    """Пустое содержимое отвергается явной ошибкой, а не молча."""
    with pytest.raises(ValueError, match="пустое содержимое"):
        parse_article({"id": 1, "name": "x", "content": "   "})


def test_parse_article_rejects_missing_id() -> None:
    """Статья без идентификатора отвергается."""
    with pytest.raises(ValueError, match="идентификатор"):
        parse_article({"content": "<p>текст</p>"})


def test_atoms_never_exceed_limit(articles: dict[str, dict]) -> None:
    """Ни один атом не превышает предела: гигантские абзацы режутся."""
    for article in articles.values():
        for atom in parse_article(article).atoms():
            assert len(atom.text) <= 1500


# --------------------------------------------------------- деанонимизация


def test_glossary_is_per_document(articles: dict[str, dict]) -> None:
    """Один и тот же плейсхолдер трактуется по-разному в разных документах.

    Это главное свойство корпуса и причина, по которой словарь пофайловый:
    глобальный словарь подставил бы уверенное враньё.
    """
    first = build_glossary(deanonymize(parse_article(articles["1002"])))
    second = build_glossary(deanonymize(parse_article(articles["1001"])))

    common = set(first) & set(second)
    assert common, "нет общих плейсхолдеров — проверить фикстуру"
    assert any(first[key] != second[key] for key in common), (
        "во всех общих плейсхолдерах расшифровки совпали: "
        "пофайловый словарь потерял бы смысл"
    )


def test_restore_text_masks_unknown_placeholder() -> None:
    """Нераспознанная метка маскируется и не утекает пользователю."""
    text, restored, unresolved = restore_text("Ответственный — [ОРГАНИЗАЦИЯ_99].", {})
    assert MASK in text
    assert not PLACEHOLDER.search(text)
    assert restored == []
    assert unresolved == ["[ОРГАНИЗАЦИЯ_99]"]


def test_restore_text_substitutes_known_placeholder() -> None:
    """Известная метка заменяется расшифровкой и попадает в aliases."""
    glossary = {"[ОРГАНИЗАЦИЯ_18]": "пожарная безопасность"}
    text, restored, unresolved = restore_text("Требования [ОРГАНИЗАЦИЯ_18].", glossary)
    assert "пожарная безопасность" in text
    assert restored == ["пожарная безопасность"]
    assert unresolved == []


# ------------------------------------------------------------------ чанкинг


def test_chunks_are_bounded_and_unique(articles: dict[str, dict]) -> None:
    """Чанки укладываются в предел, идентификаторы не повторяются."""
    seen: set[str] = set()
    for article in articles.values():
        document = deanonymize(parse_article(article))
        chunks = chunk_document(document)
        assert chunks, "конвейер не дал ни одного чанка"
        for chunk in chunks:
            assert len(chunk.text) <= MAX_CHARS
            assert chunk.chunk_id not in seen
            seen.add(chunk.chunk_id)


def test_chunks_never_leak_placeholders(articles: dict[str, dict]) -> None:
    """В индексируемом тексте не остаётся ни одной метки анонимизации."""
    for article in articles.values():
        document = deanonymize(parse_article(article))
        for chunk in chunk_document(document):
            assert not PLACEHOLDER.search(chunk.text), (
                f"метка утекла в чанк {chunk.chunk_id}"
            )


def test_chunk_ids_are_stable_across_runs(articles: dict[str, dict]) -> None:
    """Повторный прогон на тех же данных даёт те же идентификаторы.

    От стабильности зависит золотой набор eval: он ссылается на chunk_id.
    """
    article = articles["1002"]
    first = [c.chunk_id for c in chunk_document(deanonymize(parse_article(article)))]
    second = [c.chunk_id for c in chunk_document(deanonymize(parse_article(article)))]
    assert first == second


def test_point_id_is_deterministic_uuid(articles: dict[str, dict]) -> None:
    """Идентификатор точки Qdrant детерминирован и имеет форму UUID."""
    chunk = chunk_document(deanonymize(parse_article(articles["1002"])))[0]
    point_id = chunk.point_id()
    assert point_id == chunk.point_id()
    assert len(point_id) == 36
    assert point_id.count("-") == 4


def test_table_of_contents_is_not_indexed() -> None:
    """Раздел «Содержание» в индекс не попадает."""
    article = {
        "id": 7,
        "name": "Тест",
        "content": (
            "<main><h2>1. Содержание</h2><p>1.1. Оглавление документа целиком.</p>"
            "<h2>2. Общие требования</h2>"
            "<p>2.1. Настоящий документ устанавливает требования безопасности "
            "для работников подразделения при выполнении работ.</p></main>"
        ),
    }
    chunks = chunk_document(deanonymize(parse_article(article)))
    assert chunks
    assert all(c.section_type is not SectionType.SODERZHANIE for c in chunks)


def test_breadcrumb_replaces_overlap(articles: dict[str, dict]) -> None:
    """Каждый чанк начинается со своего навигационного пути."""
    document = deanonymize(parse_article(articles["1002"]))
    for chunk in chunk_document(document):
        assert chunk.text.startswith(chunk.breadcrumb)
        assert chunk.doc_code in chunk.breadcrumb


def test_chunker_rejects_inconsistent_limits(articles: dict[str, dict]) -> None:
    """Противоречивые пределы отвергаются, а не приводят к тихому переполнению."""
    document = deanonymize(parse_article(articles["1002"]))
    with pytest.raises(ValueError, match="не может превышать"):
        chunk_document(document, target_chars=2000, max_chars=1000)
