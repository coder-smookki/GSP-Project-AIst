"""Тесты структурного разбора таблиц в HTML: рендер в Markdown.

Фокус — блоки `("table", markdown)` вместо прежнего уплощения ячеек
в отдельные `td`-блоки, а также флаг `render_tables_as_markdown=False`,
сохраняющий прежнее поведение.
"""

from __future__ import annotations

from infrastructure.ingest.config import ParsingConfig
from infrastructure.ingest.html_parse import (
    HtmlParser,
    extract_blocks,
    parse_article,
)


def _html_table() -> str:
    return (
        "<table>"
        "<tr><th>Наименование</th><th>Класс</th><th>Расстояние, м</th></tr>"
        "<tr><td>Жилое здание</td><td>I-II</td><td>6</td></tr>"
        "<tr><td>Склад</td><td>IV</td><td>12</td></tr>"
        "</table>"
    )


def test_extract_blocks_renders_table_as_markdown() -> None:
    blocks = extract_blocks(f"<h2>5. Приложение 1</h2>{_html_table()}")
    assert blocks == [
        ("h2", "5. Приложение 1"),
        (
            "table",
            "| Наименование | Класс | Расстояние, м |\n"
            "| --- | --- | --- |\n"
            "| Жилое здание | I-II | 6 |\n"
            "| Склад | IV | 12 |",
        ),
    ]


def test_extract_blocks_keeps_order_around_table() -> None:
    html = f"<p>Перед таблицей.</p>{_html_table()}<p>После таблицы.</p>"
    tags = [tag for tag, _ in extract_blocks(html)]
    assert tags == ["p", "table", "p"]


def test_extract_blocks_disabled_keeps_flat_cells() -> None:
    parser = HtmlParser(ParsingConfig(render_tables_as_markdown=False))
    blocks = parser.extract_blocks(f"<h2>5. Приложение 1</h2>{_html_table()}")
    assert blocks[1:] == [
        ("td", "Наименование"),
        ("td", "Класс"),
        ("td", "Расстояние, м"),
        ("td", "Жилое здание"),
        ("td", "I-II"),
        ("td", "6"),
        ("td", "Склад"),
        ("td", "IV"),
        ("td", "12"),
    ]


def test_extract_blocks_handles_paragraphs_inside_cells() -> None:
    html = (
        "<table><tr><th>Показатель</th><th>Значение</th></tr>"
        "<tr><td><p>Первый абзац</p><p>Второй абзац</p></td><td>10</td></tr></table>"
    )
    blocks = extract_blocks(html)
    expected = (
        "| Показатель | Значение |\n"
        "| --- | --- |\n"
        "| Первый абзац Второй абзац | 10 |"
    )
    assert blocks == [("table", expected)]


def test_parse_article_table_atoms_are_table_region_without_clause() -> None:
    article = {
        "id": "9001",
        "name": "ДОК-1 Таблицы",
        "content": (
            "<h2 id='section-5'>5. Приложение 1</h2>"
            "<p>4.1. Предельный срок поставки не должен превышать 30 дней.</p>"
            f"{_html_table()}"
        ),
    }
    document = parse_article(article)
    atoms = document.atoms()
    assert len(atoms) == 2
    clause, table = atoms
    assert clause.clause == "4.1"
    assert clause.is_table_region is False
    assert table.clause is None
    assert table.is_table_region is True
    assert table.text.startswith("| Наименование")