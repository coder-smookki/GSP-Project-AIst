"""Тесты структурного разбора таблиц в .docx: рендер в Markdown.

Файлы .docx собираются минимальными zip-архивами с одним
`word/document.xml` — парсер конвейера читает только этот член архива.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from infrastructure.ingest.config import DocxConfig
from infrastructure.ingest.docx_parse import DocxParser
from infrastructure.ingest.html_parse import build_document


def _make_docx(path: Path, body: str) -> Path:
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>'
        f"{body}"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return path


def _table_body() -> str:
    cells = [
        "Наименование",
        "Класс",
        "Расстояние, м",
        "Жилое здание",
        "I-II",
        "6",
        "Склад",
        "IV",
        "12",
    ]
    rows = []
    for offset in range(0, len(cells), 3):
        tcs = "".join(
            f"<w:tc><w:p><w:r><w:t>{cell}</w:t></w:r></w:p></w:tc>"
            for cell in cells[offset : offset + 3]
        )
        rows.append(f"<w:tr>{tcs}</w:tr>")
    return f"<w:tbl>{''.join(rows)}</w:tbl>"


def _p(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _blocks_from(path: Path) -> list[tuple[str, str]]:
    return DocxParser()._extract_blocks(path)


def test_docx_table_renders_as_markdown(tmp_path: Path) -> None:
    path = _make_docx(tmp_path / "doc.docx", f"{_p('5. Приложение 1')}{_table_body()}")
    assert _blocks_from(path) == [
        ("h2", "5. Приложение 1"),
        (
            "table",
            "| Наименование | Класс | Расстояние, м |\n"
            "| --- | --- | --- |\n"
            "| Жилое здание | I-II | 6 |\n"
            "| Склад | IV | 12 |",
        ),
    ]


def test_docx_keeps_order_around_table(tmp_path: Path) -> None:
    body = f"{_p('Перед таблицей.')}{_table_body()}{_p('После таблицы.')}"
    path = _make_docx(tmp_path / "doc.docx", body)
    tags = [tag for tag, _ in _blocks_from(path)]
    assert tags == ["p", "table", "p"]


def test_docx_disabled_keeps_flat_cells(tmp_path: Path) -> None:
    path = _make_docx(tmp_path / "doc.docx", f"{_p('5. Приложение 1')}{_table_body()}")
    blocks = DocxParser(
        DocxConfig(render_tables_as_markdown=False)
    )._extract_blocks(path)
    assert blocks[1:] == [
        ("td", cell)
        for cell in (
            "Наименование",
            "Класс",
            "Расстояние, м",
            "Жилое здание",
            "I-II",
            "6",
            "Склад",
            "IV",
            "12",
        )
    ]


def test_docx_table_atom_not_merged_with_clause(tmp_path: Path) -> None:
    body = (
        f"{_p('4.1. Предельный срок поставки не должен превышать 30 дней.')}"
        f"{_table_body()}"
    )
    path = _make_docx(tmp_path / "doc.docx", body)
    document = build_document(
        _blocks_from(path),
        doc_id="doc.docx",
        doc_code="ДОК-1",
        doc_title="Регламент",
        source="files",
    )
    atoms = document.atoms()
    assert len(atoms) == 2
    clause, table = atoms
    assert clause.clause == "4.1"
    assert clause.is_table_region is False
    assert table.clause is None
    assert table.is_table_region is True
    assert table.text.startswith("| Наименование")


def test_docx_heading_around_table_still_detected(tmp_path: Path) -> None:
    body = (
        f"{_p('3. Общие требования')}"
        f"{_p('3.1. Требования настоящего Регламента распространяются на все '
             'подразделения.')}"
        f"{_table_body()}"
    )
    path = _make_docx(tmp_path / "doc.docx", body)
    document = build_document(
        _blocks_from(path),
        doc_id="doc.docx",
        doc_code="ДОК-1",
        doc_title="Регламент",
        source="files",
    )
    sections = document.sections
    assert sections[0].no == "3"
    assert sections[0].atoms[1].is_table_region is True