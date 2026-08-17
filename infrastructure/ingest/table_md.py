from __future__ import annotations

import re
from collections.abc import Sequence

_WHITESPACE: re.Pattern[str] = re.compile(r"\s+")

_PIPE: re.Pattern[str] = re.compile(r"\|")


def _cell(text: str) -> str:
    value = _WHITESPACE.sub(" ", text).strip()
    return _PIPE.sub(r"\\|", value)


def render_table(rows: Sequence[Sequence[str]]) -> str:
    """Превращает грид ячеек в Markdown-таблицу GitHub-стиля.

    Первая строка используется как заголовок, вторая — разделитель,
    остальные — тело. Пустые строки и строки без текста отбрасываются;
    при пустом входе возвращается пустая строка. Переносы строк внутри
    ячейки схлопываются в пробелы, вертикальная черта экранируется.
    """
    normalized: list[list[str]] = []
    for row in rows:
        cells = [_cell(cell) for cell in row]
        if any(cells):
            normalized.append(cells)
    if not normalized:
        return ""

    width = max(len(row) for row in normalized)
    padded = [row + [""] * (width - len(row)) for row in normalized]

    lines: list[str] = [_row(padded[0]), _row(["---"] * width)]
    lines.extend(_row(row) for row in padded[1:])
    return "\n".join(lines)


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"
