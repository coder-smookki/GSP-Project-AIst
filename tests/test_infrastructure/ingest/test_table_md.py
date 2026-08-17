from infrastructure.ingest.table_md import render_table


def test_render_two_rows_uses_first_as_header() -> None:
    rendered = render_table(
        [
            ["Наименование", "Класс", "Расстояние, м"],
            ["Жилое здание", "I-II", "6"],
            ["Склад", "IV", "12"],
        ]
    )
    assert rendered == (
        "| Наименование | Класс | Расстояние, м |\n"
        "| --- | --- | --- |\n"
        "| Жилое здание | I-II | 6 |\n"
        "| Склад | IV | 12 |"
    )


def test_render_single_row_gives_header_only() -> None:
    assert render_table([["Один", "Два"]]) == "| Один | Два |\n| --- | --- |"


def test_render_empty_returns_empty() -> None:
    assert render_table([]) == ""
    assert render_table([[""]]) == ""
    assert render_table([["", "  "], ["", ""]]) == ""


def test_render_pads_shorter_rows() -> None:
    rendered = render_table([["A", "B", "C"], ["x"]])
    lines = rendered.splitlines()
    assert lines[2] == "| x |  |  |"


def test_render_collapses_whitespace_and_newlines_in_cells() -> None:
    rendered = render_table([["H1", "H2"], ["a\nb", "  c   d"]])
    assert "| a b | c d |" in rendered


def test_render_escapes_pipes_in_cells() -> None:
    rendered = render_table([["H1"], ["a|b"]])
    assert "| a\\|b |" in rendered
