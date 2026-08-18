"""Тесты сборки выдачи из веток мультиагентного поиска.

Скоры разных подзапросов несопоставимы, поэтому общая сортировка по скору
может отдать все места одной ветви. Для составного вопроса это означает,
что ответ на вторую его часть не дойдёт до модели вовсе — то есть
декомпозиция обнулится ровно там, где она нужна.
"""

from __future__ import annotations

from agent.graph import _merge_branches


def _chunk(subquery: str, score: float, name: str) -> dict:
    """Готовит фрагмент ветви.

    Args:
        subquery: Подзапрос, которым фрагмент найден.
        score: Счёт в пределах своей ветви.
        name: Метка для проверки порядка.

    Returns:
        dict: Фрагмент в формате цитаты.
    """
    return {"subquery": subquery, "score": score, "chunk_id": name, "text": name}


class TestMergeBranches:
    """Раздача мест между ветвями."""

    def test_every_branch_gets_a_place(self) -> None:
        """Ветвь со слабыми скорами всё равно попадает в выдачу.

        Ключевой случай: «нужны нормы ГОСТ и как носить каски». Первый
        подзапрос совпал сильнее просто по формулировке, но ответ на второй
        обязан дойти до модели.
        """
        items = [
            _chunk("гост", 0.9, "гост-1"),
            _chunk("гост", 0.88, "гост-2"),
            _chunk("гост", 0.87, "гост-3"),
            _chunk("каски", 0.6, "каски-1"),
            _chunk("каски", 0.5, "каски-2"),
        ]
        merged = _merge_branches(items, 3)
        assert [c["chunk_id"] for c in merged] == ["гост-1", "каски-1", "гост-2"]

    def test_within_branch_order_is_by_score(self) -> None:
        """Внутри ветви порядок по счёту: там запрос один, счета сравнимы."""
        items = [
            _chunk("а", 0.3, "слабый"),
            _chunk("а", 0.9, "сильный"),
            _chunk("а", 0.6, "средний"),
        ]
        merged = _merge_branches(items, 3)
        assert [c["chunk_id"] for c in merged] == ["сильный", "средний", "слабый"]

    def test_exhausted_branch_yields_its_slots(self) -> None:
        """Исчерпанная ветвь отдаёт места остальным, предел расходуется весь."""
        items = [
            _chunk("а", 0.9, "а1"),
            _chunk("б", 0.8, "б1"),
            _chunk("б", 0.7, "б2"),
            _chunk("б", 0.6, "б3"),
        ]
        merged = _merge_branches(items, 4)
        assert len(merged) == 4
        assert "а1" in [c["chunk_id"] for c in merged]

    def test_single_branch_behaves_as_plain_sort(self) -> None:
        """Один подзапрос — обычная сортировка по счёту."""
        items = [_chunk("один", 0.2, "низ"), _chunk("один", 0.8, "верх")]
        assert [c["chunk_id"] for c in _merge_branches(items, 2)] == ["верх", "низ"]

    def test_limit_is_respected(self) -> None:
        """Предел не превышается даже при множестве ветвей."""
        items = [_chunk(f"под{i}", 0.5, f"ф{i}") for i in range(10)]
        assert len(_merge_branches(items, 5)) == 5

    def test_empty_input(self) -> None:
        """Пустой вход не роняет сборку."""
        assert _merge_branches([], 5) == []

    def test_missing_subquery_is_grouped_together(self) -> None:
        """Фрагменты без пометки ветви не теряются."""
        items = [{"score": 0.5, "chunk_id": "x"}, {"score": 0.9, "chunk_id": "y"}]
        assert [c["chunk_id"] for c in _merge_branches(items, 2)] == ["y", "x"]
