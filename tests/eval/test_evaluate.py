"""Тесты сопоставления цитаты с найденным фрагментом.

Это ядро замера: если сопоставление ошибается, все цифры качества
становятся выдумкой — причём правдоподобной, что хуже явной поломки.
Поэтому проверяются обе стороны: и что верное попадание не теряется от
различий в регистре и пунктуации, и что случайное совпадение общих слов
попаданием не считается.
"""

from __future__ import annotations

import pytest

from eval.golden import chunk_contains, longest_common_run, normalize
from eval.metrics import percentile


class TestNormalize:
    """Приведение текста к словам."""

    def test_strips_punctuation_and_case(self) -> None:
        """Регистр и пунктуация не влияют на сравнение."""
        assert normalize("Срок — 15 (пятнадцать) дней.") == [
            "срок",
            "15",
            "пятнадцать",
            "дней",
        ]

    def test_collapses_yo(self) -> None:
        """«ё» и «е» считаются одной буквой.

        В регламентах написание непоследовательно даже внутри одного
        документа, и без приведения верные попадания терялись бы.
        """
        assert normalize("учёт") == normalize("учет")


class TestLongestCommonRun:
    """Поиск самой длинной общей цепочки слов."""

    def test_finds_contiguous_run(self) -> None:
        """Считается именно цепочка подряд, а не общее число слов."""
        assert longest_common_run(["а", "б", "в"], ["х", "а", "б", "в", "у"]) == 3

    def test_gap_breaks_the_run(self) -> None:
        """Разрыв обрывает цепочку."""
        assert longest_common_run(["а", "б", "в"], ["а", "б", "х", "в"]) == 2

    def test_empty_input_is_zero(self) -> None:
        """Пустой вход даёт ноль, а не ошибку."""
        assert longest_common_run([], ["а"]) == 0
        assert longest_common_run(["а"], []) == 0


class TestChunkContains:
    """Признак «фрагмент содержит эталонную цитату»."""

    def test_matches_despite_formatting(self) -> None:
        """Цитата находится, хотя оформление отличается.

        Проиндексированный текст проходит через разбор HTML и склейку
        строк, поэтому дословного совпадения байтов не бывает почти
        никогда.
        """
        quote = (
            "Срок проведения внутреннего расследования не может превышать "
            "15 календарных дней."
        )
        chunk = (
            "4.10. Срок проведения\nвнутреннего расследования, установленный "
            "распорядительным документом подрядной организации, НЕ МОЖЕТ "
            "ПРЕВЫШАТЬ 15 календарных дней!"
        )
        assert chunk_contains(quote, chunk)

    def test_rejects_topical_but_different_text(self) -> None:
        """Текст по той же теме, но без цитаты, попаданием не считается.

        Самая опасная ошибка замера: фрагмент выглядит подходящим, общие
        слова совпадают, а искомых сведений в нём нет.
        """
        quote = (
            "Срок проведения внутреннего расследования не может превышать "
            "15 календарных дней."
        )
        chunk = (
            "Работодатель обязан обеспечить проведение расследования "
            "происшествия и оформление его результатов в установленном "
            "порядке с участием представителей организации."
        )
        assert not chunk_contains(quote, chunk)

    def test_short_quote_requires_full_match(self) -> None:
        """Короткая цитата должна совпасть целиком."""
        assert chunk_contains("15 календарных дней", "срок — 15 календарных дней")
        assert not chunk_contains("15 календарных дней", "срок — 10 календарных дней")

    @pytest.mark.parametrize("empty", ["", "   ", "!!!"])
    def test_empty_quote_never_matches(self, empty: str) -> None:
        """Пустая цитата не даёт ложного попадания.

        Иначе вопрос с незаполненной цитатой засчитывался бы найденным на
        любом фрагменте и завышал качество.
        """
        assert not chunk_contains(empty, "любой текст")


class TestPercentile:
    """Квантили времени."""

    def test_median_and_edges(self) -> None:
        """Квантиль берётся из значений выборки."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 0.5) == 3.0
        assert percentile(values, 1.0) == 5.0

    def test_empty_is_zero(self) -> None:
        """Пустая выборка не роняет отчёт."""
        assert percentile([], 0.95) == 0.0
