"""Юнит-тесты чистых функций замера извлечения чисел из таблиц."""

from __future__ import annotations

from eval.table_extraction import build_prompt, expected_number, extraction_correct


def test_expected_number_extracts_first_integer() -> None:
    assert expected_number("12 метров") == 12
    assert expected_number("500 м2") == 500
    assert expected_number("не реже одного раза в полгода") is None


def test_extraction_correct_requires_exact_number() -> None:
    assert extraction_correct("Расстояние 6 метров", 6)
    assert not extraction_correct("Расстояние 16 метров", 6)
    assert not extraction_correct("НЕТ ДАННЫХ", 12)
    assert extraction_correct("Верно, это 1200 м2", 1200)


def test_build_prompt_contains_question_and_chunks() -> None:
    messages = build_prompt("Вопрос?", ["Первый чанк", "Второй чанк"])
    text = " ".join(str(message.content) for message in messages)
    assert "Вопрос?" in text
    assert "Первый чанк" in text
    assert "Фрагмент 1" in text
    assert "Фрагмент 2" in text


def test_build_prompt_empty_chunks() -> None:
    messages = build_prompt("Вопрос?", [])
    text = " ".join(str(message.content) for message in messages)
    assert "Вопрос?" in text