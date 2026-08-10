"""Тесты сверки чисел ответа с контекстом.

Проверяются обе стороны. Пропущенная выдумка оставляет пользователя с
несуществующим требованием регламента. Ложное срабатывание портит верный
ответ предупреждением и учит не доверять предупреждениям вообще, поэтому
случаи «число есть, но в другой форме» проверяются так же придирчиво.
"""

from __future__ import annotations

import pytest

from agent.numbers import WARNING_TEMPLATE, annotate, unsupported_numbers


class TestUnsupportedNumbers:
    """Поиск неподтверждённых чисел."""

    def test_detects_invented_deadline(self) -> None:
        """Срок, которого нет в контексте, находится.

        Замеренный случай: модель написала «в течение 24 часов [1]», хотя
        такого срока в корпусе нет ни в одном документе.
        """
        answer = "Регистрация проводится в течение 24 часов с момента получения [1]."
        context = "Микротравма — ссадины, кровоподтёки и другие повреждения."
        assert unsupported_numbers(answer, context) == ["24 час"]

    def test_bare_number_match_is_not_enough(self) -> None:
        """Совпадения одной цифры мало: она есть в номерах пунктов.

        «5 рабочих дней» не должно проходить проверку только потому, что в
        документе существует пункт 5.2.
        """
        answer = "Комиссия создаётся в течение 5 рабочих дней."
        context = "5.2. Мероприятия разрабатываются в отношении всех рисков."
        assert unsupported_numbers(answer, context) == ["5 раб"]

    def test_supported_number_passes(self) -> None:
        """Подтверждённое число предупреждения не вызывает."""
        answer = "Срок не может превышать 15 календарных дней [4]."
        context = "Срок расследования не может превышать 15 календарных дней."
        assert unsupported_numbers(answer, context) == []

    def test_case_endings_do_not_break_match(self) -> None:
        """Падеж единицы не мешает: «24 часа» и «24 часов» — одно и то же."""
        assert unsupported_numbers("не позднее 24 часа", "в течение 24 часов") == []

    def test_citation_markers_are_ignored(self) -> None:
        """Маркеры источников числами ответа не считаются."""
        assert unsupported_numbers("Ответ по регламенту [1] и [2].", "текст") == []

    @pytest.mark.parametrize(
        ("answer", "context"), [("", "текст"), ("15 дней", ""), ("", "")]
    )
    def test_empty_input_is_safe(self, answer: str, context: str) -> None:
        """Пустой ввод не даёт ложного срабатывания."""
        assert unsupported_numbers(answer, context) == []


class TestAnnotate:
    """Пометка ответа."""

    def test_warning_added_for_invented_number(self) -> None:
        """К ответу с выдуманным числом добавляется предупреждение."""
        result = annotate("Срок — 24 часов.", "текст без сроков")
        assert WARNING_TEMPLATE.split("{")[0].strip() in result
        assert "24 часов" in result

    def test_clean_answer_is_untouched(self) -> None:
        """Подтверждённый ответ не меняется ни на символ."""
        answer = "Срок — 15 календарных дней [1]."
        assert annotate(answer, "не превышает 15 календарных дней") == answer


class TestUnitWhitelist:
    """Проверяются только величины, а не любое слово после числа.

    Замер на демо-машине дал ложные срабатывания «1 док», «1 б, 1 в»,
    «101 с, 31 или» — это куски перечислений и маркеры списка. Такие
    предупреждения бесполезны и приучают не доверять предупреждениям
    вообще, поэтому слово после числа обязано быть единицей измерения.
    """

    @pytest.mark.parametrize(
        "answer",
        [
            "Перечень: 1 документ и приложение.",
            "Пункты: 1 б) и 1 в) регламента.",
            "Срок 31 или 32 по решению комиссии.",
            "Ответ приведён в 2 абзацах текста.",
        ],
    )
    def test_non_units_are_ignored(self, answer: str) -> None:
        """Слово, не являющееся единицей, предупреждения не вызывает."""
        assert unsupported_numbers(answer, "контекст без этих чисел") == []

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("в течение 24 часов", "24 час"),
            ("в течение 5 рабочих дней", "5 раб"),
            ("не реже 1 раза в квартал", "1 раз"),
            ("не менее 2 огнетушителей на 15 человек", "15 чел"),
        ],
    )
    def test_real_units_are_checked(self, answer: str, expected: str) -> None:
        """Величины по-прежнему проверяются."""
        assert expected in unsupported_numbers(answer, "контекст без этих чисел")
