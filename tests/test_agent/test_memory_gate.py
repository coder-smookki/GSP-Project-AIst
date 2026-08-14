"""Тесты отбора сообщений для извлечения памяти.

Цена ошибки несимметрична, и тесты это отражают. Лишний вызов модели —
потеря нескольких секунд, ровно как было до отбора. Пропущенный факт о
пользователе теряется навсегда, поэтому маркеры первого лица проверяются
подробнее, чем случаи пропуска.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.memory_gate import mentions_user, worth_extracting


class TestMentionsUser:
    """Признак «человек говорит о себе»."""

    @pytest.mark.parametrize(
        "text",
        [
            "меня зовут Иван",
            "Я инженер по охране труда",
            "я работаю во втором управлении",
            "моя должность — начальник участка",
            "мне положены СИЗ?",
            "наш объект относится к первой категории",
            "запомни: я ответственный за пожарную безопасность",
            "у меня допуск к работам повышенной опасности",
            "не забудь, что я в отпуске с понедельника",
        ],
    )
    def test_personal_messages_pass(self, text: str) -> None:
        """Сообщения о себе распознаются."""
        assert mentions_user(text)

    @pytest.mark.parametrize(
        "text",
        [
            "в какой срок проводится расследование несчастного случая",
            "как часто проверяют огнетушители",
            "кто утверждает состав комиссии по СОУТ",
            "что делать при обнаружении пожара",
            "порядок проведения вводного инструктажа",
            "какие документы оформляются при работах на высоте",
        ],
    )
    def test_regulation_questions_are_skipped(self, text: str) -> None:
        """Вопросы по регламентам сведений о человеке не содержат.

        Это основной путь: именно на них экономится вызов модели, и
        именно они раньше засоряли память журналом вопросов.
        """
        assert not mentions_user(text)

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_is_skipped(self, text: str) -> None:
        """Пустое сообщение в модель не идёт."""
        assert not mentions_user(text)

    def test_yo_is_folded(self) -> None:
        """Написание через «ё» не мешает распознаванию."""
        assert mentions_user("моё подразделение — второе управление")

    def test_substring_does_not_trigger(self) -> None:
        """Маркер ищется как отдельное слово, а не как подстрока.

        Иначе «мыс», «мять» или «изъян» срабатывали бы на «мы» и «я».
        """
        assert not mentions_user("мыс Доброй Надежды и объём работ")


class TestWorthExtracting:
    """Решение по диалогу целиком."""

    def test_personal_fact_in_previous_message_still_counts(self) -> None:
        """Факт из предпоследней реплики не теряется.

        Человек часто представляется отдельным сообщением, а следующим уже
        задаёт вопрос. Если смотреть только на последнюю реплику, имя
        осталось бы незаписанным.
        """
        messages = [
            HumanMessage("меня зовут Иван"),
            AIMessage("Приятно познакомиться."),
            HumanMessage("как часто проверяют огнетушители"),
            AIMessage("Раз в год."),
        ]
        assert worth_extracting(messages)

    def test_only_regulation_questions_are_skipped(self) -> None:
        """Диалог без личных сведений вызова не требует."""
        messages = [
            HumanMessage("что делать при обнаружении пожара"),
            AIMessage("Сообщить в пожарную охрану."),
            HumanMessage("как часто проверяют огнетушители"),
            AIMessage("Раз в год."),
        ]
        assert not worth_extracting(messages)

    def test_accepts_dict_messages(self) -> None:
        """Сообщения приходят и словарями, и объектами.

        Chainlit передаёт словари, а внутри графа они уже объекты; отбор
        обязан работать с обоими видами.
        """
        assert worth_extracting([{"role": "human", "content": "меня зовут Иван"}])
        assert not worth_extracting(
            [{"role": "human", "content": "как часто проверяют огнетушители"}]
        )

    def test_empty_dialog_is_skipped(self) -> None:
        """Пустой диалог не роняет отбор."""
        assert not worth_extracting([])

    def test_assistant_text_is_ignored(self) -> None:
        """Реплики ассистента не учитываются.

        Ассистент постоянно пишет «я» и «мне», и без этого правила отбор
        срабатывал бы всегда, то есть не работал бы вовсе.
        """
        messages = [
            HumanMessage("как часто проверяют огнетушители"),
            AIMessage("Я нашёл ответ: мне известно из регламента, что раз в год."),
        ]
        assert not worth_extracting(messages)
