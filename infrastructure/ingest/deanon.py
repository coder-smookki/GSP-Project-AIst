"""Восстановление плейсхолдеров анонимизации по пофайловому словарю.

Корпус пришёл обезличенным: вместо аббревиатур и названий стоят метки вида
``[ОРГАНИЗАЦИЯ_12]``. Без восстановления поиск ломается — запрос «СОУТ» не
находит документ, в котором аббревиатура заменена меткой.

Ключевой факт, проверенный на корпусе из 20 статей раздела 44030: **один и
тот же идентификатор означает разное в разных документах**. Примеры
``[ОРГАНИЗАЦИЯ_18]``:

* статья 88499: «[ОРГАНИЗАЦИЯ_18] — пожарная безопасность»;
* статья 88535: «[ОРГАНИЗАЦИЯ_18] — сигнал оповещения»;
* статья 88532: «результаты анализа [ОРГАНИЗАЦИЯ_18] руководством».

Поэтому словарь строится **пофайлово**. Глобальный словарь подставил бы
уверенное враньё, а это хуже отказа: пользователь не может отличить
восстановленный термин от выдуманного.

Канал восстановления — непоследовательность самого анонимайзера: он заменил
аббревиатуру, но оставил рядом её расшифровку в разделе «Термины,
определения и сокращения». Ровно эти пары и вычитываются.
"""

from __future__ import annotations

import logging
import re

from infrastructure.ingest.models import ParsedDocument, SectionType
from infrastructure.ingest.patterns import (
    ABBREVIATION_LINE,
    GLOSSARY_LINE,
    GLOSSARY_LINE_REVERSED,
    PLACEHOLDER,
)

logger = logging.getLogger(__name__)

#: Текст, которым заменяется невосстановимый плейсхолдер.
#:
#: Показывать пользователю сырую метку нельзя: она выглядит как ошибка
#: системы. Угадывать значение нельзя тем более — правило кейса прямое:
#: лучше честно сказать «нет данных», чем выдумать.
MASK = "(наименование скрыто)"

#: Минимальная длина расшифровки. Более короткие — мусор разбора.
MIN_VALUE_LEN = 3

#: Максимальная длина расшифровки. Более длинные — это уже предложение,
#: а не термин: подстановка такого в текст ломает читаемость.
MAX_VALUE_LEN = 120


def _normalize_value(value: str) -> str:
    """Приводит расшифровку к виду, пригодному для подстановки.

    Args:
        value: Сырой текст расшифровки из раздела сокращений.

    Returns:
        str: Расшифровка без хвостовой пунктуации и лишних пробелов.
    """
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip(" ;,.:—–-")
    return value


def build_glossary(document: ParsedDocument) -> dict[str, str]:
    """Строит словарь ``плейсхолдер -> расшифровка`` для одного документа.

    Читается только раздел терминов и сокращений: там расшифровки идут в
    предсказуемой форме ``ключ — значение``. Поиск по всему документу дал бы
    ложные пары из обычного текста с тире.

    Args:
        document: Разобранный документ.

    Returns:
        dict[str, str]: Словарь для этого и только этого документа. Пустой,
        если раздел терминов отсутствует или пары не найдены.
    """
    glossary: dict[str, str] = {}
    for section in document.sections:
        if section.type is not SectionType.TERMINY:
            continue
        for atom in section.atoms:
            # Прямой порядок: [ОРГАНИЗАЦИЯ_18] — пожарная безопасность.
            for match in GLOSSARY_LINE.finditer(atom.text):
                key = match.group("key")
                value = _normalize_value(match.group("value"))
                if not (MIN_VALUE_LEN <= len(value) <= MAX_VALUE_LEN):
                    continue
                if PLACEHOLDER.search(value):
                    # Расшифровка сама содержит метку — подставлять нечего.
                    continue
                glossary.setdefault(key, value)
            # Обратный порядок: Общество — [ОРГАНИЗАЦИЯ_1].
            for match in GLOSSARY_LINE_REVERSED.finditer(atom.text):
                key = match.group("key")
                value = _normalize_value(match.group("value"))
                if not (MIN_VALUE_LEN <= len(value) <= MAX_VALUE_LEN):
                    continue
                glossary.setdefault(key, value)

    if glossary:
        logger.info(
            "документ %s (%s): восстановлено расшифровок — %d",
            document.doc_id,
            document.doc_code,
            len(glossary),
        )
    else:
        logger.warning(
            "документ %s (%s): раздел терминов не дал ни одной расшифровки, "
            "плейсхолдеры будут замаскированы",
            document.doc_id,
            document.doc_code,
        )
    return glossary


def build_aliases(document: ParsedDocument) -> dict[str, str]:
    """Собирает уцелевшие аббревиатуры документа для расширения запроса.

    Анонимайзер заменил не все аббревиатуры: часть (например ``ЛВЖ``,
    ``СМОЗиОБТ``) уцелела вместе с расшифровкой. Такие пары не нужны для
    подстановки, но полезны в индексе: по ним запрос «легковоспламеняющаяся
    жидкость» находит чанк, где написано только «ЛВЖ».

    Args:
        document: Разобранный документ.

    Returns:
        dict[str, str]: Словарь ``аббревиатура -> расшифровка``.
    """
    aliases: dict[str, str] = {}
    for section in document.sections:
        if section.type is not SectionType.TERMINY:
            continue
        for atom in section.atoms:
            for match in ABBREVIATION_LINE.finditer(atom.text):
                key = match.group("key").strip()
                value = _normalize_value(match.group("value"))
                if not (MIN_VALUE_LEN <= len(value) <= MAX_VALUE_LEN):
                    continue
                aliases.setdefault(key, value)
    return aliases


def restore_text(
    text: str, glossary: dict[str, str]
) -> tuple[str, list[str], list[str]]:
    """Подставляет расшифровки вместо плейсхолдеров в тексте.

    Args:
        text: Исходный текст с метками анонимизации.
        glossary: Пофайловый словарь ``плейсхолдер -> расшифровка``.

    Returns:
        tuple[str, list[str], list[str]]: Кортеж из трёх элементов:

        * текст с подставленными расшифровками; невосстановимые метки
          заменены на `MASK`;
        * список подставленных расшифровок (для поля ``aliases`` индекса);
        * список меток, которые восстановить не удалось.

    Notes:
        Плейсхолдер не должен попасть в ответ пользователю ни при каком
        исходе — поэтому нераспознанные метки маскируются, а не остаются
        в тексте.
    """
    restored: list[str] = []
    unresolved: list[str] = []

    def replace(match: re.Match[str]) -> str:
        """Заменяет одно вхождение метки.

        Args:
            match: Совпадение `PLACEHOLDER`.

        Returns:
            str: Расшифровка или `MASK`.
        """
        key = match.group(0)
        value = glossary.get(key)
        if value:
            restored.append(value)
            return value
        unresolved.append(key)
        return MASK

    return PLACEHOLDER.sub(replace, text), restored, unresolved


def deanonymize(document: ParsedDocument) -> ParsedDocument:
    """Строит словарь документа и прикрепляет его к документу.

    Сама подстановка выполняется на шаге чанкинга: чанку нужны обе версии
    текста — восстановленная для индекса и исходная для аудита.

    Args:
        document: Разобранный документ.

    Returns:
        ParsedDocument: Копия документа с заполненным полем ``glossary``.
        Словарь объединяет метки анонимизации и уцелевшие аббревиатуры.
    """
    glossary = build_glossary(document)
    aliases = build_aliases(document)
    merged = {**glossary, **{k: v for k, v in aliases.items() if k not in glossary}}
    return document.with_glossary(merged)
