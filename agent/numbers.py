from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_NUMBER_WITH_UNIT = re.compile(r"\b(\d+)\s+([а-я]+)")

_CITATION = re.compile(r"\[\d+\]")

_UNIT_PREFIX = 3

#: Основы единиц измерения, которые проверяются.
#:
#: Сверять «число + любое следующее слово» нельзя: замер на демо-машине дал
#: ложные срабатывания «1 док», «1 б, 1 в», «101 с, 31 или» — это куски
#: перечислений и маркеры списка, а не величины. Проверка нужна ради сроков,
#: периодичности и количеств, поэтому слово после числа должно быть единицей.
#:
#: Основы даны по три буквы, как и сравнение: «дне»/«дня» сходятся, «час»
#: покрывает час/часа/часов. Список расширяемый; неизвестная единица просто
#: не проверяется, то есть ошибка идёт в сторону молчания, а не ложной
#: тревоги.
_UNITS = frozenset(
    {
        "дне",
        "дня",
        "дней",
        "ден",  # день
        "час",
        "мин",
        "сек",  # время суток
        "нед",
        "мес",
        "год",
        "лет",
        "ква",  # периоды
        "кал",
        "раб",  # календарных / рабочих
        "шту",
        "экз",
        "ком",  # количество предметов
        "чел",
        "раз",  # люди и кратность
        "лит",
        "мет",
        "кил",
        "тон",
        "про",  # объём, длина, масса, проценты
    }
)


def _pairs(text: str) -> set[tuple[str, str]]:
    """Извлекает пары «число + единица измерения».

    Args:
        text: Текст ответа или контекста.

    Returns:
        set[tuple[str, str]]: Пары «число, основа единицы». Слова, не
        входящие в перечень единиц, пропускаются.
    """
    normalized = text.lower().replace("ё", "е")
    return {
        (match.group(1), stem)
        for match in _NUMBER_WITH_UNIT.finditer(normalized)
        if (stem := match.group(2)[:_UNIT_PREFIX]) in _UNITS
    }


def unsupported_numbers(answer: str, context: str) -> list[str]:
    if not answer or not context:
        return []

    body = _CITATION.sub(" ", answer)
    missing = _pairs(body) - _pairs(context)
    return sorted(f"{number} {unit}" for number, unit in missing)


WARNING_TEMPLATE = (
    "\n\n> ⚠️ Проверьте по первоисточнику: в найденных выдержках не нашлось "
    "подтверждения для {values}. Остальная часть ответа опирается на выдержки."
)


def annotate(answer: str, context: str) -> str:
    missing = unsupported_numbers(answer, context)
    if not missing:
        return answer

    logger.warning(
        "в ответе есть числа, которых нет в контексте: %s", ", ".join(missing)
    )
    values = ", ".join(f"«{value}»" for value in missing)
    return answer + WARNING_TEMPLATE.format(values=values)
