from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_NUMBER_WITH_UNIT = re.compile(r"\b(\d+)\s+([а-я]+)")

_CITATION = re.compile(r"\[\d+\]")

_UNIT_PREFIX = 3


def _pairs(text: str) -> set[tuple[str, str]]:
    normalized = text.lower().replace("ё", "е")
    return {
        (match.group(1), match.group(2)[:_UNIT_PREFIX])
        for match in _NUMBER_WITH_UNIT.finditer(normalized)
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
