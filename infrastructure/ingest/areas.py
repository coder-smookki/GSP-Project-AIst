from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final

from infrastructure.ingest.config import DEFAULT_AREA

logger = logging.getLogger(__name__)

UNCLASSIFIED: Final[str] = "prochee"

AREA_TITLES: Final[dict[str, str]] = {
    "pozharnaya_bezopasnost": "Пожарная безопасность",
    "ohrana_truda": "Охрана труда",
    "go_i_chs": "Гражданская оборона и ЧС",
    "promyshlennaya_bezopasnost": "Промышленная безопасность",
    "proizvodstvennaya_bezopasnost": "Производственная безопасность",
    "raboty_povyshennoy_opasnosti": "Работы повышенной опасности",
    UNCLASSIFIED: "Прочее",
}

_AREA_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("pozharnaya_bezopasnost", re.compile(r"пожарн", re.I)),
    (
        "go_i_chs",
        re.compile(r"гражданск\w+\s+оборон|чрезвычайн|эвакуацион|\bго\b", re.I),
    ),
    (
        "ohrana_truda",
        re.compile(
            r"охран\w+\s+труда|услов\w+\s+труда|соут|специальн\w+\s+оценк", re.I
        ),
    ),
    (
        "promyshlennaya_bezopasnost",
        re.compile(r"промышленн\w+(?:\s+\w+){0,3}\s+безопасн", re.I),
    ),
    (
        "proizvodstvennaya_bezopasnost",
        re.compile(
            r"производственн\w+\s+безопасн|идентификац\w+\s+опасност"
            r"|\bриск|происшеств|подрядн\w+\s+организ",
            re.I,
        ),
    ),
    (
        "raboty_povyshennoy_opasnosti",
        re.compile(r"повышенн\w+\s+опасност|наряд-допуск|подъемн\w+\s+сооружен", re.I),
    ),
)


@dataclass(frozen=True, slots=True)
class _Rule:
    code: str
    pattern: re.Pattern[str]


class AreaClassifier:
    def __init__(self, config=DEFAULT_AREA) -> None:
        self.config = config
        with_rules = [
            _Rule(code, pattern) for code, pattern in (config.rules or _AREA_RULES)
        ]
        self._rules = with_rules
        titles: dict[str, str] = dict(AREA_TITLES)
        titles.update(config.titles or {})
        self._titles = titles

    def classify(self, doc_title: str) -> list[str]:
        matched = [rule.code for rule in self._rules if rule.pattern.search(doc_title)]
        if not matched:
            logger.info(
                "документ %r не отнесён ни к одной области, помечен как %s",
                doc_title[:80],
                UNCLASSIFIED,
            )
            return [UNCLASSIFIED]
        return matched

    def title(self, code: str) -> str:
        return self._titles.get(code, code)


def classify_areas(doc_title: str) -> list[str]:
    return AreaClassifier().classify(doc_title)


def area_title(code: str) -> str:
    return AreaClassifier().title(code)
