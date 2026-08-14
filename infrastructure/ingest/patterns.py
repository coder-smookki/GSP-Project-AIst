from __future__ import annotations

import re
from typing import Final

CLAUSE_HEAD: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<truncated>\.)?(?P<clause>\d{1,2}(?:\.\d{1,3}){0,4})\s*[.)]\s+"
)

CLAUSE_GLUED: Final[re.Pattern[str]] = re.compile(
    r"(?<=[:.])\s*(?P<truncated>\.)(?P<clause>\d{1,2}(?:\.\d{1,3}){0,3})\.\s+"
)

SECTION_HEAD: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<no>\d{1,2})\s*[.)]\s*(?P<title>.{2,190}?)\s*$"
)

PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\[[A-ZА-ЯЁ]+(?:_[A-ZА-ЯЁ]+)*_\d+\]")

GLOSSARY_LINE: Final[re.Pattern[str]] = re.compile(
    r"(?P<key>\[[A-ZА-ЯЁ]+(?:_[A-ZА-ЯЁ]+)*_\d+\])\s*[—–-]\s*(?P<value>[^;.]{3,160})"
)

GLOSSARY_LINE_REVERSED: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w\]])(?P<value>[А-ЯЁ][А-Яа-яЁё][А-Яа-яЁё\s-]{1,58}?)\s*[—–]\s*"
    r"(?P<key>\[[A-ZА-ЯЁ]+(?:_[A-ZА-ЯЁ]+)*_\d+\])"
)

ABBREVIATION_LINE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w\]])(?P<key>[А-ЯЁ]{2,10}(?:иОБТ|иМП)?)\s*[—–-]\s*(?P<value>[^;.]{3,160})"
)

DOC_CODE: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<code>[РP]\s?[-–]\s?[АAУY]\d\s?[-–]\s?\d{1,3})\b"
)

API_TITLE_NO: Final[re.Pattern[str]] = re.compile(
    r"^\s*\((?P<no>\d{1,3})\)\s*(?P<title>.+)$"
)

DATE_LIKE: Final[re.Pattern[str]] = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")

XML_JUNK: Final[re.Pattern[str]] = re.compile(r"</?w:[a-zA-Z]+[^>]*>")

TABLE_REF: Final[re.Pattern[str]] = re.compile(
    r"\b(?:табли\w+|приложени\w+)\s*(?:№\s*)?\d{1,2}\b", re.IGNORECASE
)

_SECTION_TYPE_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"^\s*(оглавлени\w*|содержание)\s*$", re.I), "soderzhanie"),
    (re.compile(r"назначени|область примен", re.I), "naznachenie"),
    (re.compile(r"термин|определени|сокращени", re.I), "terminy"),
    (re.compile(r"нормативн|ссылк", re.I), "normativnye_ssylki"),
    (re.compile(r"ответственн|контрол", re.I), "otvetstvennost"),
    (re.compile(r"приложени", re.I), "prilozhenie"),
    (re.compile(r"общие|основные положени", re.I), "obshchie"),
    (
        re.compile(r"порядок|процедур|провед|организац|требовани|обучени", re.I),
        "procedura",
    ),
)


def classify_section(title: str) -> str:
    for pattern, section_type in _SECTION_TYPE_RULES:
        if pattern.search(title):
            return section_type
    return "prochee"


def restore_clause(clause: str, *, truncated: bool, section_no: str) -> str:
    if not section_no:
        return clause

    if truncated:
        return f"{section_no}.{clause}"

    head, _, tail = clause.partition(".")
    if head != section_no and len(section_no) > 1 and section_no.endswith(head):
        return f"{section_no}.{tail}" if tail else section_no
    return clause


def is_false_clause(text: str, match: re.Match[str]) -> bool:
    start, end = match.span("clause")
    return any(d.start() <= start and end <= d.end() for d in DATE_LIKE.finditer(text))
