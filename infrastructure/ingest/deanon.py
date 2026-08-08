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

MASK = "(наименование скрыто)"

MIN_VALUE_LEN = 3
MAX_VALUE_LEN = 120


def _normalize_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip(" ;,.:—–-")
    return value


def build_glossary(document: ParsedDocument) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for section in document.sections:
        if section.type is not SectionType.TERMINY:
            continue
        for atom in section.atoms:
            for match in GLOSSARY_LINE.finditer(atom.text):
                key = match.group("key")
                value = _normalize_value(match.group("value"))
                if not (MIN_VALUE_LEN <= len(value) <= MAX_VALUE_LEN):
                    continue
                if PLACEHOLDER.search(value):
                    continue
                glossary.setdefault(key, value)
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
    restored: list[str] = []
    unresolved: list[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(0)
        value = glossary.get(key)
        if value:
            restored.append(value)
            return value
        unresolved.append(key)
        return MASK

    return PLACEHOLDER.sub(replace, text), restored, unresolved


def deanonymize(document: ParsedDocument) -> ParsedDocument:
    glossary = build_glossary(document)
    aliases = build_aliases(document)
    merged = {**glossary, **{k: v for k, v in aliases.items() if k not in glossary}}
    return document.with_glossary(merged)
