from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum

NAMESPACE = uuid.UUID("9f6d9c1e-0d2a-4b7f-9c33-a15700000001")


class SectionType(StrEnum):
    NAZNACHENIE = "naznachenie"
    TERMINY = "terminy"
    NORMATIVNYE_SSYLKI = "normativnye_ssylki"
    OBSHCHIE = "obshchie"
    PROCEDURA = "procedura"
    OTVETSTVENNOST = "otvetstvennost"
    PRILOZHENIE = "prilozhenie"
    SODERZHANIE = "soderzhanie"
    PROCHEE = "prochee"


NOT_INDEXED: frozenset[SectionType] = frozenset({SectionType.SODERZHANIE})


@dataclass(frozen=True, slots=True)
class Atom:
    clause: str | None
    text: str
    section_no: str
    section_title: str
    section_type: SectionType
    is_table_region: bool = False
    order: int = 0


@dataclass(frozen=True, slots=True)
class ParsedSection:
    no: str
    title: str
    type: SectionType
    atoms: tuple[Atom, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    doc_id: str
    doc_code: str
    doc_title: str
    source: str
    updated_at: str | None = None
    sections: tuple[ParsedSection, ...] = ()
    glossary: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""

    def atoms(self) -> list[Atom]:
        return [atom for section in self.sections for atom in section.atoms]

    def with_glossary(self, glossary: dict[str, str]) -> ParsedDocument:
        return replace(self, glossary=dict(glossary))


@dataclass(frozen=True, slots=True)
class IngestChunk:
    chunk_id: str
    doc_id: str
    doc_code: str
    doc_title: str
    section_no: str
    section_title: str
    section_type: SectionType
    clause: str | None
    breadcrumb: str
    text: str
    text_raw: str
    aliases: tuple[str, ...] = ()
    placeholders: tuple[str, ...] = ()
    restored: bool = False
    is_table_region: bool = False
    order_no: int = 0

    def point_id(self) -> str:
        body = hashlib.sha256(self.text_raw.encode("utf-8")).hexdigest()
        key = f"{self.doc_id}|{self.section_no}|{self.clause or ''}|{body}"
        return str(uuid.uuid5(NAMESPACE, key))
