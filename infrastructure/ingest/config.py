from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ParsingConfig:
    max_content_chars: int = 4_000_000
    max_atom_chars: int = 1_500
    skip_content: frozenset[str] = frozenset({"script", "style", "nav"})
    block_tags: frozenset[str] = frozenset(
        {"p", "li", "h1", "h2", "h3", "h4", "td", "th", "div"}
    )


@dataclass(frozen=True, slots=True)
class DocxConfig:
    max_xml_bytes: int = 64 * 1024 * 1024
    max_heading_len: int = 190
    title_scan_paragraphs: int = 40


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_chars: int = 900
    max_chars: int = 1_600
    min_chars: int = 40


@dataclass(frozen=True, slots=True)
class DeanonymizationConfig:
    mask: str = "(наименование скрыто)"
    min_value_len: int = 3
    max_value_len: int = 120


@dataclass(frozen=True, slots=True)
class IncrementalConfig:
    scroll_batch: int = 1024
    pipeline_version: str = "2026.08.08-1"


@dataclass(frozen=True, slots=True)
class AreaConfig:
    rules: tuple[tuple[str, re.Pattern[str]], ...] = ()
    titles: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestConfig:
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    docx: DocxConfig = field(default_factory=DocxConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    deanonymization: DeanonymizationConfig = field(
        default_factory=DeanonymizationConfig
    )
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)
    area: AreaConfig = field(default_factory=AreaConfig)


NAMESPACE = uuid.UUID("9f6d9c1e-0d2a-4b7f-9c33-a15700000001")

DEFAULT_PARSE = ParsingConfig()
DEFAULT_DOCX = DocxConfig()
DEFAULT_CHUNK = ChunkingConfig()
DEFAULT_DEANON = DeanonymizationConfig()
DEFAULT_INCREMENTAL = IncrementalConfig()
DEFAULT_AREA = AreaConfig()
DEFAULT_CONFIG = IngestConfig()

MAX_CONTENT_CHARS = DEFAULT_PARSE.max_content_chars
MAX_ATOM_CHARS = DEFAULT_PARSE.max_atom_chars
TARGET_CHARS = DEFAULT_CHUNK.target_chars
MAX_CHARS = DEFAULT_CHUNK.max_chars
MIN_CHARS = DEFAULT_CHUNK.min_chars
MASK = DEFAULT_DEANON.mask
MIN_VALUE_LEN = DEFAULT_DEANON.min_value_len
MAX_VALUE_LEN = DEFAULT_DEANON.max_value_len
SCROLL_BATCH = DEFAULT_INCREMENTAL.scroll_batch
PIPELINE_VERSION = DEFAULT_INCREMENTAL.pipeline_version
