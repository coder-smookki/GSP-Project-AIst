from __future__ import annotations

import logging
import re
import zipfile
from html import unescape
from pathlib import Path
from typing import Final

from infrastructure.ingest.html_parse import build_document
from infrastructure.ingest.models import ParsedDocument
from infrastructure.ingest.patterns import DOC_CODE, PLACEHOLDER, SECTION_HEAD

logger = logging.getLogger(__name__)

_PARAGRAPH: Final[re.Pattern[str]] = re.compile(r"<w:p[ >].*?</w:p>|<w:p/>", re.S)

_RUN: Final[re.Pattern[str]] = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)

_TABLE: Final[re.Pattern[str]] = re.compile(r"<w:tbl[ >].*?</w:tbl>", re.S)

_TAB: Final[re.Pattern[str]] = re.compile(r"<w:tab[^>]*/>")
_BREAK: Final[re.Pattern[str]] = re.compile(r"<w:br[^>]*/>")

MAX_XML_BYTES: Final[int] = 64 * 1024 * 1024

_MAX_HEADING_LEN: Final[int] = 190


def _unescape(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()


def read_paragraphs(path: str | Path) -> list[tuple[str, bool]]:
    target = Path(path)
    try:
        with zipfile.ZipFile(target) as archive:
            try:
                info = archive.getinfo("word/document.xml")
            except KeyError as exc:
                raise ValueError(
                    f"{target.name}: это не .docx — нет word/document.xml"
                ) from exc
            if info.file_size > MAX_XML_BYTES:
                raise ValueError(
                    f"{target.name}: распакованный размер {info.file_size} байт "
                    f"превышает предел {MAX_XML_BYTES}"
                )
            xml = archive.read("word/document.xml").decode("utf-8", "replace")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{target.name}: повреждённый архив — {exc}") from exc

    xml = _BREAK.sub(" ", _TAB.sub(" ", xml))

    table_spans = [match.span() for match in _TABLE.finditer(xml)]

    def in_table(position: int) -> bool:
        return any(start <= position < end for start, end in table_spans)

    paragraphs: list[tuple[str, bool]] = []
    for match in _PARAGRAPH.finditer(xml):
        text = _unescape("".join(_RUN.findall(match.group(0))))
        if text:
            paragraphs.append((text, in_table(match.start())))

    logger.debug("%s: абзацев %d", target.name, len(paragraphs))
    return paragraphs


def to_blocks(paragraphs: list[tuple[str, bool]]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for text, is_table in paragraphs:
        if is_table:
            blocks.append(("td", text))
            continue
        head = SECTION_HEAD.match(text)
        looks_like_heading = (
            head is not None
            and len(text) <= _MAX_HEADING_LEN
            and not text.rstrip().endswith(".")
            and "  " not in text
        )
        blocks.append(("h2", text) if looks_like_heading else ("p", text))
    return blocks


_QUOTED_TITLE: Final[re.Pattern[str]] = re.compile(r"«\s*(?P<title>[^»]{12,180})\s*»")

_BOILERPLATE: Final[re.Pattern[str]] = re.compile(
    r"^(интегрированная\s+система|издание|предисловие|регламент"
    r"|положение|инструкция)\s*$",
    re.I,
)


def _extract_title(paragraphs: list[tuple[str, bool]]) -> str:
    head = [text for text, _ in paragraphs[:40]]

    def usable(candidate: str) -> bool:
        without = PLACEHOLDER.sub("", candidate).strip(" -–—,.")
        return len(without) >= 12

    for text in head:
        found = _QUOTED_TITLE.search(text)
        if found and usable(found.group("title")):
            return found.group("title").strip()

    caps: list[str] = []
    for text in head:
        letters = [c for c in text if c.isalpha()]
        is_caps = bool(letters) and all(c.isupper() for c in letters) and len(text) > 8
        if is_caps and usable(text):
            caps.append(text)
        elif caps:
            break
    if caps:
        return " ".join(caps).capitalize()

    for text in head:
        if (
            len(text) > 12
            and not SECTION_HEAD.match(text)
            and not _BOILERPLATE.match(text)
            and usable(text)
        ):
            return text
    return ""


def parse_docx(path: str | Path, *, source: str = "files") -> ParsedDocument:
    target = Path(path)
    paragraphs = read_paragraphs(target)
    if not paragraphs:
        raise ValueError(f"{target.name}: в документе нет текста")

    joined = " ".join(text for text, _ in paragraphs[:40])
    found = DOC_CODE.search(joined) or DOC_CODE.search(target.stem.replace("_", "-"))
    doc_code = (
        re.sub(r"\s", "", found.group("code")).replace("–", "-")
        if found
        else target.stem[:40]
    )

    title = _extract_title(paragraphs) or target.stem

    document = build_document(
        to_blocks(paragraphs),
        doc_id=target.name,
        doc_code=doc_code,
        doc_title=title[:200],
        source=source,
        updated_at=str(int(target.stat().st_mtime)),
    )
    logger.info(
        "разобран файл %s (%s): разделов %d, атомов %d",
        target.name,
        document.doc_code,
        len(document.sections),
        len(document.atoms()),
    )
    return document
