from __future__ import annotations

import logging
import re
import zipfile
from html import unescape
from pathlib import Path
from typing import Final

from infrastructure.ingest.config import DEFAULT_DOCX, DocxConfig
from infrastructure.ingest.html_parse import build_document
from infrastructure.ingest.models import ParsedDocument
from infrastructure.ingest.patterns import DOC_CODE, PLACEHOLDER, SECTION_HEAD
from infrastructure.ingest.table_md import render_table

logger = logging.getLogger(__name__)

_PARAGRAPH: Final[re.Pattern[str]] = re.compile(r"<w:p[ >].*?</w:p>|<w:p/>", re.S)

_RUN: Final[re.Pattern[str]] = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)

_TABLE: Final[re.Pattern[str]] = re.compile(r"<w:tbl[ >].*?</w:tbl>", re.S)

_ROW: Final[re.Pattern[str]] = re.compile(r"<w:tr[ >].*?</w:tr>", re.S)

_CELL: Final[re.Pattern[str]] = re.compile(r"<w:tc[ >].*?</w:tc>", re.S)

_TAB: Final[re.Pattern[str]] = re.compile(r"<w:tab[^>]*/>")
_BREAK: Final[re.Pattern[str]] = re.compile(r"<w:br[^>]*/>")

MAX_XML_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_HEADING_LEN: Final[int] = 190

_QUOTED_TITLE: Final[re.Pattern[str]] = re.compile(r"«\s*(?P<title>[^»]{12,180})\s*»")

_BOILERPLATE: Final[re.Pattern[str]] = re.compile(
    r"^(интегрированная\s+система|издание|предисловие|регламент"
    r"|положение|инструкция)\s*$",
    re.I,
)


def _unescape(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()


def read_paragraphs(path: str | Path) -> list[tuple[str, bool]]:
    return DocxParser().read_paragraphs(path)


def to_blocks(paragraphs: list[tuple[str, bool]]) -> list[tuple[str, str]]:
    return DocxParser().to_blocks(paragraphs)


class DocxParser:
    def __init__(self, config: DocxConfig = DEFAULT_DOCX) -> None:
        self.config = config
        self.kind = "docx"

    def read_paragraphs(self, path: str | Path) -> list[tuple[str, bool]]:
        return [
            (text, tag == "td")
            for tag, text in self._extract_blocks(path)
            if tag != "table"
        ]

    def _read_xml(self, path: str | Path) -> str:
        target = Path(path)
        try:
            with zipfile.ZipFile(target) as archive:
                try:
                    info = archive.getinfo("word/document.xml")
                except KeyError as exc:
                    raise ValueError(
                        f"{target.name}: это не .docx — нет word/document.xml"
                    ) from exc
                if info.file_size > self.config.max_xml_bytes:
                    raise ValueError(
                        f"{target.name}: распакованный размер {info.file_size} байт "
                        f"превышает предел {self.config.max_xml_bytes}"
                    )
                xml = archive.read("word/document.xml").decode("utf-8", "replace")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{target.name}: повреждённый архив — {exc}") from exc
        return _BREAK.sub(" ", _TAB.sub(" ", xml))

    def _paragraph_text(self, paragraph_xml: str) -> str:
        return _unescape("".join(_RUN.findall(paragraph_xml)))

    def _looks_like_heading(self, text: str) -> bool:
        head = SECTION_HEAD.match(text)
        return (
            head is not None
            and len(text) <= self.config.max_heading_len
            and not text.rstrip().endswith(".")
            and "  " not in text
        )

    def _table_rows(self, table_xml: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in _ROW.finditer(table_xml):
            cells = [
                self._paragraph_text(cell.group(0))
                for cell in _CELL.finditer(row.group(0))
            ]
            if any(cells):
                rows.append(cells)
        return rows

    def _extract_blocks(self, path: str | Path) -> list[tuple[str, str]]:
        xml = self._read_xml(path)
        spans = [match.span() for match in _TABLE.finditer(xml)]

        def in_table(position: int) -> bool:
            return any(start <= position < end for start, end in spans)

        if not self.config.render_tables_as_markdown:
            blocks: list[tuple[str, str]] = []
            for match in _PARAGRAPH.finditer(xml):
                text = self._paragraph_text(match.group(0))
                if not text:
                    continue
                if in_table(match.start()):
                    blocks.append(("td", text))
                elif self._looks_like_heading(text):
                    blocks.append(("h2", text))
                else:
                    blocks.append(("p", text))
            return blocks

        events: list[tuple[int, tuple[str, str]]] = []
        for match in _PARAGRAPH.finditer(xml):
            if in_table(match.start()):
                continue
            text = self._paragraph_text(match.group(0))
            if not text:
                continue
            tag = "h2" if self._looks_like_heading(text) else "p"
            events.append((match.start(), (tag, text)))
        for match in _TABLE.finditer(xml):
            markdown = render_table(self._table_rows(match.group(0)))
            if markdown:
                events.append((match.start(), ("table", markdown)))
        events.sort(key=lambda item: item[0])
        return [block for _, block in events]

    def to_blocks(self, paragraphs: list[tuple[str, bool]]) -> list[tuple[str, str]]:
        blocks: list[tuple[str, str]] = []
        for text, is_table in paragraphs:
            if is_table:
                blocks.append(("td", text))
                continue
            blocks.append(
                ("h2", text) if self._looks_like_heading(text) else ("p", text)
            )
        return blocks

    def extract_title(self, texts: list[str]) -> str:
        head = texts[: self.config.title_scan_paragraphs]

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
            is_caps = (
                bool(letters) and all(c.isupper() for c in letters) and len(text) > 8
            )
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

    def parse_file(self, path: str | Path, *, source: str = "files") -> ParsedDocument:
        target = Path(path)
        blocks = self._extract_blocks(target)
        texts = [text for _, text in blocks]
        if not texts:
            raise ValueError(f"{target.name}: в документе нет текста")

        joined = " ".join(texts[:40])
        found = DOC_CODE.search(joined) or DOC_CODE.search(
            target.stem.replace("_", "-")
        )
        doc_code = (
            re.sub(r"\s", "", found.group("code")).replace("–", "-")
            if found
            else target.stem[:40]
        )

        title = self.extract_title(texts) or target.stem

        document = build_document(
            blocks,
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


def parse_docx(path: str | Path, *, source: str = "files") -> ParsedDocument:
    return DocxParser().parse_file(path, source=source)
