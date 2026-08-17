from __future__ import annotations

import logging
import re
from html import unescape
from html.parser import HTMLParser
from typing import Final

from infrastructure.ingest.config import (
    DEFAULT_PARSE,
    ParsingConfig,
)
from infrastructure.ingest.models import (
    Atom,
    ParsedDocument,
    ParsedSection,
    SectionType,
)
from infrastructure.ingest.patterns import (
    API_TITLE_NO,
    CLAUSE_GLUED,
    CLAUSE_HEAD,
    DOC_CODE,
    SECTION_HEAD,
    TABLE_REF,
    classify_section,
    is_false_clause,
    restore_clause,
)
from infrastructure.ingest.table_md import render_table
from infrastructure.retrieval.source import RawDocument

logger = logging.getLogger(__name__)


class _BlockExtractor(HTMLParser):
    def __init__(self, config: ParsingConfig) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_content = config.skip_content
        self._block_tags = config.block_tags
        self._render_markdown = config.render_tables_as_markdown
        self.blocks: list[tuple[str, str]] = []
        self._buffer: list[str] = []
        self._tag: str = "p"
        self._skip_depth = 0
        self._in_table = False
        self._rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_content:
            self._skip_depth += 1
            return
        if tag == "table":
            self._flush()
            self._in_table = True
            self._rows = []
            self._row = None
            self._cell = None
            return
        if tag == "br":
            if self._in_table and self._cell is not None:
                self._cell.append(" ")
            elif not self._in_table:
                self._buffer.append(" ")
            return
        if self._in_table:
            if not self._render_markdown:
                if tag in self._block_tags:
                    self._flush()
                    self._tag = tag
                return
            if tag == "tr":
                self._close_cell()
                self._row = []
            elif tag in {"td", "th"}:
                self._close_cell()
                self._cell = []
            elif tag in self._block_tags and self._cell is not None:
                self._cell.append(" ")
            return
        if tag in self._block_tags:
            self._flush()
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_content:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "table":
            self._close_row()
            self._in_table = False
            if self._render_markdown:
                markdown = render_table(self._rows)
                if markdown:
                    self.blocks.append(("table", markdown))
            else:
                self._flush()
            self._rows = []
            self._row = None
            self._cell = None
            return
        if self._in_table and self._render_markdown:
            if tag == "tr":
                self._close_row()
            elif tag in {"td", "th"}:
                self._close_cell()
            elif tag in self._block_tags and self._cell is not None:
                self._cell.append(" ")
            return
        if tag in self._block_tags:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_table and self._render_markdown:
            if self._cell is not None:
                self._cell.append(data)
            return
        self._buffer.append(data)

    def _close_cell(self) -> None:
        if self._cell is None:
            return
        text = re.sub(r"\s+", " ", unescape("".join(self._cell))).strip()
        self._cell = None
        if self._row is not None:
            self._row.append(text)

    def _close_row(self) -> None:
        self._close_cell()
        if self._row:
            self._rows.append(self._row)
        self._row = None

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", unescape("".join(self._buffer))).strip()
        self._buffer.clear()
        if not text:
            self._tag = "p"
            return
        tag = self._tag
        if self._in_table and tag not in {"h1", "h2", "h3"}:
            tag = "td"
        self.blocks.append((tag, text))
        self._tag = "p"

    def close(self) -> None:
        super().close()
        if self._in_table and self._render_markdown:
            self._close_row()
            self._in_table = False
            markdown = render_table(self._rows)
            if markdown:
                self.blocks.append(("table", markdown))
            self._rows = []
            self._row = None
            self._cell = None
        self._flush()


_GLUED_HEADING: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.:;])\s+(?P<no>\d{1,2})\.\s+(?P<title>[А-ЯЁ][^.]{8,120})\s*$"
)


class HtmlParser:
    def __init__(self, config: ParsingConfig = DEFAULT_PARSE) -> None:
        self.config = config
        self.kind = "html"

    def extract_blocks(self, html: str) -> list[tuple[str, str]]:
        if len(html) > self.config.max_content_chars:
            raise ValueError(
                f"содержимое статьи слишком велико: {len(html)} симв. "
                f"при пределе {self.config.max_content_chars}"
            )
        parser = _BlockExtractor(self.config)
        try:
            parser.feed(html)
            parser.close()
        except Exception as exc:
            logger.warning(
                "разбор HTML прерван, используем то, что успели собрать: %s", exc
            )
        return parser.blocks

    def parse(self, raw: RawDocument) -> ParsedDocument:
        article = {
            "id": raw.doc_id,
            "name": raw.title,
            "content": raw.content,
            "section_id": raw.section_id,
            "updated_at": raw.updated_at,
        }
        return self.parse_article(article, source="api")

    def parse_article(self, article: dict, *, source: str = "api") -> ParsedDocument:
        article_id = str(article.get("id") or "").strip()
        if not article_id:
            raise ValueError("в статье отсутствует идентификатор")
        content = article.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"статья {article_id}: пустое содержимое")

        raw_title = str(article.get("name") or "").strip()
        document = build_document(
            recover_glued_headings(self.extract_blocks(content)),
            doc_id=article_id,
            doc_code=_doc_code(article_id, raw_title),
            doc_title=_clean_title(raw_title),
            source=source,
            updated_at=str(article.get("updated_at") or "") or None,
        )

        logger.info(
            "разобрана статья %s (%s): разделов %d, атомов %d",
            article_id,
            document.doc_code,
            len(document.sections),
            len(document.atoms()),
        )
        return document


def recover_glued_headings(
    blocks: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    current = 0
    recovered = 0

    for tag, text in blocks:
        if tag == "h2":
            head = SECTION_HEAD.match(text)
            if head:
                current = int(head.group("no"))
            result.append((tag, text))
            continue

        if tag != "p":
            result.append((tag, text))
            continue

        found = _GLUED_HEADING.search(text)
        if not found or not current:
            result.append((tag, text))
            continue

        expected = current + 1
        number = int(found.group("no"))
        if number != expected and not str(expected).endswith(str(number)):
            result.append((tag, text))
            continue

        body = text[: found.start()].strip()
        if body:
            result.append(("p", body))
        result.append(("h2", f"{expected}. {found.group('title').strip()}"))
        current = expected
        recovered += 1

    if recovered:
        logger.info("восстановлено приклеенных заголовков разделов: %d", recovered)
    return result


def _split_clauses(text: str, section_no: str) -> list[tuple[str | None, str]]:
    head = CLAUSE_HEAD.match(text)
    parts: list[tuple[str | None, str]] = []
    cursor = 0
    current_clause: str | None = None

    if head and not is_false_clause(text, head):
        current_clause = restore_clause(
            head.group("clause"),
            truncated=bool(head.group("truncated")),
            section_no=section_no,
        )
        cursor = head.end()

    for glued in CLAUSE_GLUED.finditer(text, cursor):
        if is_false_clause(text, glued):
            continue
        body = text[cursor : glued.start()].strip()
        if body:
            parts.append((current_clause, body))
        current_clause = restore_clause(
            glued.group("clause"),
            truncated=bool(glued.group("truncated")),
            section_no=section_no,
        )
        cursor = glued.end()

    tail = text[cursor:].strip()
    if tail:
        parts.append((current_clause, tail))
    return parts


def _split_oversized(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    buffer = ""
    for sentence in re.split(r"(?<=[.;!?])\s+", text):
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= limit:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        buffer = sentence
    if buffer:
        pieces.append(buffer)
    return [p for p in pieces if p]


def _doc_code(article_id: str, title: str) -> str:
    found = DOC_CODE.search(title)
    if found:
        return re.sub(r"\s", "", found.group("code")).replace("–", "-")
    numbered = API_TITLE_NO.match(title)
    if numbered:
        return f"ДОК-{numbered.group('no')}"
    return f"ДОК-{article_id}"


def _clean_title(title: str) -> str:
    numbered = API_TITLE_NO.match(title)
    return numbered.group("title").strip() if numbered else title.strip()


def build_document(
    blocks: list[tuple[str, str]],
    *,
    doc_id: str,
    doc_code: str,
    doc_title: str,
    source: str,
    updated_at: str | None = None,
    max_atom_chars: int = 1_500,
) -> ParsedDocument:
    sections: list[ParsedSection] = []
    current_no = ""
    current_title = ""
    current_type = SectionType.PROCHEE
    atoms: list[Atom] = []
    pending_clause: str | None = None
    order = 0
    title = doc_title

    def close_section() -> None:
        if not current_title and not atoms:
            return
        sections.append(
            ParsedSection(
                no=current_no,
                title=current_title,
                type=current_type,
                atoms=tuple(atoms),
            )
        )
        atoms.clear()

    for tag, text in blocks:
        if tag == "h1":
            if not SECTION_HEAD.match(text) and not title:
                title = text
            continue

        if tag == "h3":
            pending_clause = None
            order += 1
            atoms.append(
                Atom(
                    clause=None,
                    text=text,
                    section_no=current_no,
                    section_title=current_title,
                    section_type=current_type,
                    is_table_region=bool(TABLE_REF.search(text)),
                    order=order,
                )
            )
            continue

        if tag == "h2":
            head = SECTION_HEAD.match(text)
            close_section()
            pending_clause = None
            if head:
                current_no = head.group("no")
                current_title = head.group("title").strip()
            else:
                current_no = ""
                current_title = text
            current_type = SectionType(classify_section(current_title))
            continue

        is_table = tag in {"td", "th", "table"} or bool(TABLE_REF.search(text))
        for clause, body in _split_clauses(text, current_no):
            previous = atoms[-1] if atoms else None
            joinable = (
                clause is None
                and tag != "table"
                and pending_clause is not None
                and previous is not None
                and len(previous.text) + len(body) + 1 <= max_atom_chars
            )
            if joinable and previous is not None:
                atoms[-1] = Atom(
                    clause=previous.clause,
                    text=f"{previous.text} {body}".strip(),
                    section_no=previous.section_no,
                    section_title=previous.section_title,
                    section_type=previous.section_type,
                    is_table_region=previous.is_table_region or is_table,
                    order=previous.order,
                )
                continue
            for piece in _split_oversized(body, max_atom_chars):
                order += 1
                atoms.append(
                    Atom(
                        clause=clause,
                        text=piece,
                        section_no=current_no,
                        section_title=current_title,
                        section_type=current_type,
                        is_table_region=is_table,
                        order=order,
                    )
                )
            pending_clause = clause if clause is not None else pending_clause

    close_section()

    return ParsedDocument(
        doc_id=doc_id,
        doc_code=doc_code,
        doc_title=title or doc_code,
        source=source,
        updated_at=updated_at,
        sections=tuple(sections),
    )


def extract_blocks(html: str) -> list[tuple[str, str]]:
    return HtmlParser().extract_blocks(html)


def parse_article(article: dict, *, source: str = "api") -> ParsedDocument:
    return HtmlParser().parse_article(article, source=source)


def parse_raw(document: RawDocument) -> ParsedDocument:
    return HtmlParser().parse(document)
