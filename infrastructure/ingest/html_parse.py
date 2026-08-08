"""Разбор HTML-статьи базы знаний в структуру `ParsedDocument`.

Используется стандартная библиотека (`html.parser`): у образа не должно быть
ни одной зависимости, способной сходить в сеть, а внешний парсер здесь ничего
не даёт — разметка редактора froala простая и предсказуемая.

Разметка источника (проверено на всех 20 статьях раздела 44030):
    <title>…</title><section class="cover">…<h1>…</h1></section>
    <main><nav><ol>…оглавление…</ol></nav>
      <h2 id="section-1">1. Назначение и область применения</h2>
      <p>1.1. …</p><p>продолжение пункта…</p>
      <h2 id="section-2">2. Термины, определения и сокращения</h2>
      …

Два дефекта конвертера источника обрабатываются явно:
    1. у первого пункта раздела срезана первая цифра: ``.1.`` вместо ``1.1.``;
    2. тело пункта разорвано на несколько ``<p>``, а иногда наоборот —
       два пункта склеены в один абзац.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from html.parser import HTMLParser

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

logger = logging.getLogger(__name__)

#: Теги, содержимое которых не является текстом документа.
_SKIP_CONTENT = frozenset({"script", "style", "nav"})

#: Блочные теги, порождающие отдельный блок текста.
_BLOCK_TAGS = frozenset({"p", "li", "h1", "h2", "h3", "h4", "td", "th", "div"})

#: Предел размера статьи. Защищает конвейер от аномального ответа источника:
#: без него один битый документ съедает память индексатора.
MAX_CONTENT_CHARS = 4_000_000

#: Предел склейки продолжений в один атом.
#:
#: В приложениях и бланках номеров пунктов нет вовсе, поэтому склейка
#: «продолжение предыдущего пункта» без предела собирает всё приложение в
#: один атом на десятки тысяч символов. Такой атом бесполезен: он не влезает
#: в контекст и размывает эмбеддинг. По достижении предела начинается новый
#: атом с тем же номером пункта.
MAX_ATOM_CHARS = 1_500


class _BlockExtractor(HTMLParser):
    """Превращает HTML в плоский список блоков ``(тег, текст)``.

    Содержимое ``<nav>`` отбрасывается: это оглавление, дублирующее заголовки
    разделов. Без отбрасывания оглавление даёт попадание почти на любой
    запрос по названию раздела и вытесняет содержательные чанки.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._buffer: list[str] = []
        self._tag: str = "p"
        self._skip_depth = 0
        self._in_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Обрабатывает открывающий тег.

        Args:
            tag: Имя тега в нижнем регистре.
            attrs: Атрибуты тега (не используются, требуются сигнатурой).
        """
        if tag in _SKIP_CONTENT:
            self._skip_depth += 1
            return
        if tag == "table":
            self._in_table = True
        if tag == "br":
            self._buffer.append(" ")
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        """Обрабатывает закрывающий тег.

        Args:
            tag: Имя тега в нижнем регистре.
        """
        if tag in _SKIP_CONTENT:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "table":
            self._flush()
            self._in_table = False
            return
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        """Накапливает текстовое содержимое текущего блока.

        Args:
            data: Фрагмент текста между тегами.
        """
        if self._skip_depth:
            return
        self._buffer.append(data)

    def _flush(self) -> None:
        """Закрывает текущий блок и кладёт его в результат, если он непустой."""
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
        """Завершает разбор, выталкивая незакрытый блок."""
        super().close()
        self._flush()


def extract_blocks(html: str) -> list[tuple[str, str]]:
    """Извлекает из HTML плоский список блоков.

    Args:
        html: Содержимое поля ``content`` статьи.

    Returns:
        list[tuple[str, str]]: Пары ``(тег, текст)`` в порядке следования.
        Пустые блоки отброшены, пробелы схлопнуты.

    Raises:
        ValueError: Если размер содержимого превышает `MAX_CONTENT_CHARS`.
    """
    if len(html) > MAX_CONTENT_CHARS:
        raise ValueError(
            f"содержимое статьи слишком велико: {len(html)} симв. "
            f"при пределе {MAX_CONTENT_CHARS}"
        )
    parser = _BlockExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        logger.warning(
            "разбор HTML прерван, используем то, что успели собрать: %s", exc
        )
    return parser.blocks


def _split_clauses(text: str, section_no: str) -> list[tuple[str | None, str]]:
    """Разбивает абзац на пункты по якорным номерам.

    Обрабатывает оба дефекта конвертера: срезанную первую цифру в начале
    абзаца и номер пункта, приклеенный к концу предыдущего предложения.

    Args:
        text: Текст одного блока.
        section_no: Номер текущего раздела для восстановления срезанных номеров.

    Returns:
        list[tuple[str | None, str]]: Пары ``(номер пункта, текст)``. Номер
        равен ``None``, если фрагмент является продолжением предыдущего пункта.
    """
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


def _split_oversized(text: str, limit: int = MAX_ATOM_CHARS) -> list[str]:
    """Режет слишком длинный фрагмент на части по границам предложений.

    В приложениях и бланках источник иногда отдаёт целое приложение одним
    абзацем в десятки тысяч символов. Такой атом бесполезен: он не влезает в
    контекст модели, а его эмбеддинг усредняет десяток разных тем.

    Args:
        text: Текст фрагмента.
        limit: Предельная длина части в символах.

    Returns:
        list[str]: Части длиной не более `limit`, кроме случая, когда одно
        предложение само длиннее предела — оно режется по словам.
    """
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
    """Определяет код документа для цитат.

    Args:
        article_id: Идентификатор статьи в источнике.
        title: Название статьи.

    Returns:
        str: Код вида ``Р-У2-59``, если он есть в названии; иначе порядковый
        номер из названия вида ``(20) …`` как ``ДОК-20``; иначе суррогат
        ``ДОК-<id>``. Код обязан быть стабильным — на него ссылаются цитаты
        в ответах и золотой набор eval.
    """
    found = DOC_CODE.search(title)
    if found:
        return re.sub(r"\s", "", found.group("code")).replace("–", "-")
    numbered = API_TITLE_NO.match(title)
    if numbered:
        return f"ДОК-{numbered.group('no')}"
    return f"ДОК-{article_id}"


def _clean_title(title: str) -> str:
    """Убирает из названия служебный порядковый префикс.

    Args:
        title: Название статьи, например ``(20) Регламент …``.

    Returns:
        str: Название без префикса ``(N)``.
    """
    numbered = API_TITLE_NO.match(title)
    return numbered.group("title").strip() if numbered else title.strip()


def parse_article(article: dict, *, source: str = "api") -> ParsedDocument:
    """Разбирает статью базы знаний в структурированный документ.

    Args:
        article: Объект статьи из API — поле ``data`` ответа
            ``GET /api/knowledgebase/article/{id}``. Обязательны ключи
            ``id`` и ``content``; ``name`` и ``updated_at`` необязательны.
        source: Идентификатор источника, попадает в метаданные чанков.

    Returns:
        ParsedDocument: Документ с разделами и атомами. Словарь
        деанонимизации не заполняется — это задача шага `deanon`.

    Raises:
        ValueError: Если в статье нет идентификатора или содержимого.
    """
    article_id = str(article.get("id") or "").strip()
    if not article_id:
        raise ValueError("в статье отсутствует идентификатор")
    content = article.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"статья {article_id}: пустое содержимое")

    raw_title = str(article.get("name") or "").strip()
    blocks = extract_blocks(content)

    doc_title = _clean_title(raw_title)
    doc_code = _doc_code(article_id, raw_title)

    sections: list[ParsedSection] = []
    current_no = ""
    current_title = ""
    current_type = SectionType.PROCHEE
    atoms: list[Atom] = []
    pending_clause: str | None = None
    order = 0

    def close_section() -> None:
        """Закрывает текущий раздел и добавляет его в результат."""
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
            if not SECTION_HEAD.match(text) and not doc_title:
                doc_title = text
            continue

        if tag == "h3":
            # Подзаголовок внутри раздела: раздел не закрывает, но обрывает
            # склейку — следующий абзац не является продолжением пункта.
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

        is_table = tag in {"td", "th"} or bool(TABLE_REF.search(text))
        for clause, body in _split_clauses(text, current_no):
            previous = atoms[-1] if atoms else None
            joinable = (
                clause is None
                and pending_clause is not None
                and previous is not None
                and len(previous.text) + len(body) + 1 <= MAX_ATOM_CHARS
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
            for piece in _split_oversized(body):
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

    document = ParsedDocument(
        doc_id=article_id,
        doc_code=doc_code,
        doc_title=doc_title or doc_code,
        source=source,
        updated_at=str(article.get("updated_at") or "") or None,
        sections=tuple(sections),
    )
    logger.info(
        "разобрана статья %s (%s): разделов %d, атомов %d",
        article_id,
        doc_code,
        len(document.sections),
        len(document.atoms()),
    )
    return document
