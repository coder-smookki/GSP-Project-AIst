"""Разбор .docx в те же блоки, что и HTML-статьи.

Свой разбор, а не `python-docx` или `Docling`, по трём причинам:

1. **Структурной разметки в наших файлах нет.** Проверено чтением
   ``word/document.xml`` всех трёх регламентов: ноль вхождений ``w:pStyle``,
   ``outlineLvl`` и ``numPr``. Главный козырь готовых библиотек — построение
   иерархии заголовков по стилям — здесь не даёт ничего, опираться всё равно
   приходится на регулярки по тексту.
2. **Вес.** Docling тянет torch в образ, который мы передаём заказчику.
3. **Офлайн.** Чем меньше зависимостей, тем меньше поводов сходить в сеть.

Формат ``.docx`` — это zip с XML. Нужны ровно две вещи: текст абзацев
(``w:p`` c вложенными ``w:t``) и границы таблиц (``w:tbl``), поэтому хватает
``zipfile`` и ``re`` из стандартной библиотеки.

Ключевая тонкость: текст одного абзаца разложен по нескольким ``w:t`` —
Word режет его по границам форматирования. Склеивать надо **в пределах
абзаца**, иначе слова рвутся на куски прямо посреди предложения.
"""

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

#: Абзац документа.
_PARAGRAPH: Final[re.Pattern[str]] = re.compile(r"<w:p[ >].*?</w:p>|<w:p/>", re.S)

#: Текстовый фрагмент внутри абзаца.
_RUN: Final[re.Pattern[str]] = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)

#: Таблица целиком — нужна, чтобы пометить её абзацы как табличную зону.
_TABLE: Final[re.Pattern[str]] = re.compile(r"<w:tbl[ >].*?</w:tbl>", re.S)

#: Табуляция и разрыв строки внутри абзаца.
_TAB: Final[re.Pattern[str]] = re.compile(r"<w:tab[^>]*/>")
_BREAK: Final[re.Pattern[str]] = re.compile(r"<w:br[^>]*/>")

#: Предел размера распакованного XML. Защита от zip-бомбы: архив в сотню
#: килобайт может разворачиваться в гигабайты.
MAX_XML_BYTES: Final[int] = 64 * 1024 * 1024

#: Длина строки, выше которой она заведомо не является заголовком раздела.
_MAX_HEADING_LEN: Final[int] = 190


def _unescape(text: str) -> str:
    """Раскрывает XML-сущности и схлопывает пробелы.

    Args:
        text: Сырой текст из XML.

    Returns:
        str: Очищенный текст.
    """
    return re.sub(r"\s+", " ", unescape(text)).strip()


def read_paragraphs(path: str | Path) -> list[tuple[str, bool]]:
    """Читает абзацы документа с признаком табличной зоны.

    Args:
        path: Путь к файлу ``.docx``.

    Returns:
        list[tuple[str, bool]]: Пары ``(текст абзаца, находится ли в таблице)``
        в порядке следования. Пустые абзацы отброшены.

    Raises:
        ValueError: Если файл не является ``.docx``, повреждён или его
            распакованный размер превышает `MAX_XML_BYTES`.
    """
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

    # Границы таблиц по позициям в исходном XML: так абзац внутри таблицы
    # опознаётся без разбора структуры строк и ячеек.
    table_spans = [match.span() for match in _TABLE.finditer(xml)]

    def in_table(position: int) -> bool:
        """Проверяет, попадает ли позиция внутрь таблицы.

        Args:
            position: Смещение в XML.

        Returns:
            bool: ``True``, если позиция внутри ``w:tbl``.
        """
        return any(start <= position < end for start, end in table_spans)

    paragraphs: list[tuple[str, bool]] = []
    for match in _PARAGRAPH.finditer(xml):
        # Склейка ИМЕННО в пределах абзаца: Word разрезает текст по
        # границам форматирования, и склейка через границы абзацев
        # порвала бы слова.
        text = _unescape("".join(_RUN.findall(match.group(0))))
        if text:
            paragraphs.append((text, in_table(match.start())))

    logger.debug("%s: абзацев %d", target.name, len(paragraphs))
    return paragraphs


def to_blocks(paragraphs: list[tuple[str, bool]]) -> list[tuple[str, str]]:
    """Превращает абзацы в блоки, понятные общей сборке документа.

    Заголовки разделов опознаются только по тексту: структурных стилей в
    файлах нет. Абзац считается заголовком, если он начинается с номера,
    короткий и не заканчивается точкой — у пунктов регламента точка в конце
    есть почти всегда.

    Args:
        paragraphs: Пары ``(текст, признак таблицы)``.

    Returns:
        list[tuple[str, str]]: Пары ``(тег, текст)`` в формате
        `infrastructure.ingest.html_parse.build_document`.
    """
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


#: Название документа в кавычках-ёлочках на титульном листе.
_QUOTED_TITLE: Final[re.Pattern[str]] = re.compile(r"«\s*(?P<title>[^»]{12,180})\s*»")

#: Шапка титульного листа, которая названием не является.
_BOILERPLATE: Final[re.Pattern[str]] = re.compile(
    r"^(интегрированная\s+система|издание|предисловие|регламент"
    r"|положение|инструкция)\s*$",
    re.I,
)


def _extract_title(paragraphs: list[tuple[str, bool]]) -> str:
    """Достаёт название документа с титульного листа.

    Порядок попыток отражает то, как устроен титул регламента:

    1. название в кавычках-ёлочках — оно уже в правильном регистре;
    2. подряд идущие строки капсом — там же название, но заглавными,
       разбитое переносом на две строки;
    3. первая содержательная строка.

    Args:
        paragraphs: Абзацы документа с признаком таблицы.

    Returns:
        str: Название документа или пустая строка, если найти не удалось.
    """
    head = [text for text, _ in paragraphs[:40]]

    def usable(candidate: str) -> bool:
        """Проверяет, годится ли строка на роль названия.

        Args:
            candidate: Строка-кандидат.

        Returns:
            bool: ``False``, если после удаления плейсхолдеров почти ничего
            не остаётся — такое «название» бесполезно и для показа, и для
            определения области знаний.
        """
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
    """Разбирает файл ``.docx`` в структурированный документ.

    Args:
        path: Путь к файлу.
        source: Идентификатор источника, попадает в метаданные чанков.

    Returns:
        ParsedDocument: Документ с разделами и атомами. Словарь
        деанонимизации не заполняется — это задача шага `deanon`.

    Raises:
        ValueError: Если файл не читается как ``.docx`` или пуст.
    """
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
