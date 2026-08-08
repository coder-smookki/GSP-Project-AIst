"""Модели данных офлайн-конвейера индексации.

Структуры намеренно неизменяемые: конвейер функциональный, каждый шаг
(разбор -> деанонимизация -> чанкинг) возвращает новый объект, поэтому
повторный прогон на тех же входных данных даёт тот же индекс.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import StrEnum


class SectionType(StrEnum):
    """Тип раздела регламента.

    Определяет вес раздела при поиске: `SODERZHANIE` (оглавление) не
    индексируется вовсе, `NORMATIVNYE_SSYLKI` индексируется с понижением,
    остальные — на общих правах.
    """

    NAZNACHENIE = "naznachenie"
    TERMINY = "terminy"
    NORMATIVNYE_SSYLKI = "normativnye_ssylki"
    OBSHCHIE = "obshchie"
    PROCEDURA = "procedura"
    OTVETSTVENNOST = "otvetstvennost"
    PRILOZHENIE = "prilozhenie"
    SODERZHANIE = "soderzhanie"
    PROCHEE = "prochee"


#: Разделы, которые не попадают в индекс: оглавление дублирует заголовки
#: и создаёт ложные попадания на любой запрос по названию раздела.
NOT_INDEXED: frozenset[SectionType] = frozenset({SectionType.SODERZHANIE})


@dataclass(frozen=True, slots=True)
class Atom:
    """Атом текста — один пункт регламента вместе со своим списком.

    Args:
        clause: Номер пункта в исходной нумерации, например ``4.1.4``.
            ``None`` для текста раздела без собственного номера.
        text: Текст пункта, склеенный из всех его абзацев и элементов списка.
        section_no: Номер раздела, которому принадлежит пункт.
        section_title: Заголовок раздела.
        section_type: Тип раздела.
        is_table_region: Признак табличной зоны. Для таких атомов цитата
            деградирует до уровня раздела: номер пункта в таблицах
            недостоверен.
        order: Порядковый номер атома в документе, нужен для склейки соседей.
    """

    clause: str | None
    text: str
    section_no: str
    section_title: str
    section_type: SectionType
    is_table_region: bool = False
    order: int = 0


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """Раздел документа после разбора.

    Args:
        no: Номер раздела как он указан в заголовке, например ``4``.
        title: Заголовок раздела без номера.
        type: Классифицированный тип раздела.
        atoms: Атомы раздела в порядке следования.
    """

    no: str
    title: str
    type: SectionType
    atoms: tuple[Atom, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Документ после разбора источника (HTML статьи или .docx).

    Args:
        doc_id: Идентификатор документа в источнике (id статьи или имя файла).
        doc_code: Короткий код документа для цитат, например ``Р-У2-59``.
            Если код в документе отсутствует, подставляется устойчивый
            суррогат на основе `doc_id`.
        doc_title: Название документа.
        source: Идентификатор источника: ``api`` или ``files``.
        updated_at: Отметка последнего изменения в источнике (ISO-8601).
            Используется для инкрементальной переиндексации.
        sections: Разделы документа в порядке следования.
        glossary: Пофайловый словарь ``плейсхолдер -> расшифровка``.
            Заполняется шагом деанонимизации, при разборе всегда пустой.
        content_hash: SHA-256 исходного текста, позволяет пропустить
            переиндексацию неизменившегося документа.
    """

    doc_id: str
    doc_code: str
    doc_title: str
    source: str
    updated_at: str | None = None
    sections: tuple[ParsedSection, ...] = ()
    glossary: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""

    def atoms(self) -> list[Atom]:
        """Возвращает все атомы документа сквозным списком.

        Returns:
            list[Atom]: Атомы всех разделов в порядке следования, включая
            разделы, которые не подлежат индексации — фильтрация выполняется
            на шаге чанкинга.
        """
        return [atom for section in self.sections for atom in section.atoms]

    def with_glossary(self, glossary: dict[str, str]) -> ParsedDocument:
        """Возвращает копию документа с подставленным словарём.

        Args:
            glossary: Пофайловый словарь ``плейсхолдер -> расшифровка``.

        Returns:
            ParsedDocument: Новый документ; исходный не изменяется.
        """
        return replace(self, glossary=dict(glossary))


@dataclass(frozen=True, slots=True)
class IngestChunk:
    """Единица индексации — то, что кладётся точкой в Qdrant.

    Args:
        chunk_id: Стабильный идентификатор вида ``<doc_code>#c007``.
            Стабильность обязательна: на него ссылается золотой набор eval.
        doc_id: Идентификатор документа в источнике.
        doc_code: Код документа для цитаты.
        doc_title: Название документа.
        section_no: Номер раздела.
        section_title: Заголовок раздела.
        section_type: Тип раздела.
        clause: Номер пункта или ``None`` для табличных зон.
        breadcrumb: Путь ``Документ > Раздел > Пункт``, заменяет overlap
            между чанками: контекст переносится заголовками, а не дублями.
        text: Текст для индексации — после деанонимизации.
        text_raw: Исходный текст с плейсхолдерами, для отладки и аудита.
        aliases: Расшифровки плейсхолдеров, встреченных в чанке; идут в
            индекс отдельным полем для расширения запроса.
        placeholders: Плейсхолдеры, оставшиеся в тексте нерасшифрованными.
        restored: Признак, что в чанке была выполнена хотя бы одна подстановка.
        is_table_region: Признак табличной зоны.
        order_no: Порядковый номер чанка в документе.
    """

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
        """Возвращает детерминированный идентификатор точки Qdrant.

        Qdrant принимает в качестве идентификатора только целое число или
        UUID, поэтому `chunk_id` не годится напрямую. UUID выводится из
        `chunk_id` хешированием: один и тот же чанк всегда получает один и
        тот же идентификатор, поэтому повторный прогон конвейера обновляет
        точку, а не плодит дубликаты.

        Returns:
            str: UUID в каноническом строковом виде.
        """
        digest = hashlib.sha256(self.chunk_id.encode("utf-8")).hexdigest()
        return (
            f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-"
            f"{digest[16:20]}-{digest[20:32]}"
        )
