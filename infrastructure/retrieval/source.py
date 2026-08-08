"""Источники корпуса за единым протоколом `DataSource`.

Смысл протокола — чтобы переход «файлы -> API» не задел ни граф, ни UI, ни
конвейер индексации. Реализации:

* `ApiSource` — база знаний заказчика по HTTP (основной путь);
* `FilesSource` — папка с файлами (резерв и работа без доступа к API).

Источник отдаёт **сырой документ**: разбор HTML или .docx — не его дело, им
занимается `infrastructure.ingest`. Так источник остаётся тонким, а парсеры
можно менять, не трогая доступ к данным.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from infrastructure.retrieval.kb_client import (
    KnowledgeBaseClient,
    KnowledgeBaseConfig,
)

logger = logging.getLogger(__name__)

#: Расширения файлов, которые умеет читать `FilesSource`.
SUPPORTED_SUFFIXES = frozenset({".docx", ".html", ".htm"})


@dataclass(frozen=True, slots=True)
class Section:
    """Область знаний — узел дерева разделов для выбора в интерфейсе.

    Args:
        id: Идентификатор раздела в источнике.
        title: Отображаемое название.
        parent_id: Идентификатор родителя или ``None`` для корня.
    """

    id: str
    title: str
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """Карточка документа без содержимого.

    Нужна, чтобы решить, надо ли вообще качать документ: инкрементальная
    переиндексация сравнивает `updated_at` и не тратит трафик на неизменённое.

    Args:
        doc_id: Идентификатор документа в источнике.
        title: Название документа.
        section_id: Идентификатор раздела.
        updated_at: Отметка последнего изменения (ISO-8601) или ``None``.
    """

    doc_id: str
    title: str
    section_id: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class RawDocument:
    """Документ с содержимым, но ещё не разобранный.

    Args:
        doc_id: Идентификатор документа в источнике.
        title: Название документа.
        kind: Формат содержимого: ``html`` или ``docx``. По нему конвейер
            выбирает парсер.
        content: Содержимое. Для ``html`` — разметка строкой, для ``docx`` —
            путь к файлу строкой (парсер .docx читает файл сам).
        section_id: Идентификатор раздела.
        updated_at: Отметка последнего изменения.
    """

    doc_id: str
    title: str
    kind: str
    content: str
    section_id: str | None = None
    updated_at: str | None = None


@runtime_checkable
class DataSource(Protocol):
    """Доступ к корпусу базы знаний.

    Реализации обязаны быть асинхронными контекстными менеджерами: у сетевых
    источников есть сессия, которую нужно закрыть.
    """

    async def open(self) -> None:
        """Открывает источник (сессию, соединение)."""
        ...

    async def close(self) -> None:
        """Закрывает источник и освобождает ресурсы."""
        ...

    async def list_sections(self) -> list[Section]:
        """Возвращает дерево областей знаний для выбора в интерфейсе."""
        ...

    async def list_documents(self, section_id: str | None = None) -> list[DocumentRef]:
        """Возвращает карточки документов раздела без содержимого."""
        ...

    async def fetch_document(self, doc_id: str) -> RawDocument:
        """Загружает документ вместе с содержимым."""
        ...


class ApiSource:
    """Источник поверх API базы знаний заказчика.

    Тонкая обёртка над `KnowledgeBaseClient`: переводит ответы API в общие
    структуры протокола и ничего не разбирает.
    """

    def __init__(self, config: KnowledgeBaseConfig, *, offline: bool = False) -> None:
        """Создаёт источник.

        Args:
            config: Параметры подключения к API.
            offline: Признак офлайн-режима образа. При ``True`` источник
                отказывается работать: «0 внешних API» — блокирующий критерий
                кейса, и нарушить его молча нельзя.

        Raises:
            RuntimeError: Если источник создаётся в офлайн-режиме.
        """
        if offline:
            raise RuntimeError(
                "ApiSource недоступен в офлайн-режиме (AIST_OFFLINE=1): "
                "передаваемый образ не должен ходить в сеть. "
                "Используйте FilesSource с локальной выгрузкой корпуса"
            )
        self._config = config
        self._client = KnowledgeBaseClient(config)
        self._opened = False

    async def open(self) -> None:
        """Выполняет вход в базу знаний."""
        if self._opened:
            return
        await self._client.login()
        self._opened = True

    async def close(self) -> None:
        """Отзывает токен и закрывает HTTP-сессию."""
        if not self._opened:
            return
        await self._client.__aexit__(None, None, None)
        self._opened = False

    async def __aenter__(self) -> ApiSource:
        """Открывает источник.

        Returns:
            ApiSource: Готовый к работе источник.
        """
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Закрывает источник.

        Args:
            exc_type: Тип исключения, если блок завершился ошибкой.
            exc: Экземпляр исключения.
            tb: Трассировка.
        """
        await self.close()

    async def list_sections(self) -> list[Section]:
        """Возвращает раздел, заданный конфигурацией.

        API организаторов не отдаёт дерево разделов доступными нам методами:
        коллекция содержит только чтение статей раздела. Поэтому возвращается
        один известный раздел — материалы кейса.

        Returns:
            list[Section]: Список из одного раздела.
        """
        return [Section(id=str(self._config.section_id), title="База знаний")]

    async def list_documents(self, section_id: str | None = None) -> list[DocumentRef]:
        """Возвращает карточки статей раздела.

        Args:
            section_id: Идентификатор раздела. По умолчанию — из конфигурации.

        Returns:
            list[DocumentRef]: Карточки статей.
        """
        target = int(section_id) if section_id else self._config.section_id
        articles = await self._client.list_section_articles(target)
        return [
            DocumentRef(
                doc_id=str(item["id"]),
                title=str(item.get("name") or ""),
                section_id=str(item.get("section_id") or target),
                updated_at=str(item.get("updated_at") or "") or None,
            )
            for item in articles
            if item.get("id") is not None
        ]

    async def fetch_document(self, doc_id: str) -> RawDocument:
        """Загружает статью вместе с HTML-содержимым.

        Args:
            doc_id: Идентификатор статьи.

        Returns:
            RawDocument: Документ с разметкой в поле ``content``.
        """
        article = await self._client.get_article(doc_id)
        return RawDocument(
            doc_id=str(article.get("id") or doc_id),
            title=str(article.get("name") or ""),
            kind="html",
            content=str(article.get("content") or ""),
            section_id=str(article.get("section_id") or "") or None,
            updated_at=str(article.get("updated_at") or "") or None,
        )


class FilesSource:
    """Источник поверх локальной папки с документами.

    Работает и в офлайн-режиме: сеть не нужна. Используется как резерв, если
    доступа к API нет, и как способ проиндексировать выгрузку корпуса.
    """

    def __init__(self, root: str | Path) -> None:
        """Создаёт источник.

        Args:
            root: Каталог с файлами корпуса. Обходится рекурсивно.

        Raises:
            ValueError: Если каталог не существует или не является каталогом.
        """
        self._root = Path(root).expanduser().resolve()
        if not self._root.is_dir():
            raise ValueError(f"каталог корпуса не найден: {self._root}")

    async def open(self) -> None:
        """Ничего не делает: файловому источнику нечего открывать."""
        return None

    async def close(self) -> None:
        """Ничего не делает: файловому источнику нечего закрывать."""
        return None

    async def __aenter__(self) -> FilesSource:
        """Открывает источник.

        Returns:
            FilesSource: Готовый к работе источник.
        """
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Закрывает источник.

        Args:
            exc_type: Тип исключения, если блок завершился ошибкой.
            exc: Экземпляр исключения.
            tb: Трассировка.
        """
        return None

    def _files(self) -> list[Path]:
        """Возвращает отсортированный список файлов корпуса.

        Returns:
            list[Path]: Пути к файлам поддерживаемых форматов. Сортировка
            обязательна: без неё порядок обхода зависит от файловой системы,
            а от порядка зависят идентификаторы чанков.
        """
        return sorted(
            path
            for path in self._root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )

    def _ref(self, path: Path) -> DocumentRef:
        """Строит карточку документа по файлу.

        Args:
            path: Путь к файлу.

        Returns:
            DocumentRef: Карточка с идентификатором в виде относительного пути.
        """
        rel = path.relative_to(self._root).as_posix()
        return DocumentRef(
            doc_id=rel,
            title=path.stem,
            section_id=path.parent.relative_to(self._root).as_posix() or None,
            updated_at=str(int(path.stat().st_mtime)),
        )

    async def list_sections(self) -> list[Section]:
        """Возвращает подкаталоги корпуса как области знаний.

        Returns:
            list[Section]: Разделы верхнего уровня; пустой список, если файлы
            лежат плоско.
        """
        sections = [
            Section(id=child.name, title=child.name)
            for child in sorted(self._root.iterdir())
            if child.is_dir()
        ]
        logger.info("файловый источник %s: разделов %d", self._root, len(sections))
        return sections

    async def list_documents(self, section_id: str | None = None) -> list[DocumentRef]:
        """Возвращает карточки файлов корпуса.

        Args:
            section_id: Имя подкаталога для фильтрации или ``None`` для всех.

        Returns:
            list[DocumentRef]: Карточки документов.
        """
        refs = [self._ref(path) for path in self._files()]
        if section_id:
            refs = [ref for ref in refs if ref.section_id == section_id]
        logger.info("файловый источник %s: документов %d", self._root, len(refs))
        return refs

    async def fetch_document(self, doc_id: str) -> RawDocument:
        """Читает документ из файла.

        Args:
            doc_id: Относительный путь файла внутри корпуса.

        Returns:
            RawDocument: Для ``.docx`` в ``content`` кладётся путь к файлу,
            для ``.html`` — сама разметка.

        Raises:
            ValueError: Если путь выходит за пределы каталога корпуса или
                файл не найден. Проверка обязательна: `doc_id` приходит
                снаружи, и без неё возможен выход из каталога.
        """
        candidate = (self._root / doc_id).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"путь вне каталога корпуса: {doc_id!r}")
        if not candidate.is_file():
            raise ValueError(f"файл корпуса не найден: {doc_id!r}")

        ref = self._ref(candidate)
        suffix = candidate.suffix.lower()
        if suffix == ".docx":
            content = str(candidate)
            kind = "docx"
        else:
            content = candidate.read_text(encoding="utf-8", errors="replace")
            kind = "html"
        return RawDocument(
            doc_id=ref.doc_id,
            title=ref.title,
            kind=kind,
            content=content,
            section_id=ref.section_id,
            updated_at=ref.updated_at,
        )
