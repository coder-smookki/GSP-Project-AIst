"""Инкрементальная переиндексация: что перезагружать, а что пропустить.

Признак изменения — **хеш содержимого**, а не `updated_at` источника.
Причина в данных: у всех 20 статей выгрузки отметка времени попадает в один
сорокаминутный интервал 03.08.2026 — это след массовой заливки, а не смысловых
правок. Опираться на такую отметку значит либо переиндексировать всё каждый
раз, либо пропустить настоящее изменение.

Хеш считается по разобранному документу, а не по сырому HTML: разметка
источника меняется при каждом сохранении в редакторе, даже когда текст тот же.

Состояние прошлого прогона читается из самого индекса — отдельного хранилища
не заводим. У каждой точки в payload лежит `content_hash` документа, поэтому
достаточно одного прохода `scroll` по коллекции.
"""

from __future__ import annotations

import hashlib
import logging

from qdrant_client import AsyncQdrantClient, models

from infrastructure.ingest.models import ParsedDocument

logger = logging.getLogger(__name__)

#: Размер страницы при чтении состояния индекса.
SCROLL_BATCH = 1024


def document_hash(document: ParsedDocument, pipeline_version: str) -> str:
    """Считает отпечаток документа для сравнения с прошлым прогоном.

    В отпечаток входит версия конвейера: правка парсера, чанкера или правил
    деанонимизации меняет результат при том же исходном тексте, и без учёта
    версии инкремент пропустил бы документы, которые надо перестроить.

    Args:
        document: Разобранный документ.
        pipeline_version: Версия конвейера индексации.

    Returns:
        str: Шестнадцатеричный SHA-256.
    """
    digest = hashlib.sha256()
    digest.update(pipeline_version.encode("utf-8"))
    digest.update(document.doc_code.encode("utf-8"))
    digest.update(document.doc_title.encode("utf-8"))
    for atom in document.atoms():
        digest.update(atom.section_no.encode("utf-8"))
        digest.update((atom.clause or "").encode("utf-8"))
        digest.update(atom.text.encode("utf-8"))
    return digest.hexdigest()


async def load_index_state(
    client: AsyncQdrantClient, collection: str
) -> dict[str, str]:
    """Читает отпечатки документов, уже лежащих в индексе.

    Args:
        client: Клиент Qdrant.
        collection: Имя коллекции.

    Returns:
        dict[str, str]: Отображение ``doc_id -> content_hash``. Пустое, если
        коллекции нет или она пуста — тогда прогон будет полным.
    """
    if not await client.collection_exists(collection):
        logger.info("коллекция %s не существует: индексация будет полной", collection)
        return {}

    state: dict[str, str] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["doc_id", "content_hash"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            content_hash = payload.get("content_hash")
            if isinstance(doc_id, str) and isinstance(content_hash, str):
                state.setdefault(doc_id, content_hash)
        if offset is None:
            break

    logger.info(
        "коллекция %s: состояние прочитано, документов в индексе %d",
        collection,
        len(state),
    )
    return state


async def drop_documents(
    client: AsyncQdrantClient, collection: str, doc_ids: list[str]
) -> None:
    """Удаляет из индекса документы, исчезнувшие из источника.

    Args:
        client: Клиент Qdrant.
        collection: Имя коллекции.
        doc_ids: Идентификаторы документов к удалению.

    Notes:
        Вызывать только при **полностью и без ошибок** полученном списке
        документов. Оборванный обход источника выглядит как «документы
        пропали», и уборка вычистила бы живой индекс.
    """
    if not doc_ids:
        return
    await client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchAny(any=doc_ids)
                    )
                ]
            )
        ),
        wait=True,
    )
    logger.info(
        "удалены документы, исчезнувшие из источника: %s", ", ".join(doc_ids[:10])
    )
