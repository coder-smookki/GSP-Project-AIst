"""Маршруты каталога базы знаний, встроенные в Aegra.

Отдают интерфейсу то, чего нет в протоколе агента: дерево областей знаний
(обязательное по ТЗ) и текст документа для раскрытия карточки источника.

Маршруты подключаются к Aegra через ключ ``http`` в ``deploy/aegra.json`` и
попадают под ту же аутентификацию, что и остальной API - отдельного сервиса,
порта и набора ключей не появляется.

Данные берутся из индекса, а не из исходных файлов: корпус в контейнер
приложения не смонтирован, путей к файлам в индексе нет, а у документов из API
их нет в принципе. Подробнее - ``infrastructure/retrieval/catalog.py``.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Path

from infrastructure.config.settings import get_settings
from infrastructure.retrieval.catalog import CatalogUnavailable
from infrastructure.retrieval.factory import get_catalog

logger = logging.getLogger(__name__)

app = FastAPI()

#: Предел длины кода документа в запросе. Код вида «Р-У2-59» короткий, а
#: ограничение отсекает попытки прислать в фильтр мегабайтную строку.
_MAX_CODE_LENGTH = 128


def _catalog():
    """Возвращает каталог или сообщает, что индекс не подключён.

    Returns:
        Catalog: Каталог базы знаний.

    Raises:
        HTTPException: 503, если поиск работает на заглушке и перечислять
            нечего.
    """
    catalog = get_catalog(get_settings())
    if catalog is None:
        raise HTTPException(
            status_code=503, detail="каталог недоступен: индекс не подключён"
        )
    return catalog


@app.get("/knowledge/tree")
async def knowledge_tree() -> dict:
    """Отдаёт дерево «область знаний -> документ -> тип раздела».

    Returns:
        dict: Ключ ``areas`` со списком областей. Один документ может
        встретиться в нескольких областях: разметка областей у чанка - список.

    Raises:
        HTTPException: 503, если индекс недоступен.
    """
    try:
        areas = await _catalog().tree()
    except CatalogUnavailable as exc:
        logger.error("дерево каталога недоступно: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"areas": areas}


@app.get("/knowledge/documents/{doc_code}")
async def knowledge_document(
    doc_code: str = Path(..., min_length=1, max_length=_MAX_CODE_LENGTH),
) -> dict:
    """Собирает документ целиком из фрагментов индекса.

    Args:
        doc_code: Код документа, например ``Р-У2-59``.

    Returns:
        dict: Код, заголовок и фрагменты по возрастанию ``order_no``.

    Raises:
        HTTPException: 404, если документа в индексе нет; 503, если индекс
            недоступен.
    """
    try:
        document = await _catalog().document(doc_code)
    except CatalogUnavailable as exc:
        logger.error("документ %s недоступен: %s", doc_code, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not document["chunks"]:
        raise HTTPException(
            status_code=404, detail=f"документ {doc_code!r} в индексе не найден"
        )
    return document
