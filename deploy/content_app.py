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

import importlib.util
import logging
import sys
from pathlib import Path as FilePath

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request

from infrastructure.config.settings import get_settings
from infrastructure.retrieval.catalog import CatalogUnavailable
from infrastructure.retrieval.factory import get_catalog

logger = logging.getLogger(__name__)

def _load_auth():
    """Загружает проверку заголовков из соседнего ``auth.py``.

    Модуль лежит рядом и грузится Aegra по пути, а не импортом пакета,
    поэтому здесь тот же приём.

    Returns:
        Callable[[dict[str, str]], str]: Проверка заголовков.
    """
    target = FilePath(__file__).with_name("auth.py")
    spec = importlib.util.spec_from_file_location("aist_deploy_auth", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module.check_headers


_check_headers = _load_auth()


async def require_user(request: Request) -> str:
    """Пускает только доверенного посредника с именем пользователя.

    Своя проверка, а не флаг ``enable_custom_route_auth``: Aegra сообщает в
    логе, что применила защиту к чужим маршрутам, но каталог при этом отвечал
    без токена - то есть весь состав базы знаний читался кем угодно, кто
    дотянулся до порта.

    Args:
        request: Входящий запрос.

    Returns:
        str: Имя пользователя.

    Raises:
        HTTPException: 401 или 503 по результату проверки.
    """
    headers = {key.lower(): value for key, value in request.headers.items()}
    try:
        return _check_headers(headers)
    except Exception as exc:
        status = getattr(exc, "status_code", 401)
        detail = getattr(exc, "detail", "доступ запрещён")
        raise HTTPException(status_code=status, detail=str(detail)) from exc


app = FastAPI()

#: Свои маршруты и только они закрыты проверкой. Вешать её на всё приложение
#: нельзя: Aegra подключает в него и свои маршруты, и тогда под замок попадает
#: ``/info``, по которому docker решает, жив ли контейнер.
router = APIRouter(dependencies=[Depends(require_user)])

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


@router.get("/knowledge/tree")
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


@router.get("/knowledge/documents/{doc_code}")
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


app.include_router(router)
