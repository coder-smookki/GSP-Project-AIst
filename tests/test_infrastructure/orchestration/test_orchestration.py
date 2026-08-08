"""Тесты Prefect-обёрток конвейера индексации.

Работают только при установленном ``prefect`` (группа ``orchestration``):
без него Prefect-декораторы в `infrastructure.orchestration` превращаются в
тождественные, и отдельные тесты теряют смысл. В обычном dev-окружении набор
пропускается целиком.
"""

from __future__ import annotations

from pathlib import Path

import pytest

prefect = pytest.importorskip("prefect", reason="prefect не установлен")

from infrastructure.orchestration.ingest_flow import (  # noqa: E402
    _document_ref_id,
    ingest_document_step,
    ingest_flow,
)

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "corpus"


@pytest.fixture(autouse=True)
def disable_ephemeral_server(monkeypatch):
    """Запрещает поднимать временный сервер Prefect ради одного прогона.

    В CI и на машине разработчика поток должен исполняться локально, без
    запуска встроенного сервера и долгих ожиданий его остановки.
    """
    monkeypatch.setenv("PREFECT_SERVER_EPHEMERAL_ENABLED", "0")
    monkeypatch.setenv("PREFECT_API_URL", "")


async def test_ingest_flow_dry_run_counts_chunks() -> None:
    """Сухой прогон через Prefect считает чанки, ничего не записывая."""
    report = await ingest_flow(
        source_kind="files",
        corpus_dir=str(CORPUS),
        limit=2,
        dry_run=True,
    )
    assert len(report.outcomes) == 2
    assert report.total_chunks > 0
    assert report.failed == []


def test_ingest_flow_is_a_prefect_flow() -> None:
    """Обёртка — настоящий поток Prefect, а не обычная функция."""
    assert isinstance(ingest_flow, prefect.flows.Flow)


def test_ingest_document_step_is_a_prefect_task() -> None:
    """Пошаговая обработка документа — задача Prefect."""
    assert isinstance(ingest_document_step, prefect.tasks.Task)


def test_document_ref_id_is_resilient_without_parameters() -> None:
    """Колбэк имени задачи выживает при вызове без параметров.

    Prefect вызывает его через ``resolve_custom_task_run_name``; первый
    вызов может прийти без заполненного словаря параметров.
    """
    assert _document_ref_id() == "документ ?"
    assert _document_ref_id(parameters={}) == "документ ?"


async def test_missing_corpus_raises() -> None:
    """Несуществующий каталог корпуса даёт понятную ошибку."""
    with pytest.raises(ValueError, match="каталог корпуса не найден"):
        await ingest_flow(source_kind="files", corpus_dir="/нет/такого/пути")
