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
from infrastructure.orchestration.serve import _deployment_kwargs  # noqa: E402

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


async def test_ingest_flow_dry_run_accepts_qa_index_flag() -> None:
    """Флаг qa_index принимается; в dry-run модель QA не собирается.

    Сборка клиента чат-модели потребовала бы живого Ollama, поэтому в
    сухом прогоне она пропускается — как и в основном пути (scripts/ingest.py).
    Тест фиксирует, что флаг проходит до конца прогона офлайн.
    """
    report = await ingest_flow(
        source_kind="files",
        corpus_dir=str(CORPUS),
        limit=1,
        dry_run=True,
        qa_index=True,
    )
    assert len(report.outcomes) == 1
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


def test_deployment_kwargs_defaults_and_env(monkeypatch) -> None:
    """Сборка параметров deployment для serve опирается на AIST_* окружение.

    Без переменных — значения по умолчанию; заданные переменные попадают
    в параметры прогона и расписание. Это то, что контейнер раннера читает
    из `docker-compose.orchestration.yml`.
    """
    for key in (
        "AIST_CRON",
        "AIST_INTERVAL",
        "AIST_SECTION_ID",
        "AIST_LIMIT",
        "AIST_QA_INDEX",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AIST_SOURCE_KIND", "files")
    monkeypatch.setenv("AIST_SWITCH_ALIAS", "1")

    kwargs = _deployment_kwargs()
    params = kwargs["parameters"]
    assert isinstance(kwargs["parameters"], dict)
    assert params["source_kind"] == "files"
    assert params["generation"] == 1
    assert params["dry_run"] is False
    assert params["qa_index"] is False
    assert params["switch_alias"] is True
    assert params["limit"] is None
    assert "cron" not in kwargs
    assert "interval" not in kwargs

    monkeypatch.setenv("AIST_CRON", "0 2 * * *")
    monkeypatch.setenv("AIST_LIMIT", "10")
    monkeypatch.setenv("AIST_SECTION_ID", "7")
    monkeypatch.setenv("AIST_QA_INDEX", "1")
    kwargs = _deployment_kwargs()
    assert kwargs["cron"] == "0 2 * * *"
    assert kwargs["parameters"]["limit"] == 10
    assert kwargs["parameters"]["section_id"] == 7
    assert kwargs["parameters"]["qa_index"] is True
