"""Корневая конфигурация pytest.

Добавляет корень репозитория в ``sys.path``, чтобы тесты находили пакеты
проекта (``agent``, ``gui``, ``infrastructure``, ``domain``) напрямую.

Зачем это нужно. Проект ставится в окружение как editable-пакет, и обычно
этого достаточно. Но `uv run` пересобирает пакет при изменении исходников, и
если пересборка совпала с запуском, путь к пакетам временно теряется —
pytest падает на сборке тестов с ``ModuleNotFoundError: No module named
'agent'``, хотя код в порядке. Ошибка выглядит как поломка проекта и стоит
времени на ложную диагностику.

Явный путь снимает эту зависимость: тесты работают одинаково и с
установленным пакетом, и без него.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    """Отвязывает настройки от машины, на которой идут тесты.

    ``Settings`` наследует ``BaseSettings`` и потому дочитывает значения из
    файла ``.env`` и из переменных окружения — даже когда тест передаёт
    поля явно. Из-за этого набор проходил или падал в зависимости от того,
    чей это компьютер: у разработчика без ``.env`` слот вспомогательной
    модели оставался равен основному, а у коллеги с заполненным ``.env``
    подставлялась другая модель, и проверка на совпадение объектов рушилась.
    Такая же ловушка ждала все 40 мест в тестах, где создаётся ``Settings``.

    Фикстура убирает оба источника: файл ``.env`` не читается, а переменные
    окружения проекта на время теста стираются. Тест, которому нужно
    конкретное значение, задаёт его сам — явно и без сюрпризов.

    Args:
        monkeypatch: Штатная фикстура подмены значений.
    """
    from infrastructure.config.settings import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)

    prefixes = (
        "LLM_",
        "EMBEDDING_",
        "OLLAMA_",
        "LLAMACPP_",
        "VLLM_",
        "RETRIEVAL_",
        "RERANK_",
        "GRADING_",
        "QDRANT_",
        "KB_",
        "AIST_",
        "CORPUS_",
        "VERIFY_",
        "INGEST_",
        "OTLP_",
        "USE_MOCK_",
    )
    for name in list(os.environ):
        if name.startswith(prefixes):
            monkeypatch.delenv(name, raising=False)
