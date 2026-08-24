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

# Prefect: временный сервер в тестах пишет "Stopping temporary server..." уже
# после закрытия capsys/rich консоли pytest — Rich пытается писать в закрытый
# файл и падает ValueError: I/O operation on closed file (см. трейс
# prefect/logging/handlers.py:389 console.print). Это шум teardown, не ошибка
# теста. Оригинальный emit ловит ValueError и зовёт handleError, а handleError
# сам пишет в sys.stderr и тоже падает — поэтому патчим emit напрямую,
# проверяя закрытый файл до печати, как это делает _SafeStreamHandler для
# prefect._internal.
try:
    from prefect.logging.handlers import PrefectConsoleHandler

    def _patched_emit(self, record):  # type: ignore[no-untyped-def]
        try:
            # Если консольный файл уже закрыт pytest, молча пропускаем
            console = getattr(self, "console", None)
            file = getattr(console, "file", None) if console else None
            if file is not None and getattr(file, "closed", False):
                return
            message = self.format(record)
            self.console.print(message, soft_wrap=True)
        except ValueError as exc:
            if "I/O operation on closed file" in str(exc):
                return
            self.handleError(record)
        except Exception:
            self.handleError(record)

    PrefectConsoleHandler.emit = _patched_emit  # type: ignore[method-assign]

    # На случай если handleError всё же вызовется с закрытым sys.stderr
    import logging

    _orig_handler_handleError = logging.Handler.handleError

    def _safe_handler_handleError(self, record):  # type: ignore[no-untyped-def]
        try:
            return _orig_handler_handleError(self, record)
        except ValueError as exc:
            if "I/O operation on closed file" in str(exc):
                return
            raise

    logging.Handler.handleError = _safe_handler_handleError  # type: ignore[method-assign]
except ImportError:
    pass


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
