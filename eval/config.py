"""Модель данных бенчмаркинга: варианты, конфиг прогона, окружение.

Это фундамент модуля: кэш, раннеры и A/B-тесты опираются на эти типы, поэтому
они вынесены в отдельный модуль без зависимостей от слоёв `infrastructure` и
`agent`. Окружение включается в ключ кэша, чтобы результаты, снятые на разных
машинах, не смешивались.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Variant:
    """Один вариант системы: имя, описание и оверрайды настроек.

    Base-вариант — это текущее решение с пустыми оверрайдами; новый вариант
    задаётся переопределением любых настроек (модель, реранкер, top_k и т.д.).

    Attributes:
        name: Короткое имя, по которому вариант виден в отчётах и кэше.
        description: Человекочитаемое описание изменения.
        settings: Оверрайды настроек приложения (ключ — поле ``Settings``).
    """

    name: str
    description: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> str:
        """Детерминированный хеш варианта для ключа кэша.

        Returns:
            str: SHA-256 (16 hex-символов) от имени и канонизированных
            оверрайдов.
        """
        payload = json.dumps(
            {"name": self.name, "settings": self.settings},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RunConfig:
    """Параметры прогона одного варианта.

    Attributes:
        runner: Какой раннер исполняет прогон: ``golden`` (золотой набор)
            или ``operational`` (операционный замер). Определяет, что собирать
            при построении варианта — поиск или агента.
        golden: Путь к золотому набору.
        limit: Ограничить число вопросов (первые N) или ``None``.
        with_answers: Дополнительно генерировать ответы агента.
        repeats: Прогонов на вопрос для операционного замера.
        concurrency: Уровни одновременных обращений.
        seed: Зерно ГПСЧ для воспроизводимости.
    """

    runner: Literal["golden", "operational"] = "golden"
    golden: str = "tests/fixtures/golden/questions.json"
    limit: int | None = None
    with_answers: bool = False
    repeats: int = 2
    concurrency: tuple[int, ...] = (1, 2, 4)
    seed: int = 42


@dataclass(frozen=True)
class Environment:
    """Отпечаток машины и настроек, на которых снят результат.

    Результаты бенчмарка осмысленны только вместе с окружением: цифры на
    разных моделях и железе несопоставимы, а кэш должен это учитывать.

    Attributes:
        platform: ОС и архитектура (например, ``Darwin-arm64``).
        python: Версия интерпретатора.
        llm_model: Модель ответов.
        llm_support_model: Служебная модель.
        embedding_model: Модель эмбеддингов.
        retrieval_top_k: Число фрагментов в ответе.
        rerank_enabled: Включён ли реранкер.
        rerank_model: Модель реранкера.
        grading_enabled: Включён ли грейдинг релевантности.
        verify_numbers: Включена ли сверка чисел.
    """

    platform: str
    python: str
    llm_model: str
    llm_support_model: str
    embedding_model: str
    retrieval_top_k: int
    rerank_enabled: bool
    rerank_model: str
    grading_enabled: bool
    verify_numbers: bool

    def describe(self) -> str:
        """Краткое человекочитаемое описание окружения.

        Returns:
            str: Модели и ключевые настройки, по которым результат
            воспроизводим (для отчётов и истории).
        """
        rerank = "вкл" if self.rerank_enabled else "выкл"
        return (
            f"llm={self.llm_model}, emb={self.embedding_model}, "
            f"top_k={self.retrieval_top_k}, rerank={rerank}"
        )

    def fingerprint(self) -> str:
        """Хеш окружения для ключа кэша.

        Returns:
            str: SHA-256 (16 hex-символов) от всех полей.
        """
        values = [
            self.platform,
            self.python,
            self.llm_model,
            self.llm_support_model,
            self.embedding_model,
            str(self.retrieval_top_k),
            str(self.rerank_enabled),
            self.rerank_model,
            str(self.grading_enabled),
            str(self.verify_numbers),
        ]
        return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:16]


def snapshot_environment(settings) -> Environment:
    """Снимает отпечаток окружения из настроек приложения.

    Args:
        settings: Объект настроек (``infrastructure.config.settings.Settings``
            или совместимый по атрибутам).

    Returns:
        Environment: Отпечаток окружения.
    """
    import platform

    return Environment(
        platform=f"{platform.system()}-{platform.machine()}",
        python=sys.version.split()[0],
        llm_model=settings.llm_model,
        llm_support_model=settings.llm_support_model,
        embedding_model=settings.embedding_model,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_enabled=settings.rerank_enabled,
        rerank_model=settings.rerank_model,
        grading_enabled=settings.grading_enabled,
        verify_numbers=settings.verify_numbers,
    )
