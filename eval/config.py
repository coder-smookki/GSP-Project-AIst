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
        bootstraps: Сколько независимых прогонов на ресемплированных пулах
            вопросов (бутстрап с возвратом); ``1`` — один прогон без
            ресемплинга.
        sample_index: Номер бутстрап-выборки в пределах прогона
            (``0..bootstraps-1``); задаётся матричным прогоном.
        graph: Тег версии графа агента (``base``, ``rd-v2``, ...). Входит
            в ключ кэша и в A/B-вердикты, поэтому результаты разных графов
            не смешиваются и в отчёте дают отдельные секции.
        graph_retrieval: Запускать графовый замер (агент целиком) вместо
            замера сырого поиска. Включает метрики graph_hit@k, graph_mrr,
            graph_not_found, graph_refusal_rate, avg_search_calls,
            graph_latency_median.
    """

    runner: Literal["golden", "operational"] = "golden"
    golden: str = "tests/fixtures/golden/questions.json"
    limit: int | None = None
    with_answers: bool = False
    repeats: int = 2
    concurrency: tuple[int, ...] = (1, 2, 4)
    seed: int = 42
    bootstraps: int = 1
    sample_index: int = 0
    graph: str = "base"
    graph_retrieval: bool = False


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
        llm_provider: Провайдер LLM (``ollama-local``, ``llamacpp``, ``vllm``).
        llm_reasoning: Включены ли размышления модели.
        embedding_provider: Провайдер эмбеддингов.
        retrieval_prefetch: Кандидатов на канал гибрида до слияния.
        retrieval_rrf_k: Параметр RRF-слияния.
        retrieval_min_relevance: Порог релевантности.
        rerank_candidates: Сколько кандидатов переоценивает реранкер.
        qdrant_collection: Коллекция/алиас Qdrant, по которой идут запросы
            (``aist_chunks``, ``aist_chunks_v2``, ...). Без неё результаты на
            разных наборах эмбеддингов неотличимы ни в кэше, ни в отчёте.
        retrieval_max_requests: Бюджет обращений к поиску на один запрос.
        retrieval_branch_retries: Повторы поиска в ветке-агенте.
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
    llm_provider: str = "ollama-local"
    llm_reasoning: bool = False
    embedding_provider: str = "ollama-local"
    retrieval_prefetch: int = 30
    retrieval_rrf_k: int = 60
    retrieval_min_relevance: float = 0.55
    rerank_candidates: int = 10
    qdrant_collection: str = ""
    retrieval_max_requests: int = 10
    retrieval_branch_retries: int = 1

    def describe(self) -> str:
        """Краткое человекочитаемое описание окружения.

        Returns:
            str: Модели и ключевые настройки, по которым результат
            воспроизводим (для отчётов и истории).
        """
        rerank = "вкл" if self.rerank_enabled else "выкл"
        parts = (
            f"llm={self.llm_model}, emb={self.embedding_model}, "
            f"top_k={self.retrieval_top_k}, prefetch={self.retrieval_prefetch}, "
            f"rrf={self.retrieval_rrf_k}, rerank={rerank}, "
            f"cand={self.rerank_candidates}, min_rel={self.retrieval_min_relevance}, "
            f"max_req={self.retrieval_max_requests}, "
            f"retries={self.retrieval_branch_retries}"
        )
        if self.qdrant_collection:
            parts += f", coll={self.qdrant_collection}"
        return parts

    def fingerprint(self) -> str:
        """Хеш окружения для ключа кэша.

        Returns:
            str: SHA-256 (16 hex-символов) от всех полей.
        """
        values = [
            self.platform,
            self.python,
            self.llm_provider,
            self.llm_model,
            self.llm_support_model,
            str(self.llm_reasoning),
            self.embedding_provider,
            self.embedding_model,
            str(self.retrieval_top_k),
            str(self.retrieval_prefetch),
            str(self.retrieval_rrf_k),
            str(self.retrieval_min_relevance),
            str(self.rerank_enabled),
            self.rerank_model,
            str(self.rerank_candidates),
            str(self.grading_enabled),
            str(self.verify_numbers),
            self.qdrant_collection,
            str(self.retrieval_max_requests),
            str(self.retrieval_branch_retries),
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
        llm_provider=getattr(settings, "llm_provider", "ollama-local"),
        llm_model=settings.llm_model,
        llm_support_model=getattr(settings, "llm_support_model", "None"),
        llm_reasoning=bool(getattr(settings, "llm_reasoning", False)),
        embedding_provider=getattr(settings, "embedding_provider", "ollama-local"),
        embedding_model=settings.embedding_model,
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_prefetch=int(getattr(settings, "retrieval_prefetch", 30)),
        retrieval_rrf_k=int(getattr(settings, "retrieval_rrf_k", 60)),
        retrieval_min_relevance=float(
            getattr(settings, "retrieval_min_relevance", 0.55)
        ),
        rerank_enabled=settings.rerank_enabled,
        rerank_model=settings.rerank_model,
        rerank_candidates=int(getattr(settings, "rerank_candidates", 10)),
        grading_enabled=settings.grading_enabled,
        verify_numbers=settings.verify_numbers,
        qdrant_collection=getattr(settings, "qdrant_collection", ""),
        retrieval_max_requests=int(
            getattr(settings, "retrieval_max_requests", 10)
        ),
        retrieval_branch_retries=int(
            getattr(settings, "retrieval_branch_retries", 1)
        ),
    )
