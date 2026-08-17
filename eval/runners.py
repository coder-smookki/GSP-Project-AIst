"""Раннеры бенчмаркинга: прогон варианта и сбор результата.

Раннер получает построенный вариант через инъекцию ``build`` (DIP): сам он не
знает, как собрать поиск и агента, и не зависит от слоёв `infrastructure` —
благодаря этому его можно тестировать на фейках и менять сборку без правки
измерительной логики. Метрики передаются списком (OCP): новый раннер не
трогает существующие.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from eval.config import Environment, RunConfig, Variant
from eval.golden import GoldenSet, chunk_contains
from eval.metrics import (
    BaseMetric,
    LatencyMedian,
    LatencyP95,
    compute_metrics,
    default_metrics,
    graph_metrics,
)
from eval.result import Outcome, RunResult

logger = logging.getLogger("eval.runners")

#: Ответ, который система обязана дать при отсутствии сведений.
REFUSAL = "в базе знаний информации нет"

#: Вопросы операционного замера.
#:
#: Три вида намеренно: короткий фактический ответ, длинный процедурный и
#: вопрос без ответа в базе. Они дают разное время генерации, и усреднять
#: только по коротким значит завысить пропускную способность.
DEFAULT_QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "короткий факт",
        "За сколько дней предупреждают о проверке знаний по охране труда?",
    ),
    ("длинная процедура", "Что должен сделать работник при обнаружении пожара?"),
    ("числовой", "Как часто проводится плановая идентификация опасностей?"),
    (
        "перечень",
        "Какие средства индивидуальной защиты обязательны на объекте строительства?",
    ),
    ("отказ", "Какая ставка налога на прибыль организаций?"),
)


@dataclass(frozen=True)
class BuildResult:
    """Построенный вариант: зависимости, которые нужны раннерам.

    Attributes:
        search: Поиск по базе знаний (метод ``query``) или ``None``.
        agent: Скомпилированный граф агента или ``None``.
        environment: Отпечаток окружения, на котором построен вариант.
    """

    search: Any = None
    agent: Any | None = None
    environment: Environment | None = None


#: Сборка варианта: зависит от варианта и конфига прогона.
BuildCallable = Callable[[Variant, RunConfig], BuildResult]


def _now() -> str:
    """Текущее время в ISO-формате.

    Returns:
        str: Метка времени прогона.
    """
    return datetime.now(UTC).isoformat()


async def run_answerable(search, items: list[dict]) -> list[Outcome]:
    """Замеряет поиск на вопросах с известным ответом.

    Args:
        search: Поиск по базе знаний.
        items: Вопросы золотого набора с цитатами.

    Returns:
        list[Outcome]: Итоги по каждому вопросу.
    """
    results: list[Outcome] = []
    for item in items:
        started = time.perf_counter()
        hits = await search.query(item["question"])
        elapsed = time.perf_counter() - started

        outcome = Outcome(
            question=item["question"],
            returned=len(hits),
            seconds=elapsed,
            doc_codes=[chunk.doc_code for chunk, _ in hits],
        )
        for position, (chunk, _score) in enumerate(hits, 1):
            if chunk_contains(item["quote"], chunk.text):
                outcome.rank = position
                break
        results.append(outcome)
        logger.info(
            "%-60s %s",
            item["question"][:60],
            f"позиция {outcome.rank}" if outcome.rank else "НЕ НАЙДЕНО",
        )
    return results


async def run_unanswerable(search, items: list[dict], agent=None) -> list[Outcome]:
    """Замеряет поведение на вопросах, ответа на которые в базе нет.

    Args:
        search: Поиск по базе знаний.
        items: Вопросы без ответа.
        agent: Скомпилированный граф; если задан, дополнительно проверяется
            текст ответа. Без него отказ определяется только по пустой
            выдаче поиска.

    Returns:
        list[Outcome]: Итоги по каждому вопросу.
    """
    results: list[Outcome] = []
    for item in items:
        started = time.perf_counter()
        hits = await search.query(item["question"])
        elapsed = time.perf_counter() - started

        outcome = Outcome(
            question=item["question"],
            answerable=False,
            returned=len(hits),
            seconds=elapsed,
            refused=not hits,
        )

        if agent is not None:
            state = {"messages": [{"role": "human", "content": item["question"]}]}
            # Свой thread_id на каждый вопрос: чекпоинтер его требует, а
            # общий на весь замер тянул бы историю предыдущих вопросов в
            # контекст и портил результат.
            config = {
                "configurable": {
                    "thread_id": f"eval-{uuid.uuid4()}",
                    "user_id": "evaluate",
                }
            }
            final = await agent.ainvoke(state, config=config)
            text = str(final["messages"][-1].content)
            outcome.answer = text
            outcome.refused = REFUSAL in text.lower().replace("ё", "е")

        results.append(outcome)
        logger.info(
            "%-60s %s",
            item["question"][:60],
            "отказ" if outcome.refused else "ОТВЕТИЛА",
        )
    return results


class GoldenSetRunner:
    """Прогоняет вариант на золотом наборе и считает метрики качества.

    Args:
        build: Сборка поиска/агента для варианта.
        metrics: Метрики для расчёта; по умолчанию — стандартный набор.
    """

    name = "golden"

    def __init__(
        self,
        build: BuildCallable,
        metrics: list[BaseMetric] | None = None,
    ) -> None:
        self._build = build
        self._metrics = list(metrics) if metrics is not None else default_metrics()

    async def run(self, variant: Variant, config: RunConfig) -> RunResult:
        """Прогоняет вариант на золотом наборе.

        Args:
            variant: Вариант системы.
            config: Параметры прогона.

        Returns:
            RunResult: Результат с метриками и итогами по вопросам.

        Raises:
            ValueError: В ``BuildResult`` нет поиска.
        """
        golden = GoldenSet.load(config.golden).limited(config.limit)
        if config.bootstraps > 1:
            golden = golden.resample(config.seed, config.sample_index)
        built = self._build(variant, config)
        if built.search is None:
            raise ValueError("GoldenSetRunner требует поиск в BuildResult")
        agent = built.agent if config.with_answers else None

        answerable = await run_answerable(built.search, list(golden.answerable))
        unanswerable = await run_unanswerable(
            built.search, list(golden.unanswerable), agent
        )
        outcomes = [*answerable, *unanswerable]

        return RunResult(
            variant=variant.name,
            environment=built.environment.fingerprint() if built.environment else "",
            metrics=compute_metrics(self._metrics, outcomes),
            outcomes=outcomes,
            created_at=_now(),
            meta={
                "golden": config.golden,
                "limit": config.limit,
                "with_answers": config.with_answers,
                "bootstraps": config.bootstraps,
                "sample_index": config.sample_index,
                "graph": config.graph,
            },
        )


async def one_answer(app, question: str) -> tuple[float, int, bool]:
    """Прогоняет один вопрос через граф целиком.

    Args:
        app: Скомпилированный граф.
        question: Текст вопроса.

    Returns:
        tuple[float, int, bool]: Время в секундах, длина ответа в символах,
        признак отказа «В базе знаний информации нет».
    """
    config = {
        "configurable": {
            "thread_id": f"bench-{uuid.uuid4()}",
            "user_id": "benchmark",
        }
    }
    started = time.perf_counter()
    result = await app.ainvoke(
        {"messages": [{"role": "human", "content": question}]}, config=config
    )
    elapsed = time.perf_counter() - started
    text = str(result["messages"][-1].content)
    refused = REFUSAL in text.lower().replace("ё", "е")
    return elapsed, len(text), refused


async def measure_sequential(
    app, repeats: int, questions: tuple[tuple[str, str], ...] = DEFAULT_QUESTIONS
) -> list[dict]:
    """Меряет время ответа по одному запросу за раз.

    Args:
        app: Скомпилированный граф.
        repeats: Сколько раз прогнать каждый вопрос.
        questions: Вопросы прогона.

    Returns:
        list[dict]: Замеры по каждому вопросу.
    """
    import statistics

    rows: list[dict] = []
    for kind, question in questions:
        times: list[float] = []
        chars = 0
        refused = False
        for _ in range(repeats):
            elapsed, length, was_refused = await one_answer(app, question)
            times.append(elapsed)
            chars = length
            refused = was_refused
        rows.append(
            {
                "вид": kind,
                "вопрос": question,
                "мин": min(times),
                "медиана": statistics.median(times),
                "макс": max(times),
                "символов": chars,
                "отказ": refused,
            }
        )
        logger.info(
            "%-18s медиана %5.1f с  (%d симв.%s)",
            kind,
            statistics.median(times),
            chars,
            ", отказ" if refused else "",
        )
    return rows


async def measure_concurrent(
    app, level: int, questions: tuple[tuple[str, str], ...] = DEFAULT_QUESTIONS
) -> dict:
    """Меряет поведение при одновременных обращениях.

    Ollama по умолчанию обслуживает запросы к одной модели последовательно,
    поэтому одновременные обращения не ускоряют работу, а выстраиваются в
    очередь. Замер показывает, во что это выливается для пользователя:
    растёт ли время ответа линейно числу одновременных запросов.

    Args:
        app: Скомпилированный граф.
        level: Сколько запросов запускать одновременно.
        questions: Вопросы прогона.

    Returns:
        dict: Общее время, худшее время одного ответа, пропускная
        способность в ответах в минуту.
    """
    import asyncio

    pool = [questions[i % len(questions)][1] for i in range(level)]
    started = time.perf_counter()
    results = await asyncio.gather(*(one_answer(app, q) for q in pool))
    total = time.perf_counter() - started
    worst = max(r[0] for r in results)
    return {
        "одновременно": level,
        "всего_с": round(total, 1),
        "худший_ответ_с": round(worst, 1),
        "ответов_в_минуту": round(level / total * 60, 1),
    }


async def run_graph_answerable(agent, items: list[dict]) -> list[Outcome]:
    """Запускает агент на отвечаемых вопросах и собирает графовые метрики.

    Args:
        agent: Скомпилированный граф агента.
        items: Вопросы золотого набора с цитатами.

    Returns:
        list[Outcome]: Итоги с graph_rank, search_calls, refused.
    """
    results: list[Outcome] = []
    for item in items:
        started = time.perf_counter()
        thread_id = f"eval-{uuid.uuid4()}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": "evaluate",
            }
        }
        state = {"messages": [{"role": "human", "content": item["question"]}]}
        final = await agent.ainvoke(state, config=config)
        elapsed = time.perf_counter() - started

        # Извлекаем финальные retrieved и ищем ответ
        retrieved = final.get("retrieved", []) or []
        graph_rank = None
        for pos, chunk in enumerate(retrieved, 1):
            text = chunk.get("text") or chunk.get("snippet") or ""
            if chunk_contains(item["quote"], text):
                graph_rank = pos
                break

        # Ответ модели и отказ
        text = str(final["messages"][-1].content) if final.get("messages") else ""
        refused = REFUSAL in text.lower().replace("ё", "е")

        # Число поисковых вызовов — из trace или эвристика по retrieved
        # В текущем графе trace не сохраняется в state, поэтому приблизим
        # через число уникальных подзапросов в плане (если есть) или длину retrieved
        search_calls = None

        outcome = Outcome(
            question=item["question"],
            answerable=True,
            rank=None,  # не замеряем сырой поиск
            graph_rank=graph_rank,
            search_calls=search_calls,
            returned=len(retrieved),
            seconds=elapsed,
            answer=text,
            refused=refused,
            doc_codes=[c.get("doc_code", "") for c in retrieved if c.get("doc_code")],
        )
        results.append(outcome)
        logger.info(
            "%-60s %s",
            item["question"][:60],
            f"graph позиция {graph_rank}" if graph_rank else "Graph НЕ НАЙДЕНО",
        )
    return results


async def run_graph_unanswerable(agent, items: list[dict]) -> list[Outcome]:
    """Запускает агент на вопросах без ответа и проверяет отказ."""
    results: list[Outcome] = []
    for item in items:
        started = time.perf_counter()
        thread_id = f"eval-{uuid.uuid4()}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": "evaluate",
            }
        }
        state = {"messages": [{"role": "human", "content": item["question"]}]}
        final = await agent.ainvoke(state, config=config)
        elapsed = time.perf_counter() - started

        text = str(final["messages"][-1].content) if final.get("messages") else ""
        refused = REFUSAL in text.lower().replace("ё", "е")
        retrieved = final.get("retrieved", []) or []

        outcome = Outcome(
            question=item["question"],
            answerable=False,
            rank=None,
            graph_rank=None,
            search_calls=None,
            returned=len(retrieved),
            seconds=elapsed,
            answer=text,
            refused=refused,
            doc_codes=[c.get("doc_code", "") for c in retrieved if c.get("doc_code")],
        )
        results.append(outcome)
        logger.info(
            "%-60s %s",
            item["question"][:60],
            "отказ" if refused else "ОТВЕТИЛА",
        )
    return results


class GraphSetRunner:
    """Прогоняет вариант через граф агента и считает графовые метрики качества.

    Args:
        build: Сборка агента для варианта.
        metrics: Метрики для расчёта; по умолчанию — graph_metrics().
    """

    name = "golden-graph"

    def __init__(
        self,
        build: BuildCallable,
        metrics: list[BaseMetric] | None = None,
    ) -> None:
        self._build = build
        self._metrics = list(metrics) if metrics is not None else graph_metrics()

    async def run(self, variant: Variant, config: RunConfig) -> RunResult:
        """Прогоняет вариант через граф агента на золотом наборе."""
        golden = GoldenSet.load(config.golden).limited(config.limit)
        if config.bootstraps > 1:
            golden = golden.resample(config.seed, config.sample_index)
        built = self._build(variant, config)
        if built.agent is None:
            raise ValueError("GraphSetRunner требует агента в BuildResult")

        answerable = await run_graph_answerable(
            built.agent, list(golden.answerable)
        )
        unanswerable = await run_graph_unanswerable(
            built.agent, list(golden.unanswerable)
        )
        outcomes = [*answerable, *unanswerable]

        return RunResult(
            variant=variant.name,
            environment=built.environment.fingerprint() if built.environment else "",
            metrics=compute_metrics(self._metrics, outcomes),
            outcomes=outcomes,
            created_at=_now(),
            meta={
                "golden": config.golden,
                "limit": config.limit,
                "bootstraps": config.bootstraps,
                "sample_index": config.sample_index,
                "graph": config.graph,
                "graph_retrieval": True,
            },
        )


class OperationalRunner:
    """Прогоняет вариант на операционном замере: латенси и нагрузка.

    Args:
        build: Сборка агента для варианта.
        metrics: Метрики латенси для расчёта.
        questions: Вопросы операционного замера.
    """

    name = "operational"

    def __init__(
        self,
        build: BuildCallable,
        metrics: list[BaseMetric] | None = None,
        questions: tuple[tuple[str, str], ...] = DEFAULT_QUESTIONS,
    ) -> None:
        self._build = build
        self._metrics = (
            list(metrics)
            if metrics is not None
            else [
                LatencyMedian(),
                LatencyP95(),
            ]
        )
        self._questions = questions

    async def run(self, variant: Variant, config: RunConfig) -> RunResult:
        """Прогоняет операционный замер.

        Args:
            variant: Вариант системы.
            config: Параметры прогона (repeats, concurrency).

        Returns:
            RunResult: Результат с метриками латенси и пропускной
            способности; детальные замеры лежат в ``meta``.

        Raises:
            ValueError: В ``BuildResult`` нет агента.
        """
        built = self._build(variant, config)
        if built.agent is None:
            raise ValueError("OperationalRunner требует агента в BuildResult")

        sequential = await measure_sequential(
            built.agent, config.repeats, self._questions
        )
        concurrency: list[dict] = []
        for level in config.concurrency:
            concurrency.append(
                await measure_concurrent(built.agent, level, self._questions)
            )

        outcomes = [
            Outcome(
                question=row["вопрос"],
                answerable=False,
                seconds=row["медиана"],
                refused=row["отказ"],
            )
            for row in sequential
        ]
        metrics = compute_metrics(self._metrics, outcomes)
        median = metrics.get("latency_median", 0.0)
        metrics["answers_per_minute"] = 60.0 / median if median else 0.0

        return RunResult(
            variant=variant.name,
            environment=built.environment.fingerprint() if built.environment else "",
            metrics=metrics,
            outcomes=outcomes,
            created_at=_now(),
            meta={
                "sequential": sequential,
                "concurrency": concurrency,
                "repeats": config.repeats,
            },
        )
