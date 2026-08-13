"""Генератор docs/BENCHMARK.md: документация модуля и история результатов.

Отчёт собирается из кэша (``eval/results``): каждый прогон — группа записей,
снятых на одном золотом наборе, окружении и конфиге. Строки «Истории
результатов» показывают дату, окружение, варианты, метрики, вердикт A/B и
ссылки на JSON-записи. Статичные разделы («Как пользоваться», «Метрики»,
«A/B-тест») живут здесь же, чтобы документ оставался воспроизводимым из кода.

Запуск::

    uv run python eval/report.py
    uv run eval report --output docs/BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# При прямом запуске в sys.path попадает каталог eval, а не корень репозитория.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.cache import DEFAULT_DIR, ResultCache
from eval.metrics import (
    MRR,
    AnswerChars,
    BaseMetric,
    HitAtK,
    LatencyMedian,
    LatencyP95,
    NotFoundRate,
    RefusalRate,
    default_metrics,
)
from eval.result import RunResult
from eval.stats import compare_all

logger = logging.getLogger("eval.report")

#: Куда писать отчёт по умолчанию.
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "BENCHMARK.md"

#: Соответствие ключей метрик их классам для восстановления из кэша.
_KNOWN_METRICS: dict[str, BaseMetric] = {
    "mrr": MRR(),
    "not_found": NotFoundRate(),
    "refusal_rate": RefusalRate(),
    "latency_median": LatencyMedian(),
    "latency_p95": LatencyP95(),
    "answer_chars": AnswerChars(),
}


def metric_from_key(key: str) -> BaseMetric | None:
    """Восстанавливает метрику по ключу из кэша.

    Args:
        key: Ключ метрики (``hit@3``, ``mrr`` и т.д.).

    Returns:
        BaseMetric: Метрика для A/B-сравнения или ``None``, если ключ не
        распознан.
    """
    if key.startswith("hit@"):
        return HitAtK(int(key[len("hit@") :]))
    return _KNOWN_METRICS.get(key)


def _metric_order() -> dict[str, int]:
    """Порядок метрик по умолчанию для стабильной сортировки колонок."""
    return {metric.key: index for index, metric in enumerate(default_metrics())}


def _format_date(iso: str) -> str:
    """Укорачивает ISO-метку до «дата время».

    Args:
        iso: ISO-метка времени прогона.

    Returns:
        str: ``YYYY-MM-DD HH:MM`` или исходная строка, если она не парсится.
    """
    try:
        return iso.replace("T", " ")[:16]
    except (AttributeError, ValueError):
        return iso


def collect_runs(cache: ResultCache) -> list[dict]:
    """Группирует записи кэша в прогоны.

    Прогон — записи, снятые на одном наборе (dataset), окружении, наборе
    метрик и конфиге: это варианты одной матрицы. Внутри прогона варианты
    упорядочиваются по времени создания, первый считается базовым.

    Args:
        cache: Хранилище результатов.

    Returns:
        list[dict]: Прогоны, отсортированные от свежих к старым.
    """
    groups: dict[tuple, list[dict]] = {}
    for path in sorted(cache.directory.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        config = json.dumps(
            entry.get("config", {}), ensure_ascii=False, sort_keys=True
        )
        identity = (
            entry.get("dataset", ""),
            entry.get("environment", ""),
            entry.get("metrics", ""),
            config,
        )
        groups.setdefault(identity, []).append(entry)

    runs: list[dict] = []
    for entries in groups.values():
        entries.sort(key=lambda entry: entry.get("created_at", ""))
        results = [RunResult.from_dict(entry["result"]) for entry in entries]
        metric_keys: set[str] = set()
        for result in results:
            metric_keys.update(result.metrics)
        first = entries[0]
        runs.append(
            {
                "created_at": first.get("created_at", ""),
                "variants": [result.variant for result in results],
                "environment": first.get("environment_desc", "")
                or first.get("environment", ""),
                "metric_keys": metric_keys,
                "results": results,
                "keys": [entry["key"] for entry in entries],
            }
        )
    runs.sort(key=lambda run: run["created_at"], reverse=True)
    return runs


def _metrics_cell(run: dict) -> str:
    """Компактная строка метрик по вариантам прогона.

    Args:
        run: Прогон из :func:`collect_runs`.

    Returns:
        str: Значения метрик, по строке на метрику (``<br>``-разделитель).
    """
    order = _metric_order()
    keys = sorted(run["metric_keys"], key=lambda key: (order.get(key, 99), key))
    lines: list[str] = []
    for key in keys:
        values = [f"{result.metrics.get(key, 0.0):.3f}" for result in run["results"]]
        if len(values) == 1:
            lines.append(f"{key} {values[0]}")
            continue
        pairs = " / ".join(
            f"{variant}={value}"
            for variant, value in zip(run["variants"], values, strict=True)
        )
        lines.append(f"{key} {pairs}")
    return "<br>".join(lines)


def _verdict_cell(run: dict) -> str:
    """Краткая сводка A/B-вердиктов прогона.

    Args:
        run: Прогон из :func:`collect_runs`.

    Returns:
        str: По строке на вариант: чем он лучше/хуже базы; «—» для прогона из
        одного варианта.
    """
    results = run["results"]
    if len(results) < 2:
        return "—"
    metrics = [
        metric
        for key in run["metric_keys"]
        if (metric := metric_from_key(key)) is not None
    ]
    base = results[0]
    lines: list[str] = []
    for result in results[1:]:
        try:
            verdicts = compare_all(metrics, base, result)
        except ValueError:
            lines.append(f"{result.variant}: нельзя сравнить")
            continue
        better = [v.metric for v in verdicts if v.direction == "better"]
        worse = [v.metric for v in verdicts if v.direction == "worse"]
        if not better and not worse:
            lines.append(f"{result.variant}: без значимых различий")
            continue
        parts = []
        if better:
            parts.append(f"лучше: {', '.join(better)}")
        if worse:
            parts.append(f"хуже: {', '.join(worse)}")
        lines.append(f"{result.variant} ({'; '.join(parts)})")
    return "<br>".join(lines)


def _links_cell(run: dict, cache: ResultCache, output_dir: Path) -> str:
    """Ссылки на JSON-записи кэша прогона, относительно документа.

    Args:
        run: Прогон из :func:`collect_runs`.
        cache: Хранилище результатов.
        output_dir: Каталог, в котором лежит документ.

    Returns:
        str: По ссылке на запись каждого варианта (``<br>``-разделитель).
    """
    links: list[str] = []
    for key, variant in zip(run["keys"], run["variants"], strict=True):
        relative = os.path.relpath(cache.path_for(key), output_dir)
        links.append(f"[{variant}]({relative})")
    return "<br>".join(links)


_STATIC_SECTIONS = """## Как пользоваться

Замер идёт на поднятом стеке (Qdrant, Ollama) с проиндексированным корпусом:
на заглушке поиска команды завершаются с ошибкой, чтобы не выдавать пустые
цифры за результат. Все команды — единый CLI `eval`:

| команда | что делает |
| --- | --- |
| `uv run eval run --with-answers` | качество base на золотом наборе |
| `uv run eval run --variant top6 --settings '{"retrieval_top_k": 6}'` | конкретный вариант |
| `uv run eval ab --variants variants.json --with-answers` | A/B-сравнение вариантов из файла |
| `uv run eval bench --repeats 3` | операционный замер: латенси и нагрузка |
| `uv run eval report` | обновить этот документ из кэша |
| `uv run eval cache list` | посмотреть, какие результаты сохранены |

Полный отчёт прогона (итоги по каждому вопросу, вердикты) пишется в JSON
флагом `--output`. Результаты кэшируются в `eval/results/` по детерминированному
ключу: повторный прогон без изменений берётся из кэша, `--force` переснимает.

## Метрики

| метрика | что меряет | куда лучше | вопросы |
| --- | --- | --- | --- |
| `hit@1`, `hit@3`, `hit@5` | доля вопросов, где ответ в первых k фрагментах | больше | отвечаемые |
| `mrr` | средняя обратная позиция ответа | больше | отвечаемые |
| `not_found` | доля отвечаемых вопросов без ответа в контексте | меньше | отвечаемые |
| `refusal_rate` | доля вопросов без ответа, где система честно отказала | больше | без ответа |
| `latency_median`, `latency_p95` | время поиска | меньше | все |
| `answer_chars` | средняя длина ответа | — | с ответом |

## A/B-тест

Варианты сравниваются на одном золотом наборе, поэтому выборки парные.
Категориальные метрики (`hit@k`, `refusal_rate`, `not_found`) — точный тест
МакНемара по расходящимся парам; непрерывные (`mrr`, латенси) — знаково-
ранговый критерий Уилкоксона. Без scipy обе заменяются парным пермутационным
тестом. Вердикт значим при `p < 0.05`.
"""


def render_markdown(runs: list[dict], cache: ResultCache, output_dir: Path) -> str:
    """Собирает текст документа целиком.

    Args:
        runs: Прогоны из :func:`collect_runs`.
        cache: Хранилище результатов.
        output_dir: Каталог документа (для относительных ссылок).

    Returns:
        str: Текст docs/BENCHMARK.md.
    """
    lines = [
        "# Бенчмарк качества",
        "",
        "Измерение качества поиска и ответов на золотом наборе, A/B-сравнение",
        "вариантов и история результатов. Этот документ генерируется командой",
        "`uv run eval report` из кэша `eval/results/`.",
        "",
        _STATIC_SECTIONS.strip(),
        "",
        "## История результатов",
        "",
        "Каждая строка — один прогон: группа вариантов, снятых на одном наборе,",
        "окружении и конфиге.",
        "",
    ]

    if not runs:
        lines += [
            "Кэш пуст — прогоните `uv run eval run --with-answers` и обновите отчёт.",
            "",
        ]
    else:
        lines += [
            "| дата | окружение | варианты | метрики | вердикт | кэш |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for run in runs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_date(run["created_at"]),
                        run["environment"],
                        ", ".join(run["variants"]),
                        _metrics_cell(run),
                        _verdict_cell(run),
                        _links_cell(run, cache, output_dir),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def generate(cache: ResultCache, output: str | Path) -> int:
    """Пишет docs/BENCHMARK.md из кэша.

    Args:
        cache: Хранилище результатов.
        output: Куда писать отчёт.

    Returns:
        int: Код возврата процесса (всегда 0).
    """
    output_path = Path(output)
    runs = collect_runs(cache)
    text = render_markdown(runs, cache, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    logger.info("отчёт записан в %s (прогонов: %d)", output, len(runs))
    return 0


def main() -> int:
    """Точка входа.

    Returns:
        int: Код возврата процесса.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_DIR),
        help="каталог кэша результатов (по умолчанию eval/results)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="куда писать отчёт (по умолчанию docs/BENCHMARK.md)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return generate(ResultCache(args.cache_dir), args.output)


if __name__ == "__main__":
    sys.exit(main())
