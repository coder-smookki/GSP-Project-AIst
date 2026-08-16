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
from eval.result import RunResult, Verdict
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
        # Пропускаем A/B вердикты (ab_*.json) — у них другая структура
        if path.name.startswith("ab_"):
            continue
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
                "environment_key": first.get("environment", ""),
                "metric_keys": metric_keys,
                "results": results,
                "keys": [entry["key"] for entry in entries],
            }
        )
    runs.sort(key=lambda run: run["created_at"], reverse=True)

    # Загружаем все A/B вердикты из кэша (глобально, не по прогонам)
    all_ab_verdicts = load_all_ab_verdicts(cache)

    return runs, all_ab_verdicts


def load_all_ab_verdicts(cache: ResultCache) -> list[dict]:
    """Загружает все A/B вердикты из кэша.

    Returns:
        list[dict]: Список словарей с полями base_variant, variant, verdicts, environment, dataset, metrics, config, created_at.
    """
    all_verdicts: list[dict] = []
    for path in sorted(cache.directory.glob("ab_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        all_verdicts.append({
            "base_variant": data.get("base_variant", ""),
            "variant": data.get("variant", ""),
            "verdicts": [Verdict.from_dict(v) for v in data.get("verdicts", [])],
            "environment": data.get("environment", ""),
            "environment_desc": data.get("environment_desc", ""),
            "dataset": data.get("dataset", ""),
            "metrics": data.get("metrics", ""),
            "config": data.get("config", {}),
            "created_at": data.get("created_at", ""),
        })
    # Сортируем по дате (новые первые)
    all_verdicts.sort(key=lambda x: x["created_at"], reverse=True)
    return all_verdicts


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
    
    # Используем закешированные A/B вердикты, если есть
    ab_verdicts = run.get("ab_verdicts", {})
    if ab_verdicts:
        lines: list[str] = []
        for result in results[1:]:
            variant_name = result.variant
            verdicts = ab_verdicts.get(variant_name, [])
            if not verdicts:
                lines.append(f"{variant_name}: вердикты не найдены")
                continue
            better = [v.metric for v in verdicts if v.direction == "better"]
            worse = [v.metric for v in verdicts if v.direction == "worse"]
            if not better and not worse:
                lines.append(f"{variant_name}: без значимых различий")
                continue
            parts = []
            if better:
                parts.append(f"лучше: {', '.join(better)}")
            if worse:
                parts.append(f"хуже: {', '.join(worse)}")
            lines.append(f"{variant_name} ({'; '.join(parts)})")
        return "<br>".join(lines)
    
    # Фоллбэк: вычисляем на лету (как было до добавления кэша)
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


#: Человекочитаемые имена атрибутов окружения для сводных таблиц.
_ENV_LABELS = {
    "llm": "LLM",
    "emb": "эмбеддинг",
    "top_k": "top_k",
    "prefetch": "prefetch",
    "rrf": "RRF-k",
    "rerank": "реранк",
    "cand": "кандидаты",
    "min_rel": "мин. релевантность",
}

#: Короткие имена эмбеддеров для компактного лидерборда.
_EMB_SHORT: dict[str, str] = {
    "qwen3-embedding:4b": "qwen3-emb",
    "qwen3-embedding:8b": "qwen3-emb-8b",
    "bge-m3:567m": "bge-m3",
}

#: Короткие имена метрик для сводных таблиц.
_METRIC_SHORT: dict[str, str] = {
    "latency_median": "лат., мед.",
    "latency_p95": "лат., p95",
    "refusal_rate": "отказ",
    "answer_chars": "длина ответа",
    "not_found": "not_found",
}


def _env_params(desc: str) -> dict[str, str]:
    """Разбирает описание окружения на пары ``ключ=значение``.

    Args:
        desc: Строка из :meth:`eval.config.Environment.describe`.

    Returns:
        dict[str, str]: Пары параметров.
    """
    params: dict[str, str] = {}
    for chunk in desc.split(","):
        chunk = chunk.strip()
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            params[key.strip()] = value.strip()
    return params


def _human_config(desc: str) -> str:
    """Компактная человекочитаемая метка конфигурации.

    Args:
        desc: Описание окружения.

    Returns:
        str: Например, ``LLM=gpt-oss:20b, emb=qwen3-emb, top_k=5, реранк=вкл``.
    """
    params = _env_params(desc)
    parts: list[str] = []
    if "llm" in params:
        parts.append(f"LLM={params['llm']}")
    if "emb" in params:
        parts.append(f"emb={_EMB_SHORT.get(params['emb'], params['emb'])}")
    if "top_k" in params:
        parts.append(f"top_k={params['top_k']}")
    if "rerank" in params:
        parts.append(f"реранк={params['rerank']}")
    if "cand" in params:
        parts.append(f"канд.={params['cand']}")
    return ", ".join(parts)


def _env_diff(base: str, variant: str) -> str:
    """Чем окружение variant отличается от base (человекочитаемо).

    Args:
        base: Описание окружения базового варианта.
        variant: Описание окружения нового варианта.

    Returns:
        str: Список изменений через ``; `` или пустая строка, если
        окружения идентичны.
    """
    base_params = _env_params(base)
    variant_params = _env_params(variant)
    changed: list[str] = []
    for key in base_params.keys() | variant_params.keys():
        old, new = base_params.get(key, "—"), variant_params.get(key, "—")
        if old != new:
            label = _ENV_LABELS.get(key, key)
            changed.append(f"{label}: {old} → {new}")
    return "; ".join(changed)


def _variant_envs(runs: list[dict]) -> dict[str, list[str]]:
    """Все окружения, где встречался каждый вариант.

    Args:
        runs: Прогоны из :func:`collect_runs`.

    Returns:
        dict[str, list[str]]: Имя варианта → список описаний окружения.
    """
    envs: dict[str, list[str]] = {}
    for run in runs:
        for variant in run["variants"]:
            if run["environment"] not in envs.setdefault(variant, []):
                envs[variant].append(run["environment"])
    return envs


def _best_base_env(base_name: str, variant_env: str, variant_envs: dict[str, list[str]]) -> str:
    """Выбирает окружение базы, ближайшее к окружению варианта.

    Имя базы может встречаться в нескольких прогонах (разные конфиги);
    для описания «что изменилось» берём то окружение, которое меньше всего
    отличается от окружения сравниваемого варианта.

    Args:
        base_name: Имя базового варианта.
        variant_env: Описание окружения варианта.
        variant_envs: Карта окружений вариантов.

    Returns:
        str: Описание окружения базы или пустая строка.
    """
    variant_params = _env_params(variant_env)
    best, best_score = "", None
    for env in variant_envs.get(base_name, []):
        base_params = _env_params(env)
        score = sum(
            1
            for key in base_params.keys() | variant_params.keys()
            if base_params.get(key) != variant_params.get(key)
        )
        if best_score is None or score < best_score:
            best, best_score = env, score
    return best


def _metric_short(key: str) -> str:
    """Короткое имя метрики для сводных таблиц."""
    return _METRIC_SHORT.get(key, key)


def _aggregate_configs(runs: list[dict]) -> list[dict]:
    """Сводные конфигурации: окружение → средние метрики по прогонам.

    Одна и та же конфигурация (модель, эмбеддер, top_k, реранк) может быть
    прогнана несколько раз или с разными вариантами промпта; лидерборд
    показывает её среднее, чтобы не дублировать строки.

    Args:
        runs: Прогоны из :func:`collect_runs`.

    Returns:
        list[dict]: Конфигурации, отсортированные по ``mrr`` (и ``hit@1``).
    """
    grouped: dict[str, list[dict]] = {}
    labels: dict[str, str] = {}
    order: list[str] = []
    for run in runs:
        key = run.get("environment_key") or run["environment"]
        if key not in grouped:
            grouped[key] = []
            labels[key] = run["environment"]
            order.append(key)
        for result in run["results"]:
            grouped[key].append(result.metrics)

    rows: list[dict] = []
    for env in order:
        metrics = grouped[env]

        def avg(key: str, metrics: list[dict] = metrics) -> float | None:
            values = [m.get(key) for m in metrics if m.get(key) is not None]
            if not values:
                return None
            return sum(values) / len(values)

        rows.append(
            {
                "environment": labels[env],
                "n": len(metrics),
                "hit@1": avg("hit@1"),
                "mrr": avg("mrr"),
                "not_found": avg("not_found"),
                "latency_median": avg("latency_median"),
            }
        )

    rows.sort(
        key=lambda row: (
            row["mrr"] is not None,
            row["mrr"] if row["mrr"] is not None else -1.0,
            row["hit@1"] if row["hit@1"] is not None else -1.0,
        ),
        reverse=True,
    )
    return rows


def _verdict_line(verdict: Verdict) -> str:
    """Одна строка значимого изменения метрики.

    Args:
        verdict: Вердикт A/B.

    Returns:
        str: ``метрика: base→variant (p=..., направление)``.
    """
    arrow = {"better": "улучшение", "worse": "ухудшение"}.get(verdict.direction, "без разницы")
    return (
        f"{_metric_short(verdict.metric)}: {verdict.base_value:.3f}→{verdict.variant_value:.3f} "
        f"(p={verdict.p_value:.3f}, {arrow})"
    )


def _ab_groups(all_ab_verdicts: list[dict], runs: list[dict]) -> list[dict]:
    """Свёрнутые A/B-сравнения: одна строка на пару вариантов.

    Если пара (base, variant, окружение) прогнана несколько раз, берётся
    последний прогон. Значимыми считаются вердикты с ``p < 0.05`` и
    ненулевой разницей.

    Args:
        all_ab_verdicts: Все A/B вердикты из :func:`load_all_ab_verdicts`.
        runs: Прогоны из :func:`collect_runs` (для описания изменений).

    Returns:
        list[dict]: Группы, отсортированные от свежих к старым.
    """
    latest: dict[tuple, dict] = {}
    for av in all_ab_verdicts:
        identity = (av["base_variant"], av["variant"], av["environment"], av.get("environment_desc", "") or av["environment"])
        if identity not in latest or av["created_at"] > latest[identity]["created_at"]:
            latest[identity] = av

    variant_envs = _variant_envs(runs)
    groups: list[dict] = []
    for av in latest.values():
        significant = [
            verdict
            for verdict in av["verdicts"]
            if verdict.p_value < 0.05 and verdict.direction != "no_diff"
        ]
        variant_env = av.get("environment_desc", "") or av["environment"]
        base_env = _best_base_env(av["base_variant"], variant_env, variant_envs)
        essence = _env_diff(base_env, variant_env) if base_env else "—"
        groups.append(
            {
                "date": av["created_at"],
                "base": av["base_variant"],
                "variant": av["variant"],
                "environment": variant_env,
                "essence": essence,
                "verdicts": av["verdicts"],
                "significant": significant,
            }
        )
    groups.sort(key=lambda group: group["date"], reverse=True)
    return groups


def _group_verdict(significant: list[Verdict]) -> str:
    """Итоговый вердикт группы по значимым изменениям.

    Args:
        significant: Значимые вердикты группы.

    Returns:
        str: ``лучше``, ``хуже``, ``неоднозначно`` или ``паритет``.
    """
    better = any(v.direction == "better" for v in significant)
    worse = any(v.direction == "worse" for v in significant)
    if better and worse:
        return "неоднозначно"
    if worse:
        return "хуже"
    if better:
        return "лучше"
    return "паритет"


def _recommendations_section(runs: list[dict], ab_groups: list[dict]) -> str:
    """Сводка «Рекомендации»: лидер и ключевые выводы из A/B.

    Args:
        runs: Прогоны из :func:`collect_runs`.
        ab_groups: Группы из :func:`_ab_groups`.

    Returns:
        str: Markdown-раздел или пустая строка, если данных нет.
    """
    if not runs:
        return ""
    lines = ["## Рекомендации", ""]
    leaderboard = _aggregate_configs(runs)
    if leaderboard:
        leader = leaderboard[0]
        hit = f"{leader['hit@1']:.3f}" if leader["hit@1"] is not None else "—"
        mrr = f"{leader['mrr']:.3f}" if leader["mrr"] is not None else "—"
        lines.append(
            f"**Текущий лидер:** `{_human_config(leader['environment'])}` — "
            f"hit@1 = {hit}, MRR = {mrr}."
        )
        lines.append("")

    impact = [group for group in ab_groups if group["significant"]]
    if impact:
        lines.append("**Значимые эффекты (p < 0.05):**")
        for group in impact:
            essence = group["essence"] or "конфигурация не менялась"
            deltas = "; ".join(_verdict_line(v) for v in group["significant"])
            verdict = _group_verdict(group["significant"])
            lines.append(
                f"- `{group['base']}` → `{group['variant']}` (_{essence}_): "
                f"**{verdict}** — {deltas}"
            )
        lines.append("")
    else:
        lines.append("**Значимых A/B-эффектов (p < 0.05) пока нет.**")
        lines.append("")

    if ab_groups:
        parity = len(ab_groups) - len(impact)
        lines.append(
            f"Проверено A/B-сравнений: {len(ab_groups)}, из них без значимых "
            f"различий: {parity}."
        )
        lines.append("")
    return "\n".join(lines)


def _leaderboard_section(runs: list[dict]) -> str:
    """Лидерборд: агрегированные конфигурации, отсортированные по MRR.

    Args:
        runs: Прогоны из :func:`collect_runs`.

    Returns:
        str: Markdown-раздел или пустая строка, если кэш пуст.
    """
    rows = _aggregate_configs(runs)
    if not rows:
        return ""
    leader = rows[0]
    fastest = min(
        (row for row in rows if row["latency_median"] is not None),
        key=lambda row: row["latency_median"],
        default=None,
    )
    lines = [
        "## Лидерборд",
        "",
        "Каждая строка — уникальная конфигурация (модель, эмбеддер, top_k, "
        "реранк), агрегированная по всем прогонам её вариантов. Сортировка — "
        "по MRR.",
        "",
        "| конфигурация | hit@1 | MRR | not_found | лат., мед. | кол-во | заметка |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        notes: list[str] = []
        if row is leader:
            notes.append("максимальное качество")
        if fastest is not None and row is fastest:
            notes.append("самый быстрый")
        params = _env_params(row["environment"])
        if params.get("rerank") == "выкл":
            notes.append("реранк выкл")
        lines.append(
            "| "
            + " | ".join(
                [
                    _human_config(row["environment"]),
                    _fmt_number(row["hit@1"]),
                    _fmt_number(row["mrr"]),
                    _fmt_number(row["not_found"]),
                    _fmt_number(row["latency_median"]),
                    str(row["n"]),
                    ", ".join(notes) if notes else "—",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_number(value: float | None) -> str:
    """Форматирует число метрики: три знака или ``—`` для ``None``."""
    return f"{value:.3f}" if value is not None else "—"


def _ab_verdicts_section(
    all_ab_verdicts: list[dict],
    runs: list[dict],
    cache: ResultCache,
    output_dir: Path,
) -> str:
    """Секция с A/B вердиктами: сводка и подробности в ``<details>``.

    Сводка показывает одну строку на пару вариантов и только значимые
    отличия (``p < 0.05``); полная таблица по метрикам сворачивается.

    Args:
        all_ab_verdicts: Все A/B вердикты из :func:`load_all_ab_verdicts`.
        runs: Прогоны из :func:`collect_runs` (для описания изменений).
        cache: Хранилище результатов.
        output_dir: Каталог документа.

    Returns:
        str: Markdown секция с A/B вердиктами или пустая строка.
    """
    # Все A/B сравнения по метрикам (для подробной таблицы)
    ab_comparisons: list[dict] = []
    for av in all_ab_verdicts:
        for v in av["verdicts"]:
            arrow = {"better": "лучше", "worse": "хуже", "no_diff": "без разницы"}
            ab_comparisons.append(
                {
                    "date": av["created_at"],
                    "environment": av.get("environment_desc", "") or av["environment"],
                    "base": av["base_variant"],
                    "variant": av["variant"],
                    "metric": v.metric,
                    "base_val": f"{v.base_value:.3f}",
                    "var_val": f"{v.variant_value:.3f}",
                    "delta": f"{v.variant_value - v.base_value:+.3f}",
                    "p_value": f"{v.p_value:.3f}",
                    "verdict": arrow.get(v.direction, v.direction),
                    "test": v.test,
                }
            )

    groups = _ab_groups(all_ab_verdicts, runs)
    blocks: list[str] = []

    if groups:
        blocks += [
            "",
            "## A/B сравнения (сводка)",
            "",
            "По одной строке на пару вариантов. Показаны только значимые "
            "отличия (p < 0.05); остальные метрики не выводятся, чтобы не "
            "засорять таблицу константами.",
            "",
            "| дата | тест (base → variant) | что изменилось | итог | значимые отличия (p < 0.05) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for group in groups:
            deltas = (
                "; ".join(_verdict_line(v) for v in group["significant"])
                or "—"
            )
            blocks.append(
                f"| {_format_date(group['date'])} | "
                f"`{group['base']}` → `{group['variant']}` | "
                f"{group['essence']} | **{_group_verdict(group['significant'])}** | "
                f"{deltas} |"
            )
        blocks.append("")

    if ab_comparisons:
        ab_comparisons.sort(key=lambda x: x["date"], reverse=True)
        detail_lines = [
            "| дата | окружение | base | variant | метрика | base | variant | Δ | p-value | вердикт | тест |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in ab_comparisons:
            detail_lines.append(
                f"| {_format_date(c['date'])} | {c['environment']} | {c['base']} | "
                f"{c['variant']} | {c['metric']} | {c['base_val']} | {c['var_val']} | "
                f"{c['delta']} | {c['p_value']} | {c['verdict']} | {c['test']} |"
            )
        blocks += [
            "<details>",
            f"<summary>все сравнения по метрикам ({len(ab_comparisons)})</summary>",
            "",
            "«Δ» — разница variant - base. Вердикт: «лучше» — значимое улучшение "
            "(p < 0.05), «хуже» — ухудшение, «без разницы» — нет значимых отличий.",
            "",
            *detail_lines,
            "",
            "</details>",
            "",
        ]

    return "\n".join(blocks)


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
| `uv run eval tune --params '{"retrieval_top_k": [3, 5, 8]}' --objective hit@1` | перебор по сетке параметров |
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

## Бутстрап-выборки

Один прогон на наборе даёт точечные метрики без представления о разбросе.
Флаг `--bootstraps N` (в `run` и `ab`) ресемплирует пул вопросов **с
возвратом**: вариант N раз прогоняется на детерминированной выборке того же
размера, состав выбирается общим зерном `--seed`, поэтому для всех вариантов
на одной выборке вопросы совпадают. Итоговые метрики — среднее по выборкам,
разброс печатается как `± std`. A/B-сравнение при бутстрепе — по парным
разностям `variant_k − base_k` по выборкам (Уилкоксон, без scipy —
пермутационный тест). Каждая выборка кэшируется отдельно, так что добивание
уже снятых точек ничего не переснимает.

## Тюнинг по сетке

`eval tune` перебирает декартову сетку параметров и сортирует результаты по
целевой метрике — «ручной» шаг поиска оптимума перед точечным A/B:

    uv run eval tune --params '{"retrieval_top_k": [3, 5, 8], "rerank_candidates": [5, 10, 15]}' \
        --objective hit@1 --direction max --bootstraps 5

Параметры — реальные поля настроек (модель, `retrieval_top_k`,
`rerank_candidates`, `rerank_enabled` и т.д.), имя точки собирается из её
значений. Каждая точка кэшируется, поэтому расширение сетки доганяет только
новые комбинации. `--max-points` (по умолчанию 100) защищает от гигантского
перебора, `--output` пишет полный отчёт с метриками всех точек в JSON.
"""


def render_markdown(runs: list[dict], all_ab_verdicts: list[dict], cache: ResultCache, output_dir: Path) -> str:
    """Собирает текст документа целиком.

    Args:
        runs: Прогоны из :func:`collect_runs`.
        all_ab_verdicts: Все A/B вердикты из :func:`load_all_ab_verdicts`.
        cache: Хранилище результатов.
        output_dir: Каталог документа (для относительных ссылок).

    Returns:
        str: Текст docs/BENCHMARK.md.
    """
    ab_groups = _ab_groups(all_ab_verdicts, runs)
    recommendations = _recommendations_section(runs, ab_groups)
    leaderboard = _leaderboard_section(runs)

    lines = [
        "# Бенчмарк качества",
        "",
        "Измерение качества поиска и ответов на золотом наборе, A/B-сравнение",
        "вариантов и история результатов. Этот документ генерируется командой",
        "`uv run eval report` из кэша `eval/results/`.",
        "",
    ]
    if recommendations:
        lines.append(recommendations)
    if leaderboard:
        lines.append(leaderboard)
    lines += [_STATIC_SECTIONS.strip(), ""]

    lines += [
        "## История результатов",
        "",
        "Каждая строка — один прогон: группа вариантов, снятых на одном наборе,",
        "окружении и конфиге. Сырые строки свёрнуты — обычно нужны только",
        "«Рекомендации» и «Лидерборд» выше.",
        "",
    ]

    if not runs:
        lines += [
            "Кэш пуст — прогоните `uv run eval run --with-answers` и обновите отчёт.",
            "",
        ]
    else:
        history = [
            "| дата | окружение | варианты | метрики | вердикт | кэш |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for run in runs:
            history.append(
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
        lines += [
            "<details>",
            f"<summary>сырые прогоны ({len(runs)})</summary>",
            "",
            *history,
            "",
            "</details>",
            "",
        ]

    ab_section = _ab_verdicts_section(all_ab_verdicts, runs, cache, output_dir)
    if ab_section:
        lines.append(ab_section)

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
    runs, all_ab_verdicts = collect_runs(cache)
    text = render_markdown(runs, all_ab_verdicts, cache, output_path.parent)
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
