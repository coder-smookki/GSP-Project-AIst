"""Тесты генератора docs/BENCHMARK.md.

Проверяется восстановление метрик из ключа, группировка записей кэша в
прогоны, содержимое раздела «История результатов» (включая вердикт A/B и
относительные ссылки на JSON) и запись файла.
"""

from __future__ import annotations

from eval.cache import ResultCache, build_entry, cache_key
from eval.config import Variant
from eval.metrics import MRR, HitAtK, compute_metrics
from eval.report import collect_runs, generate, metric_from_key, render_markdown, load_all_ab_verdicts
from tests.eval.test_eval_cache import CONFIG, ENV, METRICS
from tests.eval.test_eval_stats import make_result


def _result(ranks: list, variant: str, created: str):
    """Результат с подсчитанными метриками и именем варианта."""
    result = make_result(ranks)
    result.variant = variant
    result.created_at = created
    result.metrics = compute_metrics([HitAtK(1), MRR()], result.outcomes)
    return result


def _save(cache: ResultCache, variant: Variant, result, dataset: str) -> None:
    """Кладёт запись в кэш с типовым окружением и метриками."""
    key = cache_key(variant, dataset, METRICS, ENV, CONFIG)
    cache.save(build_entry(key, variant, result, ENV, dataset, METRICS, CONFIG))


class TestMetricFromKey:
    """Восстановление метрик из ключа кэша."""

    def test_hit_at_k(self) -> None:
        """hit@3 — HitAtK(3)."""
        metric = metric_from_key("hit@3")
        assert isinstance(metric, HitAtK)
        assert metric.k == 3

    def test_known_keys(self) -> None:
        """Известные ключи возвращают метрики."""
        assert metric_from_key("mrr").key == "mrr"

    def test_unknown_key(self) -> None:
        """Неизвестный ключ — None."""
        assert metric_from_key("strange") is None


class TestCollectRuns:
    """Группировка записей кэша в прогоны."""

    def test_groups_by_dataset(self, tmp_path) -> None:
        """Разные наборы — разные прогоны."""
        cache = ResultCache(tmp_path)
        _save(
            cache,
            Variant(name="base"),
            _result([1, 2], "base", "2026-01-01T10:00:00+00:00"),
            dataset="abc",
        )
        _save(
            cache,
            Variant(name="top6", settings={"retrieval_top_k": 6}),
            _result([2, 1], "top6", "2026-01-01T10:05:00+00:00"),
            dataset="abc",
        )
        _save(
            cache,
            Variant(name="base"),
            _result([1, 2], "base", "2026-01-02T10:00:00+00:00"),
            dataset="def",
        )

        runs, _ = collect_runs(cache)
        assert len(runs) == 2
        paired = [run for run in runs if len(run["variants"]) == 2]
        single = [run for run in runs if len(run["variants"]) == 1]
        assert paired[0]["variants"] == ["base", "top6"]
        assert single[0]["variants"] == ["base"]

    def test_newest_first(self, tmp_path) -> None:
        """Прогоны отсортированы от свежих к старым."""
        cache = ResultCache(tmp_path)
        _save(
            cache,
            Variant(name="base"),
            _result([1, 2], "base", "2026-01-02T10:00:00+00:00"),
            dataset="b",
        )
        _save(
            cache,
            Variant(name="base"),
            _result([1, 2], "base", "2026-01-01T10:00:00+00:00"),
            dataset="a",
        )
        runs, _ = collect_runs(cache)
        assert runs[0]["created_at"].startswith("2026-01-02")

    def test_empty_cache(self, tmp_path) -> None:
        """Пустой кэш — пустой список."""
        runs, _ = collect_runs(ResultCache(tmp_path))
        assert runs == []


class TestRenderMarkdown:
    """Содержимое документа."""

    def test_static_sections_present(self, tmp_path) -> None:
        """Все разделы документа на месте."""
        cache = ResultCache(tmp_path)
        text = render_markdown([], [], cache, tmp_path)
        for heading in (
            "## Как пользоваться",
            "## Метрики",
            "## A/B-тест",
            "## История результатов",
        ):
            assert heading in text

    def test_history_table_with_verdict(self, tmp_path) -> None:
        """Строка прогона содержит варианты, метрики, вердикт и ссылку на JSON."""
        cache = ResultCache(tmp_path / "results")
        base = _result(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "base", "2026-01-01T10:00:00+00:00"
        )
        top6 = _result([None] * 10, "top6", "2026-01-01T10:05:00+00:00")
        _save(cache, Variant(name="base"), base, dataset="abc")
        _save(
            cache,
            Variant(name="top6", settings={"retrieval_top_k": 6}),
            top6,
            dataset="abc",
        )

        runs, _ = collect_runs(cache)
        ab_verdicts = load_all_ab_verdicts(cache)
        text = render_markdown(runs, ab_verdicts, cache, tmp_path)
        assert "| дата | окружение | варианты | метрики | вердикт | кэш |" in text
        assert "base, top6" in text
        assert "hit@1" in text
        assert "хуже" in text
        assert f"results/{runs[0]['keys'][0]}.json" in text

    def test_empty_history(self, tmp_path) -> None:
        """Пустой кэш — подсказка вместо таблицы."""
        cache = ResultCache(tmp_path)
        text = render_markdown([], [], cache, tmp_path)
        assert "Кэш пуст" in text


class TestGenerate:
    """Запись файла отчёта."""

    def test_writes_file(self, tmp_path) -> None:
        """Отчёт пишется по указанному пути."""
        cache = ResultCache(tmp_path)
        _save(
            cache,
            Variant(name="base"),
            _result([1, 2], "base", "2026-01-01T10:00:00+00:00"),
            dataset="abc",
        )
        output = tmp_path / "BENCHMARK.md"
        assert generate(cache, output) == 0
        text = output.read_text(encoding="utf-8")
        assert text.startswith("# Бенчмарк качества")
        assert "История результатов" in text
