"""Тесты генератора docs/BENCHMARK.md.

Проверяется восстановление метрик из ключа, группировка записей кэша в
прогоны, содержимое раздела «История результатов» (включая вердикт A/B и
относительные ссылки на JSON) и запись файла.
"""

from __future__ import annotations

from eval.cache import ResultCache, build_entry, cache_key
from eval.config import Environment, Variant
from eval.metrics import MRR, HitAtK, compute_metrics
from eval.report import (
    _ab_groups,
    _aggregate_configs,
    _env_diff,
    _human_config,
    collect_runs,
    generate,
    load_all_ab_verdicts,
    metric_from_key,
    render_markdown,
)
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

    def test_recommendations_and_leaderboard_present(self, tmp_path) -> None:
        """Сводка и лидерборд генерируются при наличии данных."""
        cache = ResultCache(tmp_path / "results")
        env_on = Environment(
            platform="test",
            python="3.14",
            llm_model="m",
            llm_support_model="s",
            embedding_model="e",
            retrieval_top_k=5,
            rerank_enabled=True,
            rerank_model="",
            grading_enabled=False,
            verify_numbers=True,
        )
        env_off = Environment(
            platform="test",
            python="3.14",
            llm_model="m",
            llm_support_model="s",
            embedding_model="e",
            retrieval_top_k=5,
            rerank_enabled=False,
            rerank_model="",
            grading_enabled=False,
            verify_numbers=True,
        )
        _save_with(cache, Variant(name="base"), env_on, [1] * 6, "d")
        _save_with(cache, Variant(name="novariant"), env_off, [None] * 6, "d")

        runs, _ = collect_runs(cache)
        text = render_markdown(runs, [], cache, tmp_path)
        assert "## Рекомендации" in text
        assert "## Лидерборд" in text
        assert "Текущий лидер" in text
        assert "максимальное качество" in text


def _save_with(cache, variant, env, ranks, dataset) -> None:
    """Кладёт запись в кэш с произвольным окружением и набором."""
    result = make_result(ranks)
    result.variant = variant.name
    result.created_at = "2026-01-01T10:00:00+00:00"
    result.metrics = compute_metrics([HitAtK(1), MRR()], result.outcomes)
    key = cache_key(variant, dataset, METRICS, env, CONFIG)
    cache.save(build_entry(key, variant, result, env, dataset, METRICS, CONFIG))


class TestReportHelpers:
    """Вспомогательные функции сводного отчёта."""

    def test_aggregate_group_by_environment(self, tmp_path) -> None:
        """Одна конфигурация агрегируется в одну строку."""
        cache = ResultCache(tmp_path)
        _save(cache, Variant(name="base"), _result([1, 2], "base", "d-1"), dataset="a")
        _save(cache, Variant(name="rep"), _result([2, 1], "rep", "d-1"), dataset="b")
        runs, _ = collect_runs(cache)
        rows = _aggregate_configs(runs)
        assert len(rows) == 1
        assert rows[0]["n"] == 2

    def test_human_config_shortens_embedder(self) -> None:
        """Полное имя эмбеддера сокращается в лидерборде."""
        desc = "llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=вкл"
        label = _human_config(desc)
        assert "Qwen3" not in label and "qwen3-emb" in label
        assert "gpt-oss:20b" in label

    def test_human_config_keeps_collection(self) -> None:
        """Коллекция не теряется в метке конфигурации."""
        desc = (
            "llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=вкл, "
            "cand=10, coll=aist_chunks_v2"
        )
        label = _human_config(desc)
        assert "колл.=aist_chunks_v2" in label

    def test_env_diff_detects_only_change(self) -> None:
        """Различаются только изменённые параметры."""
        base = "llm=m, emb=e, top_k=5, rerank=вкл"
        variant = "llm=m, emb=e, top_k=5, rerank=выкл"
        diff = _env_diff(base, variant)
        assert "реранк: вкл → выкл" in diff
        assert "LLM" not in diff

    def test_ab_groups_filters_non_significant(self) -> None:
        """В значимые отличия попадают только вердикты с p < 0.05."""
        from eval.result import Verdict

        av = {
            "base_variant": "base",
            "variant": "top",
            "verdicts": [
                Verdict("hit@1", "mcnemar", 0.03, 0.7, 0.5, "worse"),
                Verdict("mrr", "wilcoxon", 1.0, 0.7, 0.7, "no_diff"),
            ],
            "environment": "env1",
            "environment_desc": "llm=m, rerank=вкл",
            "created_at": "2026-01-01T10:00:00+00:00",
        }
        groups = _ab_groups([av], [])
        assert len(groups) == 1
        assert [v.metric for v in groups[0]["significant"]] == ["hit@1"]


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
