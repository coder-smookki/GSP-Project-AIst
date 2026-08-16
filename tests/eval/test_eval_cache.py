"""Тесты кэша результатов.

Ключевые свойства: детерминизм ключа (одинаковый конфиг → одинаковый ключ),
чувствительность ключа к каждому из входов, круговая сериализация результата,
метаданные и очистка.
"""

from __future__ import annotations

import json

from eval.cache import (
    ResultCache,
    build_entry,
    cache_key,
    file_hash,
    metrics_version,
)
from eval.config import Environment, RunConfig, Variant
from eval.metrics import MRR, HitAtK
from tests.eval.test_eval_stats import make_result

ENV = Environment(
    platform="test",
    python="3.14",
    llm_model="m1",
    llm_support_model="m2",
    embedding_model="e1",
    retrieval_top_k=5,
    rerank_enabled=False,
    rerank_model="r",
    grading_enabled=False,
    verify_numbers=True,
)
METRICS = [HitAtK(1), HitAtK(3), MRR()]
CONFIG = RunConfig(runner="golden")
DATASET = "abc123"


def key_for(
    variant: Variant | None = None,
    env: Environment = ENV,
    metrics: list | None = None,
    config: RunConfig = CONFIG,
    dataset: str = DATASET,
) -> str:
    """Ключ с типовыми параметрами."""
    return cache_key(
        variant if variant is not None else Variant(name="base"),
        dataset,
        metrics if metrics is not None else METRICS,
        env,
        config,
    )


class TestCacheKey:
    """Детерминизм и чувствительность ключа."""

    def test_deterministic(self) -> None:
        """Одинаковый ввод — одинаковый ключ."""
        assert key_for() == key_for()

    def test_changes_with_settings(self) -> None:
        """Изменение оверрайдов варианта меняет ключ."""
        variant = Variant(name="base", settings={"retrieval_top_k": 6})
        assert key_for() != key_for(variant=variant)

    def test_changes_with_dataset(self) -> None:
        """Изменение хеша набора меняет ключ."""
        assert key_for() != key_for(dataset="other")

    def test_changes_with_metrics(self) -> None:
        """Изменение набора метрик меняет ключ."""
        assert key_for() != key_for(metrics=[HitAtK(1)])

    def test_changes_with_environment(self) -> None:
        """Изменение окружения меняет ключ."""
        other = Environment(
            platform="test",
            python="3.14",
            llm_model="m1",
            llm_support_model="m2",
            embedding_model="e1",
            retrieval_top_k=5,
            rerank_enabled=True,
            rerank_model="r",
            grading_enabled=False,
            verify_numbers=True,
        )
        assert key_for() != key_for(env=other)

    def test_changes_with_config(self) -> None:
        """Изменение параметров прогона меняет ключ."""
        config = RunConfig(runner="golden", limit=10)
        assert key_for() != key_for(config=config)

    def test_graph_changes_key(self) -> None:
        """Разный граф агента меняет ключ: кэши графов не смешиваются."""
        config = RunConfig(runner="golden", graph="rd-v2")
        assert key_for() != key_for(config=config)

    def test_bootstrap_sample_split_keys(self) -> None:
        """Разные бутстрап-выборки не должны делиться одним кэшем."""
        base = RunConfig(runner="golden", bootstraps=3, sample_index=0)
        other = RunConfig(runner="golden", bootstraps=3, sample_index=1)
        assert key_for(config=base) != key_for(config=other)
        assert key_for(config=RunConfig(runner="golden", seed=1)) != key_for()

    def test_bootstraps_count_in_digest(self) -> None:
        """Число выборок участвует в разграничении ключа."""
        one = RunConfig(runner="golden", bootstraps=1)
        many = RunConfig(runner="golden", bootstraps=5)
        assert key_for(config=one) != key_for(config=many)


class TestVersioning:
    """Версия метрик и хеш набора."""

    def test_metrics_version_sorted(self) -> None:
        """Версия не зависит от порядка метрик."""
        assert metrics_version([MRR(), HitAtK(1)]) == metrics_version(
            [HitAtK(1), MRR()]
        )

    def test_file_hash_changes_with_content(self, tmp_path) -> None:
        """Хеш файла реагирует на содержимое."""
        path = tmp_path / "q.json"
        path.write_text("one", encoding="utf-8")
        first = file_hash(path)
        path.write_text("two", encoding="utf-8")
        assert file_hash(path) != first


class TestResultCache:
    """Сохранение и чтение записей."""

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """Сохранённый результат восстанавливается без потерь."""
        cache = ResultCache(tmp_path)
        result = make_result([1, 2, None])
        entry = build_entry(
            key_for(),
            Variant(name="base"),
            result,
            ENV,
            DATASET,
            METRICS,
            CONFIG,
        )
        cache.save(entry)
        restored = cache.load(key_for())
        assert restored is not None
        assert restored == result

    def test_load_missing_returns_none(self, tmp_path) -> None:
        """Отсутствующая запись — ``None``."""
        assert ResultCache(tmp_path).load("no-such-key") is None

    def test_metadata_lists_saved(self, tmp_path) -> None:
        """Метаданные показывают сохранённые результаты."""
        cache = ResultCache(tmp_path)
        result = make_result([1, 2])
        entry = build_entry(
            key_for(), Variant(name="base"), result, ENV, DATASET, METRICS, CONFIG
        )
        cache.save(entry)
        rows = cache.metadata()
        assert len(rows) == 1
        assert rows[0]["variant"] == "base"
        assert rows[0]["key"] == key_for()

    def test_clear_removes_all(self, tmp_path) -> None:
        """Очистка удаляет все записи."""
        cache = ResultCache(tmp_path)
        result = make_result([1, 2])
        cache.save(
            build_entry(
                key_for(), Variant(name="base"), result, ENV, DATASET, METRICS, CONFIG
            )
        )
        cache.save(
            build_entry(
                key_for(variant=Variant(name="other")),
                Variant(name="other"),
                result,
                ENV,
                DATASET,
                METRICS,
                CONFIG,
            )
        )
        assert cache.clear() == 2
        assert cache.metadata() == []

    def test_entry_is_valid_json(self, tmp_path) -> None:
        """Запись кэша — валидный JSON с результатом внутри."""
        entry = build_entry(
            key_for(),
            Variant(name="base"),
            make_result([1, 2]),
            ENV,
            DATASET,
            METRICS,
            CONFIG,
        )
        payload = json.dumps(entry, ensure_ascii=False)
        restored = json.loads(payload)
        assert restored["variant"]["name"] == "base"
        assert "outcomes" in restored["result"]


class TestGraphSeparation:
    """Граф агента разделяет кэш и вердикты."""

    def test_build_entry_records_graph(self) -> None:
        """Запись кэша хранит тег графа агента."""
        config = RunConfig(runner="golden", graph="rd-v2")
        entry = build_entry(
            key_for(config=config),
            Variant(name="base"),
            make_result([1, 2]),
            ENV,
            DATASET,
            METRICS,
            config,
        )
        assert entry["graph"] == "rd-v2"
        assert entry["config"]["graph"] == "rd-v2"

    def test_config_digest_distinguishes_graph(self) -> None:
        """Дайджест конфига включает граф: без этого A/B смешал бы графы."""
        base = key_for()
        other = key_for(config=RunConfig(runner="golden", graph="base"))
        assert base == other
        assert base != key_for(config=RunConfig(runner="golden", graph="rd-v2"))

    def test_ab_cache_key_isolates_graphs(self) -> None:
        """Ключ A/B-вердиктов различается по графу."""
        from eval.cache import ab_cache_key

        base = Variant(name="base")
        variant = Variant(name="v", settings={"retrieval_top_k": 6})
        a = ab_cache_key(base, variant, DATASET, METRICS, ENV, CONFIG)
        b = ab_cache_key(
            base,
            variant,
            DATASET,
            METRICS,
            ENV,
            RunConfig(runner="golden", graph="rd-v2"),
        )
        assert a != b

    def test_build_ab_entry_records_graph(self) -> None:
        """A/B-запись хранит граф, чтобы вердикты не смешивались."""
        from eval.cache import ab_cache_key, build_ab_entry

        base = Variant(name="base")
        variant = Variant(name="v", settings={"retrieval_top_k": 6})
        config = RunConfig(runner="golden", graph="rd-v2")
        key = ab_cache_key(base, variant, DATASET, METRICS, ENV, config)
        entry = build_ab_entry(
            key, base, variant, [], ENV, DATASET, METRICS, config
        )
        assert entry["graph"] == "rd-v2"
        assert entry["config"]["graph"] == "rd-v2"
