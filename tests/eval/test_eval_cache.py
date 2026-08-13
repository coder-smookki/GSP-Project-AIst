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
