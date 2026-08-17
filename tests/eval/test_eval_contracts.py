"""Тесты модели данных и контрактов бенчмаркинга.

Детерминизм ключей (вариант, окружение) — фундамент кэша: если ключ
плавает между запусками, кэш бесполезен. Контракты проверяются на то, что
протоколы распознают реализации (runtime_checkable), а не на детали.
"""

from __future__ import annotations

import pytest

from eval.config import Environment, RunConfig, Variant, snapshot_environment
from eval.contracts import Cache, Metric, Reporter, Runner, StatTest
from eval.result import Outcome, RunResult, Verdict


class TestVariant:
    """Идентичность варианта."""

    def test_identity_is_deterministic(self) -> None:
        """Одинаковый вариант даёт одинаковый ключ."""
        a = Variant(name="base", settings={"retrieval_top_k": 5})
        b = Variant(name="base", settings={"retrieval_top_k": 5})
        assert a.identity() == b.identity()
        assert len(a.identity()) == 16

    def test_identity_differs_on_settings(self) -> None:
        """Изменение настроек меняет ключ."""
        a = Variant(name="base", settings={"retrieval_top_k": 5})
        b = Variant(name="base", settings={"retrieval_top_k": 6})
        assert a.identity() != b.identity()

    def test_identity_differs_on_name(self) -> None:
        """Разные имена — разные ключи, даже при тех же настройках."""
        a = Variant(name="base", settings={"retrieval_top_k": 5})
        b = Variant(name="new", settings={"retrieval_top_k": 5})
        assert a.identity() != b.identity()

    def test_identity_ignores_dict_order(self) -> None:
        """Порядок оверрайдов не влияет на ключ."""
        a = Variant(name="x", settings={"a": 1, "b": 2})
        b = Variant(name="x", settings={"b": 2, "a": 1})
        assert a.identity() == b.identity()


class TestEnvironment:
    """Отпечаток окружения."""

    def test_fingerprint_is_deterministic(self) -> None:
        """Одинаковое окружение — одинаковый ключ."""
        fields = dict(
            platform="Darwin-arm64",
            python="3.14.0",
            llm_model="m",
            llm_support_model="s",
            embedding_model="e",
            retrieval_top_k=5,
            rerank_enabled=False,
            rerank_model="",
            grading_enabled=False,
            verify_numbers=True,
            retrieval_max_requests=10,
            retrieval_branch_retries=1,
        )
        assert (
            Environment(**fields).fingerprint() == Environment(**fields).fingerprint()
        )

    def test_fingerprint_sensitive_to_model(self) -> None:
        """Смена модели меняет ключ."""
        base = dict(
            platform="Darwin-arm64",
            python="3.14.0",
            llm_model="m",
            llm_support_model="s",
            embedding_model="e",
            retrieval_top_k=5,
            rerank_enabled=False,
            rerank_model="",
            grading_enabled=False,
            verify_numbers=True,
            retrieval_max_requests=10,
            retrieval_branch_retries=1,
        )
        other = {**base, "llm_model": "m2"}
        assert Environment(**base).fingerprint() != Environment(**other).fingerprint()


class _Settings:
    """Минимально совместимый с Settings объект."""

    def __init__(self, model: str = "m") -> None:
        self.llm_provider = "ollama-local"
        self.llm_model = model
        self.llm_support_model = "s"
        self.llm_reasoning = False
        self.embedding_provider = "ollama-local"
        self.embedding_model = "e"
        self.retrieval_top_k = 5
        self.retrieval_prefetch = 30
        self.retrieval_rrf_k = 60
        self.retrieval_min_relevance = 0.55
        self.rerank_enabled = False
        self.rerank_model = ""
        self.rerank_candidates = 10
        self.grading_enabled = False
        self.verify_numbers = True
        self.qdrant_collection = ""
        self.retrieval_max_requests = 10
        self.retrieval_branch_retries = 1


class TestSnapshotEnvironment:
    """Съём отпечатка из настроек."""

    def test_builds_from_settings_like_object(self) -> None:
        """Отпечаток собирается из атрибутов настроек."""
        env = snapshot_environment(_Settings(model="qwen"))
        assert env.llm_model == "qwen"
        assert env.platform and env.python

    def test_fingerprint_available(self) -> None:
        """Отпечаток сразу пригоден как ключ кэша."""
        env = snapshot_environment(_Settings())
        assert env.fingerprint()

    def test_new_params_in_describe(self) -> None:
        """Параметры из .env попадают в описание окружения."""
        env = snapshot_environment(_Settings())
        assert "prefetch=30" in env.describe()
        assert "rrf=60" in env.describe()
        assert "cand=10" in env.describe()
        assert "min_rel=0.55" in env.describe()

    def test_retrieval_budget_in_describe(self) -> None:
        """Бюджет запросов и повторы попадают в описание окружения."""
        env = snapshot_environment(_Settings())
        assert "max_req=10" in env.describe()
        assert "retries=1" in env.describe()

    def test_prefetch_changes_fingerprint(self) -> None:
        """Смена prefetch меняет ключ кэша."""
        base = _Settings()
        other = _Settings()
        other.retrieval_prefetch = 60
        assert snapshot_environment(base).fingerprint() != snapshot_environment(
            other
        ).fingerprint()

    def test_reasoning_changes_fingerprint(self) -> None:
        """Включение размышлений меняет ключ кэша."""
        base = _Settings()
        other = _Settings()
        other.llm_reasoning = True
        assert snapshot_environment(base).fingerprint() != snapshot_environment(
            other
        ).fingerprint()

    def test_collection_in_describe(self) -> None:
        """Коллекция попадает в описание окружения."""
        settings = _Settings()
        settings.qdrant_collection = "aist_chunks_v2"
        assert "coll=aist_chunks_v2" in snapshot_environment(settings).describe()

    def test_collection_changes_fingerprint(self) -> None:
        """Разные коллекции не должны делиться одним кэшем."""
        base = _Settings()
        other = _Settings()
        other.qdrant_collection = "aist_chunks_v2"
        assert snapshot_environment(base).fingerprint() != snapshot_environment(
            other
        ).fingerprint()

    def test_max_requests_changes_fingerprint(self) -> None:
        """Смена бюджета запросов меняет ключ кэша."""
        base = _Settings()
        other = _Settings()
        other.retrieval_max_requests = 20
        assert snapshot_environment(base).fingerprint() != snapshot_environment(
            other
        ).fingerprint()

    def test_branch_retries_changes_fingerprint(self) -> None:
        """Смена числа повторов ветки меняет ключ кэша."""
        base = _Settings()
        other = _Settings()
        other.retrieval_branch_retries = 2
        assert snapshot_environment(base).fingerprint() != snapshot_environment(
            other
        ).fingerprint()


class TestRunResult:
    """Сериализация результата."""

    def test_round_trip(self) -> None:
        """to_dict/from_dict не теряет данные."""
        result = RunResult(
            variant="base",
            environment="env1",
            metrics={"hit@1": 0.5},
            outcomes=[Outcome(question="q", rank=1, returned=5, seconds=0.1)],
            meta={"seed": 42},
        )
        restored = RunResult.from_dict(result.to_dict())
        assert restored == result

    def test_to_json_keeps_cyrillic(self) -> None:
        """JSON сохраняет кириллицу читаемо."""
        result = RunResult(
            variant="base",
            environment="env1",
            metrics={},
            outcomes=[Outcome(question="какой срок?")],
        )
        assert "какой срок?" in result.to_json()


class TestVerdict:
    """Вердикт статистического теста."""

    def test_frozen(self) -> None:
        """Вердикт неизменяем."""
        verdict = Verdict("hit@1", "mcnemar", 0.01, 0.5, 0.7, "better")
        with pytest.raises(AttributeError):
            verdict.direction = "worse"


class TestRunConfig:
    """Дефолты конфига прогона."""

    def test_defaults(self) -> None:
        """Конфиг сразу готов к прогону."""
        config = RunConfig()
        assert config.repeats == 2
        assert config.concurrency == (1, 2, 4)
        assert config.limit is None


class TestProtocols:
    """Протоколы распознают реализации."""

    def test_metric_protocol(self) -> None:
        """Метрика реализуется классом с evaluate."""

        class _Impl:
            key = "hit@1"
            display_name = "Hit@1"
            kind = "categorical"

            def evaluate(self, outcomes) -> float:
                return 1.0

        assert isinstance(_Impl(), Metric)

    def test_runner_protocol(self) -> None:
        """Раннер реализуется классом с run."""

        class _Impl:
            name = "golden"

            async def run(self, variant, config) -> RunResult:
                return RunResult(
                    variant=variant.name, environment="e", metrics={}, outcomes=[]
                )

        assert isinstance(_Impl(), Runner)

    def test_cache_protocol(self) -> None:
        """Кэш реализуется классом с get/put."""

        class _Impl:
            def get(self, key):
                return None

            def put(self, key, result):
                return None

        assert isinstance(_Impl(), Cache)

    def test_stat_test_protocol(self) -> None:
        """Тест реализуется классом с compare."""

        class _Impl:
            name = "mcnemar"

            def compare(self, base, variant):
                return []

        assert isinstance(_Impl(), StatTest)

    def test_reporter_protocol(self) -> None:
        """Репортёр реализуется классом с render."""

        class _Impl:
            def render(self, results, verdicts):
                return ""

        assert isinstance(_Impl(), Reporter)
