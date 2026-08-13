"""Тесты операционного замера (eval.benchmark).

Проверяются только управляющие потоки: mock-защита, выбор конфигурации прогона,
печать таблиц и запись JSON. Сам замер покрыт тестами eval.runners, поэтому
здесь `OperationalRunner` подменяется фейком, и живая инфраструктура не нужна.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from eval import benchmark
from eval.config import RunConfig, Variant


class _FakeSettings:
    """Настройки, достаточные для печати окружения."""

    llm_model = "model-x"
    llm_support_model = "support-y"
    embedding_model = "emb-z"
    retrieval_top_k = 5
    rerank_enabled = False
    llm_reasoning = False

    def resolved_retrieval_provider(self) -> str:
        return "qdrant"


def _fake_result(concurrency: bool = True) -> SimpleNamespace:
    """Результат прогона с готовыми замерами."""
    return SimpleNamespace(
        meta={
            "sequential": [
                {
                    "вид": "короткий факт",
                    "мин": 1.0,
                    "медиана": 1.5,
                    "макс": 2.0,
                    "символов": 120,
                    "отказ": False,
                },
                {
                    "вид": "отказ",
                    "мин": 0.5,
                    "медиана": 0.7,
                    "макс": 0.9,
                    "символов": 40,
                    "отказ": True,
                },
            ],
            "concurrency": (
                [
                    {
                        "одновременно": 1,
                        "всего_с": 5.0,
                        "худший_ответ_с": 2.0,
                        "ответов_в_минуту": 12.0,
                    },
                    {
                        "одновременно": 2,
                        "всего_с": 8.0,
                        "худший_ответ_с": 4.0,
                        "ответов_в_минуту": 15.0,
                    },
                ]
                if concurrency
                else []
            ),
        },
        metrics={"latency_median": 1.5, "answers_per_minute": 40.0},
    )


class _FakeRunner:
    """Подмена `OperationalRunner`: отдаёт готовый результат, не измеряя."""

    def __init__(self, result=None) -> None:
        self.calls: list[tuple[Variant, RunConfig]] = []
        self.result = result if result is not None else _fake_result()

    async def run(self, variant: Variant, config: RunConfig):
        self.calls.append((variant, config))
        return self.result


def _namespace(**overrides) -> argparse.Namespace:
    """Аргументы команды с типовыми значениями."""
    defaults = {"repeats": 2, "concurrency": [1, 2], "output": None}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patch(monkeypatch, runner=None, provider="qdrant"):
    """Подменяет настройки и раннер; возвращает фейковый раннер."""
    settings = _FakeSettings()
    settings.resolved_retrieval_provider = lambda: provider
    monkeypatch.setattr("infrastructure.config.settings.get_settings", lambda: settings)
    if runner is None:
        runner = _FakeRunner()
    monkeypatch.setattr(benchmark, "OperationalRunner", lambda *a, **k: runner)
    return runner


class TestBenchmarkMainAsync:
    """Управляющие потоки операционного замера."""

    async def test_mock_guard_short_circuits(self, monkeypatch) -> None:
        """На заглушке поиска раннер не запускается."""
        runner = _patch(monkeypatch, provider="mock")
        code = await benchmark.main_async(_namespace())
        assert code == 2
        assert runner.calls == []

    async def test_runs_with_operational_config(self, monkeypatch, capsys) -> None:
        """Прогон идёт через OperationalRunner на базовом варианте."""
        runner = _patch(monkeypatch)
        code = await benchmark.main_async(_namespace(repeats=3, concurrency=[1, 2]))
        assert code == 0
        assert len(runner.calls) == 1
        variant, config = runner.calls[0]
        assert variant.name == "base"
        assert config.runner == "operational"
        assert config.repeats == 3
        assert config.concurrency == (1, 2)

        out = capsys.readouterr().out
        assert "ОКРУЖЕНИЕ" in out
        assert "model-x" in out
        assert "ВРЕМЯ ОТВЕТА" in out
        assert "короткий факт" in out
        assert "ОДНОВРЕМЕННЫЕ ОБРАЩЕНИЯ" in out

    async def test_empty_concurrency_skips_section(self, monkeypatch, capsys) -> None:
        """Пустой concurrency — секция одновременных обращений не печатается."""
        fake = _FakeRunner(_fake_result(concurrency=False))
        _patch(monkeypatch, runner=fake)
        code = await benchmark.main_async(_namespace(concurrency=[]))
        assert code == 0
        out = capsys.readouterr().out
        assert "ВРЕМЯ ОТВЕТА" in out
        assert "ОДНОВРЕМЕННЫЕ" not in out

    async def test_output_is_written(self, monkeypatch, tmp_path) -> None:
        """При --output результат пишется в JSON."""
        _patch(monkeypatch)
        out_file = tmp_path / "bench.json"
        code = await benchmark.main_async(
            _namespace(concurrency=[1], output=str(out_file))
        )
        assert code == 0
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert payload["окружение"]["llm_model"] == "model-x"
        assert len(payload["последовательно"]) == 2
        assert len(payload["одновременно"]) == 2
        assert payload["метрики"]["answers_per_minute"] == 40.0
