"""Тесты CLI A/B-сравнения: загрузка и валидация вариантов.

Сама матрица прогонов гоняет живые модели, поэтому тестируется только
парсинг файла вариантов и сериализация отчёта — без зависимостей от
инфраструктуры.
"""

from __future__ import annotations

import json

import pytest

from eval.compare import _build_payload, load_variants
from eval.metrics import HitAtK
from eval.result import Outcome, RunResult


class TestLoadVariants:
    """Чтение и валидация файла вариантов."""

    def test_loads_list(self, tmp_path) -> None:
        """Корректный файл читается в варианты."""
        path = tmp_path / "variants.json"
        path.write_text(
            json.dumps(
                {
                    "variants": [
                        {"name": "base"},
                        {
                            "name": "top6",
                            "description": "top_k=6",
                            "settings": {"retrieval_top_k": 6},
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        variants = load_variants(path)
        assert [v.name for v in variants] == ["base", "top6"]
        assert variants[1].settings == {"retrieval_top_k": 6}
        assert variants[1].description == "top_k=6"

    def test_loads_bare_list(self, tmp_path) -> None:
        """Допускается файл-список без обёртки ``variants``."""
        path = tmp_path / "variants.json"
        path.write_text(
            json.dumps([{"name": "base"}, {"name": "x"}]),
            encoding="utf-8",
        )
        assert [v.name for v in load_variants(path)] == ["base", "x"]

    def test_empty_raises(self, tmp_path) -> None:
        """Пустой список — ошибка."""
        path = tmp_path / "variants.json"
        path.write_text(json.dumps({"variants": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="нет вариантов"):
            load_variants(path)

    def test_duplicate_names_raise(self, tmp_path) -> None:
        """Дубли имён — ошибка: варианты неотличимы в отчёте."""
        path = tmp_path / "variants.json"
        path.write_text(
            json.dumps({"variants": [{"name": "a"}, {"name": "a"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="уникальны"):
            load_variants(path)

    def test_missing_file_raises(self, tmp_path) -> None:
        """Отсутствующий файл — ошибка чтения."""
        with pytest.raises(OSError):
            load_variants(tmp_path / "nope.json")


class TestPayload:
    """Полный отчёт для вывода в JSON."""

    def test_build_payload_roundtrip(self) -> None:
        """Отчёт содержит варианты, результаты и вердикты."""
        result = RunResult(
            variant="base",
            environment="env",
            metrics={"hit@1": 1.0},
            outcomes=[Outcome(question="q", rank=1)],
        )
        payload = _build_payload(
            [],
            [result],
            [HitAtK(1)],
            [{"variant": "x", "metric": "hit@1"}],
        )
        assert payload["metrics"] == ["hit@1"]
        assert payload["results"][0]["variant"] == "base"
        assert payload["verdicts"] == [{"variant": "x", "metric": "hit@1"}]
