"""Тесты сеточного тюнинга: построение сетки и ранжирование по цели.

Прогон матрицы гоняет живые модели и покрывается смоук-запуском, здесь же —
чистые функции: декартова сетка из спеки (уникальные читаемые имена, отказ на
мусорной спеке) и сортировка результатов по целевой метрике.
"""

from __future__ import annotations

import pytest

from eval.result import RunResult
from eval.tune import build_grid, rank_results


def _result(name: str, value: float, std: float = 0.0) -> RunResult:
    """Результат прогона с одной метрикой и разбросом бутстрапа."""
    meta = {"bootstrap": {"std": {}}}
    if std:
        meta["bootstrap"]["std"] = {"hit@1": std}
    return RunResult(
        variant=name,
        environment="env",
        metrics={"hit@1": value},
        outcomes=[],
        meta=meta,
    )


class TestBuildGrid:
    """Декартова сетка из спеки параметров."""

    def test_product_size_and_names(self) -> None:
        """На каждое пересечение значений — вариант с читаемым именем."""
        grid = build_grid(
            {"retrieval_top_k": [3, 5], "rerank_candidates": [5, 10, 15]}
        )
        assert len(grid) == 6
        names = {variant.name for variant in grid}
        assert "retrieval_top_k=3;rerank_candidates=5" in names
        assert names == {
            f"retrieval_top_k={k};rerank_candidates={c}"
            for k in (3, 5)
            for c in (5, 10, 15)
        }
        assert len(names) == 6

    def test_settings_map_to_grid(self) -> None:
        """Настройки варианта совпадают с пересечением."""
        grid = build_grid({"retrieval_top_k": [8], "rerank_enabled": [True, False]})
        assert {v.settings["retrieval_top_k"] for v in grid} == {8}
        assert {v.settings["rerank_enabled"] for v in grid} == {True, False}

    def test_single_param(self) -> None:
        """Один параметр — это просто список его значений."""
        grid = build_grid({"retrieval_top_k": [3, 5, 8]})
        assert [v.settings["retrieval_top_k"] for v in grid] == [3, 5, 8]

    def test_empty_spec_raises(self) -> None:
        """Пустая спека — ошибка."""
        with pytest.raises(ValueError, match="пуста"):
            build_grid({})

    def test_empty_values_raise(self) -> None:
        """Параметр с пустым списком значений — ошибка."""
        with pytest.raises(ValueError, match="непустой список"):
            build_grid({"retrieval_top_k": []})

    def test_non_list_raises(self) -> None:
        """Значение не список — ошибка."""
        with pytest.raises(ValueError, match="непустой список"):
            build_grid({"retrieval_top_k": 5})


class TestRankResults:
    """Сортировка результатов по целевой метрике."""

    def _variants(self, count: int):
        from eval.config import Variant

        return [Variant(name=f"v{i}") for i in range(count)]

    def test_sorts_descending_for_max(self) -> None:
        """``direction=max`` — от лучшего к худшему."""
        variants = self._variants(3)
        results = [_result("v0", 0.3), _result("v1", 0.9), _result("v2", 0.6)]
        ranked = rank_results(variants, results, "hit@1", "max")
        assert [variant.name for variant, _, _, _ in ranked] == ["v1", "v2", "v0"]

    def test_sorts_ascending_for_min(self) -> None:
        """``direction=min`` — от минимального значения."""
        variants = self._variants(3)
        results = [_result("v0", 0.3), _result("v1", 0.9), _result("v2", 0.6)]
        ranked = rank_results(variants, results, "hit@1", "min")
        assert [variant.name for variant, _, _, _ in ranked] == ["v0", "v2", "v1"]

    def test_std_comes_from_bootstrap(self) -> None:
        """Разброс цели берётся из метаданных бутстрапа."""
        variants = self._variants(1)
        results = [_result("v0", 0.5, std=0.05)]
        ranked = rank_results(variants, results, "hit@1", "max")
        _, _, value, std = ranked[0]
        assert value == pytest.approx(0.5)
        assert std == pytest.approx(0.05)

    def test_missing_objective_raises(self) -> None:
        """Отсутствие цели в результатах — ошибка."""
        variants = self._variants(1)
        result = RunResult(
            variant="v0", environment="env", metrics={"mrr": 0.1}, outcomes=[]
        )
        with pytest.raises(ValueError, match="не рассчитана"):
            rank_results(variants, [result], "hit@1", "max")