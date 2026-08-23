"""Тесты загрузки и валидации золотого набора.

Набор задаёт эталон для всех метрик, поэтому сломанная структура набора хуже,
чем плохая метрика: цифры остаются красивыми, а смысл теряется. Загрузка
должна отказываться на мусоре сразу, а не молча считать по нему.
"""

from __future__ import annotations

import json

import pytest

from eval.golden import DEFAULT_GOLDEN, GoldenSet, GoldenSetError

FIXTURE = DEFAULT_GOLDEN


def _write(tmp_path, data) -> str:
    """Пишет данные в JSON-файл во временном каталоге."""
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _valid() -> dict:
    """Валидный минимальный набор."""
    return {
        "answerable": [
            {
                "question": "Какой срок?",
                "quote": "Срок не превышает 15 дней",
                "kind": "числовой",
            }
        ],
        "unanswerable": [
            {
                "question": "Какая ставка НДС?",
                "kind": "вне области",
            }
        ],
    }


class TestLoad:
    """Загрузка корректных наборов."""

    def test_loads_real_fixture(self) -> None:
        """Боевой набор загружается и не мельчает.

        Проверяется нижняя граница, а не точное число: набор пополняется, и
        жёсткое равенство пришлось бы править при каждом добавлении вопроса.
        А вот уменьшение набора - всегда повод остановиться: значит вопросы
        пропали, а не добавились.
        """
        golden = GoldenSet.load(FIXTURE)
        assert len(golden.answerable) >= 121
        assert len(golden.unanswerable) >= 10
        assert golden.total == len(golden.answerable) + len(golden.unanswerable)

    def test_load_minimal(self, tmp_path) -> None:
        """Минимальный валидный набор загружается."""
        golden = GoldenSet.load(_write(tmp_path, _valid()))
        assert golden.total == 2

    def test_ignores_metadata_keys(self, tmp_path) -> None:
        """Служебные ключи (_описание и т.п.) не мешают загрузке."""
        data = _valid()
        data["_описание"] = "описание"
        golden = GoldenSet.load(_write(tmp_path, data))
        assert golden.total == 2

    def test_limited(self, tmp_path) -> None:
        """Ограничение режет оба вида вопросов."""
        base = _valid()
        answerable = [
            {**base["answerable"][0], "question": f"вопрос {i}"} for i in range(3)
        ]
        unanswerable = [
            {**base["unanswerable"][0], "question": f"вне области {i}"}
            for i in range(3)
        ]
        payload = {"answerable": answerable, "unanswerable": unanswerable}
        golden = GoldenSet.load(_write(tmp_path, payload))
        assert golden.limited(2).total == 4
        assert golden.limited(None).total == 6


class TestResample:
    """Бутстрап-выборка вопросов."""

    def test_same_size_and_source(self, tmp_path) -> None:
        """Выборка сохраняет размер и берёт вопросы из пула."""
        base = _valid()
        answerable = [
            {**base["answerable"][0], "question": f"вопрос {i}"} for i in range(5)
        ]
        unanswerable = [
            {**base["unanswerable"][0], "question": f"вне области {i}"}
            for i in range(3)
        ]
        payload = {"answerable": answerable, "unanswerable": unanswerable}
        golden = GoldenSet.load(_write(tmp_path, payload))
        sample = golden.resample(seed=42, sample_index=0)

        assert len(sample.answerable) == 5
        assert len(sample.unanswerable) == 3
        source = {item["question"] for item in golden.answerable}
        assert all(item["question"] in source for item in sample.answerable)

    def test_deterministic(self, tmp_path) -> None:
        """Та же пара (seed, index) даёт ту же выборку."""
        golden = GoldenSet.load(_write(tmp_path, _valid()))
        rng_answer = [
            golden.resample(seed=7, sample_index=3).answerable
            for _ in range(2)
        ]
        assert rng_answer[0] == rng_answer[1]

    def test_samples_differ_across_indexes(self, tmp_path) -> None:
        """Разные индексы дают разные выборки (на достаточно большом пуле)."""
        base = _valid()
        answerable = [
            {**base["answerable"][0], "question": f"вопрос {i}"} for i in range(20)
        ]
        payload = {"answerable": answerable, "unanswerable": []}
        golden = GoldenSet.load(_write(tmp_path, payload))
        a = golden.resample(seed=42, sample_index=0).answerable
        b = golden.resample(seed=42, sample_index=1).answerable
        assert a != b

    def test_empty_category(self, tmp_path) -> None:
        """Пустая категория не приводит к ошибке ресемплинга."""
        golden = GoldenSet.load(_write(tmp_path, _valid()))
        sample = golden.resample(seed=1, sample_index=0)
        assert len(sample.unanswerable) == 1


class TestErrors:
    """Отказ на нарушении целостности."""

    def test_missing_file(self, tmp_path) -> None:
        """Нет файла — ошибка."""
        with pytest.raises(GoldenSetError, match="не найден"):
            GoldenSet.load(tmp_path / "absent.json")

    def test_invalid_json(self, tmp_path) -> None:
        """Битый JSON — ошибка."""
        path = tmp_path / "golden.json"
        path.write_text("{не json", encoding="utf-8")
        with pytest.raises(GoldenSetError, match="не является JSON"):
            GoldenSet.load(str(path))

    def test_answerable_without_quote(self, tmp_path) -> None:
        """У отвечаемого вопроса нет цитаты — ошибка."""
        data = _valid()
        del data["answerable"][0]["quote"]
        with pytest.raises(GoldenSetError, match="нет цитаты"):
            GoldenSet.load(_write(tmp_path, data))

    def test_empty_question(self, tmp_path) -> None:
        """Пустой вопрос — ошибка."""
        data = _valid()
        data["unanswerable"][0]["question"] = "   "
        with pytest.raises(GoldenSetError, match="вопрос не задан"):
            GoldenSet.load(_write(tmp_path, data))

    def test_unanswerable_without_kind(self, tmp_path) -> None:
        """Вопрос без ответа без типа — ошибка."""
        data = _valid()
        del data["unanswerable"][0]["kind"]
        with pytest.raises(GoldenSetError, match="тип"):
            GoldenSet.load(_write(tmp_path, data))

    def test_unknown_answerable_kind(self, tmp_path) -> None:
        """Неизвестный тип отвечаемого вопроса — ошибка."""
        data = _valid()
        data["answerable"][0]["kind"] = "странный"
        with pytest.raises(GoldenSetError, match="неизвестный тип"):
            GoldenSet.load(_write(tmp_path, data))

    def test_duplicate_question(self, tmp_path) -> None:
        """Повтор вопроса в наборе — ошибка."""
        data = _valid()
        data["answerable"].append(dict(data["answerable"][0]))
        with pytest.raises(GoldenSetError, match="повторяется"):
            GoldenSet.load(_write(tmp_path, data))

    def test_non_dict_item(self, tmp_path) -> None:
        """Элемент набора не объект — ошибка."""
        data = _valid()
        data["answerable"].append("вопрос")
        with pytest.raises(GoldenSetError, match="не объект"):
            GoldenSet.load(_write(tmp_path, data))
