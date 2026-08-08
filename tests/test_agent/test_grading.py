"""Тесты грейдера релевантности.

Проверяется прежде всего поведение при сбоях и при рассогласовании длин:
грейдер отсеивает источники, поэтому его ошибка приводит либо к потере
верного ответа, либо к сдвигу соответствия «вердикт — фрагмент», а это
хуже, чем отсутствие грейдинга.
"""

from __future__ import annotations

import pytest

from agent.grading import GradeResult, grade_chunks


class _Structured:
    """Модель со структурированным выводом, отдающая заданный результат."""

    def __init__(self, result):
        self._result = result
        self.calls = 0
        self.schema = None

    def with_structured_output(self, schema, **kwargs):
        """Запоминает схему и возвращает себя как готовый к вызову объект.

        Args:
            schema: Класс схемы структурированного вывода.
            **kwargs: Прочие параметры.

        Returns:
            _Structured: Сам объект.
        """
        self.schema = schema
        return self

    async def ainvoke(self, messages, **kwargs):
        """Возвращает заготовленный результат либо бросает ошибку.

        Args:
            messages: Сообщения запроса.
            **kwargs: Прочие параметры вызова.

        Returns:
            GradeResult: Заготовленный результат.

        Raises:
            Exception: Если вместо результата задана ошибка.
        """
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Model:
    """Клиент модели с доступом к `raw`."""

    def __init__(self, result):
        self.raw = _Structured(result)


def _items(count: int) -> list[dict]:
    """Готовит цитаты для грейдинга.

    Args:
        count: Сколько цитат создать.

    Returns:
        list[dict]: Цитаты с текстом и номером.
    """
    return [{"n": n, "text": f"фрагмент {n}"} for n in range(1, count + 1)]


async def test_keeps_only_relevant() -> None:
    """Остаются только фрагменты с вердиктом «релевантен»."""
    model = _Model(GradeResult(v=["Р", "Н", "Ч"]))
    kept = await grade_chunks(model, "вопрос", _items(3), keep_partial=False)
    assert [item["text"] for item in kept] == ["фрагмент 1"]


async def test_keep_partial_widens_selection() -> None:
    """С `keep_partial` остаются и частично относящиеся фрагменты."""
    model = _Model(GradeResult(v=["Р", "Н", "Ч"]))
    kept = await grade_chunks(model, "вопрос", _items(3), keep_partial=True)
    assert [item["text"] for item in kept] == ["фрагмент 1", "фрагмент 3"]


async def test_all_irrelevant_gives_empty_selection() -> None:
    """Полный отсев даёт пустую выдачу — основание для честного отказа."""
    model = _Model(GradeResult(v=["Н", "Н"]))
    assert await grade_chunks(model, "вопрос", _items(2)) == []


async def test_length_mismatch_keeps_everything() -> None:
    """Расхождение длин отменяет грейдинг, а не сдвигает соответствие.

    Ограниченное декодирование умеет дописывать недостающие элементы, и
    тогда вердикт с номером ``i`` относится уже не к фрагменту ``i``.
    Отсеивать по такому ответу — значит выбрасывать случайные фрагменты.
    """
    model = _Model(GradeResult(v=["Р", "Н"]))
    assert await grade_chunks(model, "вопрос", _items(3)) is None


async def test_model_failure_keeps_everything() -> None:
    """Сбой модели оставляет выдачу нетронутой, а не роняет запрос."""
    model = _Model(RuntimeError("модель недоступна"))
    assert await grade_chunks(model, "вопрос", _items(3)) is None


async def test_empty_input_does_not_call_model() -> None:
    """Пустая выдача не идёт в модель."""
    model = _Model(GradeResult(v=[]))
    assert await grade_chunks(model, "вопрос", []) == []
    assert model.raw.calls == 0


@pytest.mark.parametrize("bad", ["помогает", "helpful", "Y", ""])
def test_schema_rejects_free_form_verdicts(bad: str) -> None:
    """Схема не принимает вердикты вне трёх заданных градаций.

    Без перечисления модель наполняет поле отсебятиной: в замерах приходили
    значения «помогает» и «helpful». Проверка фиксирует, что тип поля —
    именно перечисление, а не строка с описанием.
    """
    with pytest.raises(ValueError):
        GradeResult(v=[bad])


def test_schema_exposes_enum_to_the_model() -> None:
    """В JSON-схеме вердикт представлен как ``enum``.

    Именно это включает ограниченное декодирование; описание поля таким
    ограничением не является.
    """
    schema = GradeResult.model_json_schema()
    enum = schema["$defs"]["Verdict"]["enum"] if "$defs" in schema else None
    assert enum is None or set(enum) == {"Р", "Ч", "Н"}
    assert "Р" in str(schema)
