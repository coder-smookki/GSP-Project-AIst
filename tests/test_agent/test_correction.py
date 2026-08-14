"""Тесты коррекции запроса.

Коррекция вызывается только при пустой выдаче, поэтому её поведение при
сбоях важнее, чем при успехе: неудача коррекции не должна ни ронять запрос,
ни отменять честный отказ.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agent.correction import MAX_REWRITE_CHARS, rewrite_query


class _Model:
    """Модель, отдающая заранее заданный ответ."""

    def __init__(self, answer: str | Exception):
        self._answer = answer
        self.calls = 0

    async def ainvoke(self, messages: list, **kwargs):
        """Возвращает заготовленный ответ или бросает заготовленную ошибку.

        Args:
            messages: Сообщения запроса.
            **kwargs: Прочие параметры вызова.

        Returns:
            AIMessage: Заготовленный ответ.

        Raises:
            Exception: Если вместо ответа задана ошибка.
        """
        self.calls += 1
        if isinstance(self._answer, Exception):
            raise self._answer
        return AIMessage(content=self._answer)


async def test_rewrites_colloquial_question() -> None:
    """Переписанный запрос возвращается очищенным от кавычек."""
    model = _Model('"Места эвакуации при возгорании"')
    assert await rewrite_query(model, "куда бежать если загорелось") == (
        "Места эвакуации при возгорании"
    )


async def test_takes_only_first_line() -> None:
    """Из многострочного ответа берётся только запрос.

    Модель склонна дописывать пояснения; отправлять их в поиск нельзя.
    """
    model = _Model("Порядок эвакуации при пожаре\n\nЯ переформулировал вопрос.")
    assert await rewrite_query(model, "куда бежать") == "Порядок эвакуации при пожаре"


async def test_rejects_overlong_result() -> None:
    """Слишком длинный ответ отвергается: это рассуждение, а не запрос."""
    model = _Model("а" * (MAX_REWRITE_CHARS + 1))
    assert await rewrite_query(model, "куда бежать") is None


async def test_rejects_unchanged_result() -> None:
    """Если запрос не изменился, повторять поиск незачем."""
    model = _Model("Куда Бежать")
    assert await rewrite_query(model, "куда бежать") is None


async def test_model_failure_does_not_propagate() -> None:
    """Сбой модели не роняет запрос, а лишь отменяет коррекцию.

    Иначе временная недоступность модели превращала бы честный отказ
    в ошибку интерфейса.
    """
    model = _Model(RuntimeError("модель недоступна"))
    assert await rewrite_query(model, "куда бежать") is None


@pytest.mark.parametrize("question", ["", "   "])
async def test_empty_question_does_not_call_model(question: str) -> None:
    """Пустой вопрос не идёт в модель."""
    model = _Model("что угодно")
    assert await rewrite_query(model, question) is None
    assert model.calls == 0
