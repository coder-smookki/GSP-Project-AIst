"""Показ этапов работы графа шагами Chainlit.

Узлы графа шлют события ``trace`` (см. ``agent/events.py``), а здесь они
превращаются в шаги ``cl.Step``: пока узел считает, пользователь видит его
название, после завершения - короткий итог. Раньше на планировании и поиске
интерфейс молчал, и ожидание выглядело как зависание.

Шаги ведутся вручную (``send`` и ``update``), а не через ``async with``:
события приходят вперемешку из параллельных веток поиска, и один блок
``with`` на этап тут не натянуть. Поэтому отметки ``start``/``end``
проставляются явно - по ним интерфейс и отличает идущий этап от готового.

Любой сбой отрисовки гасится и только пишется в журнал: ответ пользователю
важнее оформления этапов.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from gui.contract import StageEvent, parse_stage

logger = logging.getLogger(__name__)

#: Тип шага Chainlit для этапов графа. Тип «tool» виден при любом режиме
#: отображения хода рассуждений, кроме полностью выключенного (cot).
STAGE_STEP_TYPE = "tool"

#: Итог этапа, оборванного до завершения (разрыв потока, ошибка запуска).
INTERRUPTED_DETAIL = "Этап не завершён"


def _now() -> str:
    """Отметка времени в формате, который ждёт интерфейс Chainlit.

    Returns:
        str: Время UTC в ISO-формате с суффиксом ``Z``.
    """
    return datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z"


def _chainlit_step(title: str) -> Any:
    """Создаёт шаг Chainlit для этапа графа.

    Импорт внутри функции: модуль должен разбираться и тестироваться без
    поднятого окружения Chainlit.

    Args:
        title: Заголовок шага.

    Returns:
        Any: Экземпляр ``cl.Step``.
    """
    import chainlit as cl

    return cl.Step(name=title, type=STAGE_STEP_TYPE)


class StageView:
    """Рисует этапы графа шагами Chainlit по событиям ``trace``."""

    def __init__(self, step_factory: Callable[[str], Any] | None = None) -> None:
        """Создаёт представление этапов одного запроса.

        Args:
            step_factory: Фабрика шага по заголовку; по умолчанию ``cl.Step``.
                Подменяется в тестах.
        """
        self._make_step = step_factory or _chainlit_step
        self._steps: dict[str, Any] = {}

    async def handle(self, payload: dict) -> None:
        """Отрисовывает одно событие этапа.

        Исключений не выпускает: сбой оформления не должен ронять ответ.

        Args:
            payload: Полезная нагрузка события trace.

        Returns:
            None
        """
        try:
            event = parse_stage(payload)
            if event is None:
                return
            if event.running:
                await self._start(event)
            else:
                await self._finish(event)
        except Exception:
            logger.exception("не удалось показать этап: %s", payload)

    async def close(self) -> None:
        """Закрывает этапы, оставшиеся открытыми.

        Вызывается после потока: если узел оборвался и события о завершении
        не прислал, шаг иначе навсегда остался бы «выполняется».

        Returns:
            None
        """
        for key, step in list(self._steps.items()):
            self._steps.pop(key, None)
            try:
                if not getattr(step, "output", ""):
                    step.output = INTERRUPTED_DETAIL
                step.end = _now()
                await step.update()
            except Exception:
                logger.exception("не удалось закрыть этап %s", key)

    async def _start(self, event: StageEvent) -> None:
        """Открывает шаг начавшегося этапа.

        Args:
            event: Описание этапа.

        Returns:
            None
        """
        if event.key in self._steps:
            return
        step = self._make_step(event.title)
        step.start = _now()
        self._steps[event.key] = step
        await step.send()

    async def _finish(self, event: StageEvent) -> None:
        """Закрывает шаг завершившегося этапа.

        Если начала этапа не было (узел прислал только событие о завершении),
        шаг создаётся сразу закрытым: иначе этап потерялся бы вовсе.

        Args:
            event: Описание этапа.

        Returns:
            None
        """
        step = self._steps.pop(event.key, None)
        created = step is None
        if step is None:
            step = self._make_step(event.title)
            step.start = _now()
        step.output = event.detail
        step.is_error = event.failed
        step.end = _now()
        if created:
            await step.send()
        else:
            await step.update()
