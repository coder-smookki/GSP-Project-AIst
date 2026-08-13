"""Результаты прогона: итог по вопросу, по варианту и вердикт A/B.

Формат результата — общий для всех раннеров: кэш, отчёты и статистика
работают с этими типами, не зная, откуда пришли данные (поиск или агент).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Outcome:
    """Итог по одному вопросу.

    Attributes:
        question: Текст вопроса.
        answerable: Отвечаемый ли вопрос (есть эталонная цитата). Вопросы без
            ответа в базе помечаются ``False`` — метрики считаются по разным
            подмножествам.
        rank: Позиция первого фрагмента с ответом, начиная с 1;
            ``None`` — ответа нет ни в одном фрагменте.
        returned: Сколько фрагментов вернул поиск.
        seconds: Время поиска.
        answer: Ответ модели, если он запрашивался.
        refused: Отказалась ли система отвечать.
        doc_codes: Коды документов найденных фрагментов.
    """

    question: str
    answerable: bool = True
    rank: int | None = None
    returned: int = 0
    seconds: float = 0.0
    answer: str | None = None
    refused: bool = False
    doc_codes: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Полный результат прогона одного варианта.

    Attributes:
        variant: Имя варианта.
        environment: Отпечаток окружения (``Environment.fingerprint``).
        metrics: Сводные метрики: ключ — метрика, значение — число.
        outcomes: Итоги по каждому вопросу (источник для метрик и A/B).
        created_at: ISO-метка времени прогона.
        meta: Произвольные дополнительные данные (полный конфиг и т.п.).
    """

    variant: str
    environment: str
    metrics: dict[str, float]
    outcomes: list[Outcome]
    created_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализует результат в словарь для JSON-кэша.

        Returns:
            dict[str, Any]: Плоское представление результата.
        """
        return {
            "variant": self.variant,
            "environment": self.environment,
            "metrics": self.metrics,
            "outcomes": [asdict(o) for o in self.outcomes],
            "created_at": self.created_at,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        """Восстанавливает результат из словаря.

        Args:
            data: Словарь, ранее полученный из :meth:`to_dict`.

        Returns:
            RunResult: Восстановленный результат.
        """
        return cls(
            variant=data["variant"],
            environment=data["environment"],
            metrics=data["metrics"],
            outcomes=[Outcome(**o) for o in data["outcomes"]],
            created_at=data.get("created_at", ""),
            meta=data.get("meta", {}),
        )

    def to_json(self) -> str:
        """Сериализует результат в JSON.

        Returns:
            str: JSON с сохранением кириллицы.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class Verdict:
    """Вердикт статистического теста по одной метрике.

    Attributes:
        metric: Ключ метрики.
        test: Название применённого теста.
        p_value: Достигнутый уровень значимости.
        base_value: Значение метрики у base-варианта.
        variant_value: Значение метрики у нового варианта.
        direction: ``better`` / ``worse`` / ``no_diff`` относительно base.
    """

    metric: str
    test: str
    p_value: float
    base_value: float
    variant_value: float
    direction: str
