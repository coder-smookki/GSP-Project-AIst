"""Работа с золотым набором вопросов: загрузка, валидация, сопоставление цитат.

Золотой набор — эталон, по которому меряется качество поиска и ответов.
:class:`GoldenSet` отвечает за загрузку и структурную целостность (обязательные
поля, известные типы вопросов, отсутствие повторов), а сопоставление цитат
даёт признак «попал ли ответ в контекст». Целостность цитат против корпуса
проверяется отдельно (см. tests/test_golden_set.py), так как требует корпуса.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Путь к золотому набору по умолчанию.
DEFAULT_GOLDEN = Path("tests/fixtures/golden/questions.json")

#: Допустимые типы отвечаемых вопросов.
ANSWERABLE_KINDS = {"числовой", "фактический", "процедурный"}

#: Допустимые типы вопросов без ответа.
UNANSWERABLE_KINDS = {"вне области", "в области но нет в корпусе"}


class GoldenSetError(ValueError):
    """Нарушение целостности золотого набора."""


@dataclass(frozen=True)
class GoldenSet:
    """Загруженный золотой набор вопросов.

    Attributes:
        answerable: Вопросы с известным ответом и дословной цитатой-эталоном.
        unanswerable: Вопросы, на которые верный ответ — «В базе знаний
            информации нет».
    """

    answerable: tuple[dict[str, Any], ...]
    unanswerable: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_GOLDEN) -> GoldenSet:
        """Загружает и проверяет набор из JSON-файла.

        Args:
            path: Путь к набору.

        Returns:
            GoldenSet: Загруженный набор.

        Raises:
            GoldenSetError: Файл не найден, не является JSON или нарушает
                структуру набора.
        """
        source = Path(path)
        if not source.exists():
            raise GoldenSetError(f"золотой набор не найден: {source}")
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GoldenSetError(
                f"золотой набор не является JSON: {source}: {exc}"
            ) from exc

        answerable = cls._validate(
            data.get("answerable", []), answerable=True, source=source
        )
        unanswerable = cls._validate(
            data.get("unanswerable", []), answerable=False, source=source
        )
        cls._check_unique(answerable, unanswerable)
        return cls(answerable=tuple(answerable), unanswerable=tuple(unanswerable))

    @staticmethod
    def _validate(
        items: list[dict[str, Any]], *, answerable: bool, source: Path
    ) -> list[dict[str, Any]]:
        """Проверяет обязательные поля и типы вопросов.

        Args:
            items: Элементы набора.
            answerable: Проверять ли как отвечаемые (с цитатой).
            source: Путь к файлу для сообщения об ошибке.

        Returns:
            list[dict[str, Any]]: Проверенные элементы.

        Raises:
            GoldenSetError: Элемент без обязательного поля или с неизвестным
                типом вопроса.
        """
        for item in items:
            if not isinstance(item, dict):
                raise GoldenSetError(f"{source}: элемент набора — не объект: {item!r}")
            question = item.get("question")
            if not isinstance(question, str) or not question.strip():
                raise GoldenSetError(f"{source}: вопрос не задан: {item!r}")
            if answerable:
                quote = item.get("quote")
                if not isinstance(quote, str) or not quote.strip():
                    raise GoldenSetError(
                        f"{source}: у отвечаемого вопроса нет цитаты: {item!r}"
                    )
                kind = item.get("kind")
                if kind is not None and kind not in ANSWERABLE_KINDS:
                    raise GoldenSetError(
                        f"{source}: неизвестный тип вопроса {kind!r}: {item!r}"
                    )
            else:
                kind = item.get("kind")
                if kind not in UNANSWERABLE_KINDS:
                    raise GoldenSetError(
                        f"{source}: у вопроса без ответа не задан тип: {item!r}"
                    )
        return items

    @staticmethod
    def _check_unique(
        answerable: list[dict[str, Any]], unanswerable: list[dict[str, Any]]
    ) -> None:
        """Проверяет отсутствие повторяющихся вопросов.

        Повтор завышает вес одного и того же случая и искажает метрики.

        Args:
            answerable: Отвечаемые вопросы.
            unanswerable: Вопросы без ответа.

        Raises:
            GoldenSetError: Найден повторяющийся вопрос.
        """
        seen: dict[str, str] = {}
        for item in [*answerable, *unanswerable]:
            key = item["question"].strip().lower()
            if key in seen:
                raise GoldenSetError(
                    f"вопрос повторяется в наборе: {item['question']!r}"
                )
            seen[key] = item["question"]

    def limited(self, limit: int | None) -> GoldenSet:
        """Ограничивает набор первыми ``limit`` вопросами каждого вида.

        Args:
            limit: Сколько вопросов оставить; ``None`` — весь набор.

        Returns:
            GoldenSet: Урезанный набор.
        """
        if limit is None:
            return self
        return GoldenSet(self.answerable[:limit], self.unanswerable[:limit])

    @property
    def total(self) -> int:
        """Общее число вопросов в наборе.

        Returns:
            int: Отвечаемых и без ответа вместе.
        """
        return len(self.answerable) + len(self.unanswerable)


#: Сколько подряд идущих слов цитаты должно найтись во фрагменте, чтобы
#: счесть его попаданием.
#:
#: Точное вхождение целиком не годится: чанкинг режет текст по атомам, и
#: цитата из двух предложений запросто окажется на границе двух фрагментов.
#: Отдельные слова, наоборот, совпадают случайно — «работник», «организация»
#: и «безопасность» есть почти в каждом пункте. Шесть слов подряд —
#: компромисс: случайно такая цепочка не возникает, а разрыв по границе
#: чанка её обычно не разрушает.
MIN_MATCH_WORDS = 6


def normalize(text: str) -> list[str]:
    """Приводит текст к списку слов для сопоставления.

    Args:
        text: Исходный текст.

    Returns:
        list[str]: Слова в нижнем регистре без пунктуации.
    """
    lowered = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", lowered)


def longest_common_run(needle: list[str], haystack: list[str]) -> int:
    """Считает длину самой длинной общей цепочки слов подряд.

    Args:
        needle: Слова эталонной цитаты.
        haystack: Слова проверяемого фрагмента.

    Returns:
        int: Число слов в самой длинной совпавшей цепочке.
    """
    if not needle or not haystack:
        return 0
    positions: dict[str, list[int]] = {}
    for index, word in enumerate(haystack):
        positions.setdefault(word, []).append(index)

    best = 0
    # Классическая динамика по общей подстроке, но только по позициям, где
    # слово вообще встречается: словарь регламентов узкий, и полная матрица
    # на длинных чанках считалась бы заметно дольше без пользы.
    previous: dict[int, int] = {}
    for word in needle:
        current: dict[int, int] = {}
        for index in positions.get(word, ()):
            run = previous.get(index - 1, 0) + 1
            current[index] = run
            best = max(best, run)
        previous = current
    return best


def chunk_contains(quote: str, chunk_text: str) -> bool:
    """Проверяет, содержит ли фрагмент эталонную цитату.

    Args:
        quote: Дословная цитата из документа.
        chunk_text: Текст найденного фрагмента.

    Returns:
        bool: ``True``, если совпало достаточно слов подряд. Для коротких
        цитат требуется совпадение целиком: у них цепочка в шесть слов
        просто не наберётся.
    """
    words = normalize(quote)
    if not words:
        return False
    need = min(MIN_MATCH_WORDS, len(words))
    return longest_common_run(words, normalize(chunk_text)) >= need
