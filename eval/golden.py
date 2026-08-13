"""Работа с золотым набором вопросов: загрузка и сопоставление цитат.

Цитата из документа — единственный признак, по которому замер решает, попали
ли нужные сведения в контекст. Сопоставление должно переживать различия в
регистре, «ё», пунктуации и переносах строк между цитатой и
проиндексированным чанком, но не давать ложных попаданий по случайному
совпадению общих слов.
"""

from __future__ import annotations

import re

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
