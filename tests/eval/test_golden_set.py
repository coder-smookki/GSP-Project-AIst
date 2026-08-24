"""Проверка целостности золотого набора.

Набор задаёт эталон, по которому меряется качество поиска, поэтому молча
разойтись с корпусом он не должен. Если документ переиндексировали,
переименовали или заменили, а цитата осталась старой, замер начнёт
показывать падение качества там, где на самом деле сломался эталон.
Эти тесты ловят такое расхождение сразу.

Тесты работают с реальным корпусом, поэтому помечены маркером ``corpus``
и не мешают быстрым прогонам.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from html import unescape
from pathlib import Path

import pytest

pytestmark = pytest.mark.corpus

# parents[2], а не [1]: файл переехал в tests/eval/, и путь стал
# складываться в tests/tests/fixtures - набор «не находился», а все
# проверки молча пропускались. То есть цитаты не сверялись ни разу.
ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "golden" / "questions.json"
CORPUS_DIR = ROOT / "tests" / "fixtures" / "corpus"


def _words(text: str) -> list[str]:
    """Приводит текст к списку слов.

    Args:
        text: Исходный текст.

    Returns:
        list[str]: Слова в нижнем регистре без пунктуации.
    """
    return re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))


def _longest_run(needle: list[str], haystack: list[str]) -> int:
    """Длина самой длинной общей цепочки слов подряд.

    Args:
        needle: Слова цитаты.
        haystack: Слова документа.

    Returns:
        int: Число совпавших слов подряд.
    """
    if not needle or not haystack:
        return 0
    positions: dict[str, list[int]] = {}
    for index, word in enumerate(haystack):
        positions.setdefault(word, []).append(index)
    best = 0
    previous: dict[int, int] = {}
    for word in needle:
        current: dict[int, int] = {}
        for index in positions.get(word, ()):
            run = previous.get(index - 1, 0) + 1
            current[index] = run
            best = max(best, run)
        previous = current
    return best


@pytest.fixture(scope="module")
def golden() -> dict:
    """Загружает золотой набор.

    Returns:
        dict: Содержимое набора.
    """
    if not GOLDEN_PATH.exists():
        pytest.skip("золотой набор не найден")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    """Загружает тексты корпуса без разметки.

    Returns:
        dict[str, str]: Текст документа по его идентификатору.
    """
    if not CORPUS_DIR.exists():
        pytest.skip("корпус не найден")
    texts: dict[str, str] = {}
    for path in CORPUS_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))["data"]
        plain = unescape(re.sub(r"<[^>]+>", " ", data["content"]))
        texts[str(data["id"])] = re.sub(r"\s+", " ", plain)
    return texts


def test_every_quote_is_present_in_its_document(golden: dict, corpus: dict) -> None:
    """Каждая цитата дословно есть в указанном документе.

    Это главная гарантия набора: цитата — единственный признак, по которому
    замер решает, попал ли ответ в контекст. Пересказ вместо цитаты сделал
    бы все цифры недостоверными.
    """
    broken: list[str] = []
    for item in golden["answerable"]:
        text = corpus.get(str(item["doc_id"]))
        if text is None:
            broken.append(f"нет документа {item['doc_id']}: {item['question'][:50]}")
            continue
        quote_words = _words(item["quote"])
        need = min(len(quote_words), max(8, int(len(quote_words) * 0.8)))
        matched = _longest_run(quote_words, _words(text))
        if matched < need:
            broken.append(
                f"цитата не найдена ({matched} из {len(quote_words)} слов): "
                f"{item['question'][:50]}"
            )
    assert not broken, "\n".join(broken)


def test_no_anonymisation_placeholders_leak(golden: dict) -> None:
    """В вопросах нет следов обезличивания.

    Плейсхолдер вида ``[ОРГАНИЗАЦИЯ_7]`` в вопросе означает, что вопрос
    списан с текста, а не задан человеком: настоящий пользователь таких
    слов не пишет, и проверка на нём ничего не покажет.
    """
    pattern = re.compile(r"\[(ОРГАНИЗАЦИЯ|АДРЕС|ПЕРСОНА|ЧЕЛОВЕК)_\d+\]")
    leaked = [
        item["question"]
        for item in golden["answerable"] + golden["unanswerable"]
        if pattern.search(item["question"])
    ]
    assert not leaked, leaked


def test_questions_are_unique(golden: dict) -> None:
    """Повторов нет: они завышают вес одного и того же случая."""
    questions = [
        item["question"].strip().lower()
        for item in golden["answerable"] + golden["unanswerable"]
    ]
    duplicates = [q for q, count in Counter(questions).items() if count > 1]
    assert not duplicates, duplicates


def test_set_covers_the_corpus_broadly(golden: dict, corpus: dict) -> None:
    """Набор охватывает большую часть документов.

    Вопросы, стянутые к двум-трём документам, меряют не поиск по базе, а
    поиск по этим документам.
    """
    covered = {str(item["doc_id"]) for item in golden["answerable"]}
    assert len(covered) >= len(corpus) * 0.7, (
        f"охвачено {len(covered)} документов из {len(corpus)}"
    )


def test_has_both_kinds_of_unanswerable(golden: dict) -> None:
    """Есть оба вида вопросов без ответа.

    Явно посторонние вопросы отсекаются порогом близости и потому лёгкие.
    Ценность даёт второй вид — звучащие уместно, с лексикой корпуса, но без
    ответа в нём: поиск на них обязательно что-то вернёт, и именно там
    система склонна выдумывать.
    """
    kinds = {item["kind"] for item in golden["unanswerable"]}
    assert "вне области" in kinds
    assert "в области но нет в корпусе" in kinds


def test_numeric_questions_are_represented(golden: dict) -> None:
    """Числовых вопросов достаточно.

    Сроки, периодичность и количества проверяются однозначно, поэтому
    именно на них видно подмену ответа выдумкой.
    """
    numeric = sum(1 for item in golden["answerable"] if item["kind"] == "числовой")
    assert numeric >= len(golden["answerable"]) * 0.25, (
        f"числовых вопросов {numeric} из {len(golden['answerable'])}"
    )
