"""Регрессионные тесты конвейера на реальном корпусе базы знаний.

`tests/fixtures/corpus/` — выгрузка всех 20 статей раздела 44030
(«Пожарная безопасность»), полученная через API организаторов 08.08.2026.
Документы обезличены на стороне заказчика; репозиторий приватный.

Смысл этих тестов отличается от юнит-тестов на синтетической фикстуре: там
проверяется поведение на известных дефектах, здесь — что конвейер не
деградирует на настоящем объёме и разнообразии разметки. Порог задан с
запасом вниз, чтобы тест ловил регрессию, а не колебания.

Маркер `corpus` позволяет исключить набор из быстрого прогона:
``pytest -m "not corpus"``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from infrastructure.ingest.chunker import MAX_CHARS, chunk_document
from infrastructure.ingest.deanon import deanonymize
from infrastructure.ingest.html_parse import parse_article
from infrastructure.ingest.models import SectionType
from infrastructure.ingest.patterns import PLACEHOLDER

CORPUS_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "corpus"

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    """Загружает статьи реального корпуса.

    Returns:
        list[dict]: Объекты статей (поле ``data`` ответа API).
    """
    files = sorted(CORPUS_DIR.glob("*.json"))
    if not files:
        pytest.skip(f"выгрузка корпуса не найдена: {CORPUS_DIR}")
    return [json.loads(path.read_text(encoding="utf-8"))["data"] for path in files]


@pytest.fixture(scope="module")
def indexed(corpus: list[dict]) -> list:
    """Прогоняет корпус через весь конвейер один раз на модуль.

    Args:
        corpus: Статьи корпуса.

    Returns:
        list: Чанки всех документов.
    """
    chunks = []
    for article in corpus:
        chunks.extend(chunk_document(deanonymize(parse_article(article))))
    return chunks


def test_corpus_is_complete(corpus: list[dict]) -> None:
    """В выгрузке ровно 20 статей раздела 44030."""
    assert len(corpus) == 20
    assert {int(a["section_id"]) for a in corpus} == {44030}


def test_every_document_parses(corpus: list[dict]) -> None:
    """Каждая статья разбирается без исключений и даёт разделы."""
    for article in corpus:
        document = parse_article(article)
        assert document.sections, f"статья {article['id']}: разделов не найдено"
        assert document.atoms(), f"статья {article['id']}: атомов не найдено"


def test_pipeline_yield_does_not_degrade(indexed: list) -> None:
    """Объём индекса не падает ниже достигнутого уровня.

    На 08.08.2026 конвейер даёт 1446 чанков. Порог с запасом вниз ловит
    регрессию разбора (например, поломку регулярки номера пункта), но не
    реагирует на мелкие правки чанкинга.
    """
    assert len(indexed) >= 1300, f"чанков стало {len(indexed)} — проверить разбор"


def test_chunk_sizes_are_sane(indexed: list) -> None:
    """Размеры чанков лежат в рабочем диапазоне."""
    lengths = [len(chunk.text) for chunk in indexed]
    assert max(lengths) <= MAX_CHARS
    assert 400 <= statistics.median(lengths) <= 1200


def test_chunk_ids_are_unique(indexed: list) -> None:
    """Идентификаторы чанков не сталкиваются между документами."""
    ids = [chunk.chunk_id for chunk in indexed]
    assert len(ids) == len(set(ids))


def test_point_ids_are_unique(indexed: list) -> None:
    """Идентификаторы точек Qdrant не сталкиваются.

    Коллизия означала бы молчаливую перезапись чанка при загрузке.
    """
    point_ids = [chunk.point_id() for chunk in indexed]
    assert len(point_ids) == len(set(point_ids))


def test_no_placeholder_reaches_the_index(indexed: list) -> None:
    """Ни одна метка анонимизации не попадает в индексируемый текст."""
    leaked = [c.chunk_id for c in indexed if PLACEHOLDER.search(c.text)]
    assert not leaked, f"метки утекли в чанки: {leaked[:5]}"


def test_deanonymization_actually_restores(indexed: list) -> None:
    """Восстановление работает на заметной доле корпуса.

    Если доля упадёт до нуля, значит сломался разбор раздела терминов —
    поиск по аббревиатурам деградирует молча.
    """
    restored = sum(1 for chunk in indexed if chunk.restored)
    assert restored / len(indexed) >= 0.10


def test_table_of_contents_excluded(indexed: list) -> None:
    """Оглавление не попадает в индекс."""
    assert all(c.section_type is not SectionType.SODERZHANIE for c in indexed)


def test_citations_have_document_code(indexed: list) -> None:
    """У каждого чанка есть код документа — без него цитата бесполезна."""
    assert all(chunk.doc_code for chunk in indexed)


def test_table_regions_have_no_clause(indexed: list) -> None:
    """В табличных зонах номер пункта не выдаётся: он там недостоверен."""
    assert all(chunk.clause is None for chunk in indexed if chunk.is_table_region)
