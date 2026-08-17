"""A/B-замер чтения чисел из таблиц: поиск → фрагменты → LLM извлекает число.

Метрики поиска не видят пользу от рендера таблиц в Markdown: уплощённый фрагмент
— надмножество (пункт + шапка + все ячейки) и ранжируется не хуже. Проблема,
которую решает Markdown, — в восприятии формата LLM на этапе генерации: из
сплошного текста модель не всегда понимает, какому столбцу какое число
принадлежит.

Здесь каждый вопрос прогоняется как у агента: поиск возвращает top-k фрагментов,
LLM отвечает строго по ним, и проверяется, назвало ли оно ожидаемое число.
Сравнение base/вариант — парный точный тест МакНемара по доле верных извлечений.

Запуск::

    python eval/table_extraction.py \
        --variants variants.json \
        --golden tests/fixtures/golden/tables_questions.json

Варианты — тот же JSON-файл, что у ``eval ab``: первый вариант — base.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from eval.config import Variant
from eval.golden import GoldenSet
from eval.metrics import BaseMetric, answerable_outcomes
from eval.result import Outcome, RunResult, Verdict
from eval.stats import McNemarTest

SYSTEM_PROMPT = (
    "Ты помощник по корпоративной базе знаний. Отвечай строго на основе "
    "приведённых фрагментов документов. Если в фрагментах нет ответа, ответь "
    "одним словом «НЕТ ДАННЫХ». В остальных случаях дай краткий ответ: число "
    "с единицей измерения."
)


def expected_number(answer: str) -> int | None:
    """Первое целое число в эталонном ответе.

    Args:
        answer: Эталонный ответ из золотого набора.

    Returns:
        int | None: Ожидаемое число или ``None``, если в ответе его нет.
    """
    match = re.search(r"\d+", answer)
    return int(match.group(0)) if match else None


def extraction_correct(reply: str, expected: int) -> bool:
    """Верно ли LLM назвало ожидаемое число.

    Args:
        reply: Ответ модели.
        expected: Ожидаемое число.

    Returns:
        bool: ``True``, если число встречается в ответе отдельным числом.
    """
    return re.search(rf"(?<!\d){expected}(?!\d)", reply) is not None


def build_prompt(question: str, chunks: list[str]) -> list:
    """Собирает сообщения для модели.

    Args:
        question: Вопрос пользователя.
        chunks: Тексты найденных фрагментов.

    Returns:
        list: Сообщения диалога для ``ModelClient.ainvoke``.
    """
    context = "\n\n".join(
        f"Фрагмент {index}:\n{text}" for index, text in enumerate(chunks, start=1)
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Вопрос: {question}\n\nФрагменты документов:\n{context}"),
    ]


@dataclass
class ExtractionResult:
    """Итог замера по варианту.

    Attributes:
        outcomes: Итоги по вопросам (``rank`` — 1 верно, 0 нет, ``None`` нет
            фрагментов).
        replies: Ответы модели по вопросам.
    """

    outcomes: list[Outcome] = field(default_factory=list)
    replies: list[str | None] = field(default_factory=list)

    def accuracy(self) -> float:
        """Доля вопросов с верно извлечённым числом."""
        relevant = answerable_outcomes(self.outcomes)
        if not relevant:
            return 0.0
        return sum(1 for outcome in relevant if outcome.rank) / len(relevant)


class ExtractionAccuracy(BaseMetric):
    """Доля отвечаемых вопросов, где модель назвала ожидаемое число."""

    kind: str = "categorical"

    @property
    def key(self) -> str:
        return "extraction_acc"

    @property
    def display_name(self) -> str:
        return "Извлечение числа из таблицы"

    def evaluate(self, outcomes) -> float:
        relevant = answerable_outcomes(outcomes)
        if not relevant:
            return 0.0
        return sum(1 for outcome in relevant if outcome.rank) / len(relevant)


ACCURACY = ExtractionAccuracy()


async def run_variant(
    variant: Variant,
    golden: GoldenSet,
    *,
    top_k: int,
    limit: int | None,
    verbose: bool,
) -> ExtractionResult:
    """Прогоняет вариант: поиск, ответ модели, проверка числа.

    Args:
        variant: Вариант (оверрайды настроек, в частности коллекция Qdrant).
        golden: Золотой набор.
        top_k: Сколько фрагментов отдавать модели.
        limit: Ограничить число вопросов.
        verbose: Печатать ответы модели.

    Returns:
        ExtractionResult: Итоги по вопросам и ответы модели.
    """
    from eval.variants import build_search, make_settings
    from infrastructure.config.model_registry import get_model, setup_registry

    settings = make_settings(variant)
    setup_registry(settings)
    search = build_search(settings)
    model = get_model("default")

    items = list(golden.answerable)
    if limit is not None:
        items = items[:limit]

    result = ExtractionResult()
    for index, item in enumerate(items, start=1):
            question = item["question"]
            hits = await search.query(question)
            chunks = [chunk.text for chunk, _ in hits[:top_k]]
            expected = expected_number(item.get("answer") or "")
            reply: str | None = None
            if expected is None:
                correct = None
            elif not chunks:
                correct = False
            else:
                message = await model.ainvoke(build_prompt(question, chunks))
                reply = str(message.content).strip()
                correct = extraction_correct(reply, expected)

            outcome = Outcome(
                question=question,
                answerable=True,
                rank=1 if correct else 0 if correct is not None else None,
                returned=len(hits),
                answer=reply,
            )
            result.outcomes.append(outcome)
            result.replies.append(reply)
            if verbose:
                print(
                    f"{index:>2}/{len(items)} [{variant.name}] "
                    f"{question[:50]} -> ожидалось {expected}, "
                    f"ответ: {reply[:60]!r}"
                )

    return result


def _to_run_result(variant: Variant, extraction: ExtractionResult) -> RunResult:
    return RunResult(
        variant=variant.name,
        environment="table-extraction",
        metrics={ACCURACY.key: extraction.accuracy()},
        outcomes=extraction.outcomes,
        created_at="",
        meta={"runner": "table_extraction"},
    )


def print_report(
    base: Variant,
    md: Variant,
    base_result: ExtractionResult,
    md_result: ExtractionResult,
) -> None:
    """Печатает точность, A/B-вердикт и разбор по вопросам."""
    base_run = _to_run_result(base, base_result)
    md_run = _to_run_result(md, md_result)
    verdict: Verdict = McNemarTest().compare(ACCURACY, base_run, md_run)

    print("\n" + "=" * 60)
    print(f"{base.name} (уплощённые таблицы):")
    print(f"  Извлечение числа    {base_result.accuracy():.3f}")
    print(f"{md.name} (Markdown-таблицы):")
    print(f"  Извлечение числа    {md_result.accuracy():.3f}")
    print("=" * 60)
    print(
        f"{ACCURACY.display_name}: "
        f"{verdict.base_value:.3f} -> {verdict.variant_value:.3f} "
        f"({verdict.direction}, p={verdict.p_value:.4f})"
    )
    if verdict.direction == "better" and verdict.p_value < 0.05:
        print("Статистически значимое улучшение — критерий приёмки выполнен.")
    elif verdict.direction == "no_diff":
        print("Значимой разницы нет (p >= 0.05).")

    print("\nРазбор по вопросам:")
    print(f"{'вопрос':<55} {'ожид':>5} {'base':>7} {'md':>7}")
    for base_outcome, md_outcome in zip(
        base_result.outcomes, md_result.outcomes, strict=True
    ):
        print(
            f"{base_outcome.question[:53]:<55} "
            f"{expected_number(base_outcome.answer or '') or '—':>5} "
            f"{'✓' if base_outcome.rank else '✗':>7} "
            f"{'✓' if md_outcome.rank else '✗':>7}"
        )


async def run_all(
    args: argparse.Namespace, variants: list[Variant], golden: GoldenSet
) -> None:
    """Прогоняет base и вариант в одном цикле событий, затем закрывает клиент."""
    from infrastructure.retrieval.factory import shutdown

    base_result = await run_variant(
        variants[0], golden, top_k=args.top_k, limit=args.limit, verbose=args.verbose
    )
    md_result = await run_variant(
        variants[1], golden, top_k=args.top_k, limit=args.limit, verbose=args.verbose
    )
    await shutdown()
    print_report(variants[0], variants[1], base_result, md_result)


def main() -> int:
    """Точка входа."""
    from eval.compare import load_variants

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True, help="путь к JSON-файлу вариантов")
    parser.add_argument(
        "--golden",
        default="tests/fixtures/golden/tables_questions.json",
        help="путь к золотому набору",
    )
    parser.add_argument("--top-k", type=int, default=3, help="фрагментов на вопрос")
    parser.add_argument("--limit", type=int, help="ограничить число вопросов")
    parser.add_argument("--verbose", action="store_true", help="печатать ответы модели")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        variants = load_variants(args.variants)
        golden = GoldenSet.load(args.golden).limited(args.limit)
    except (OSError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2
    if len(variants) < 2:
        print("нужно минимум два варианта: base и новый", file=sys.stderr)
        return 2

    asyncio.run(run_all(args, variants, golden))
    return 0


if __name__ == "__main__":
    import logging

    raise SystemExit(main())