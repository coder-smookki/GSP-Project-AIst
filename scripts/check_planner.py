"""Проверка планировщика: делит ли он составные запросы на подзапросы.

Декомпозиция - заявленная возможность решения, но заметить её отказ по
одному ответу нельзя: выглядит он как обычный ответ, просто неполный. Этот
набор случаев показывает поведение сразу и повторяемо.

Запуск::

    uv run python scripts/check_planner.py
    LLM_SUPPORT_MODEL=gemma4:e2b-mlx uv run python scripts/check_planner.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("planner-check")

#: Случаи: запрос, сколько подзапросов ожидаем минимум, пояснение.
CASES = [
    (
        "Мне нужны ответы на вопросы:\n"
        "Как себя вести при пожаре\n"
        "Что делать в случае нападения\n"
        "Как оказывать первую медицинскую помощь",
        3,
        "многострочный перечень с преамбулой",
    ),
    (
        "нужны нормы ГОСТ 4.512 и 5.112 и как носить каски",
        3,
        "однострочный, части через «и»",
    ),
    (
        "Что делать при пожаре? А при обнаружении утечки газа?",
        2,
        "два предложения, второе с эллипсисом",
    ),
    (
        "Расскажи про порядок проведения СОУТ и кто входит в комиссию",
        2,
        "две части одной темы",
    ),
    (
        "Как часто проводится плановая идентификация опасностей?",
        1,
        "простой вопрос, дробить нечего",
    ),
    ("Привет!", 0, "приветствие, поиск не нужен"),
]


async def main_async() -> int:
    """Прогоняет случаи через планировщик.

    Returns:
        int: Код возврата: 1, если хотя бы один случай разобран неверно.
    """
    from agent.planner import plan_search
    from infrastructure.config.model_registry import get_model, setup_registry
    from infrastructure.config.settings import get_settings

    settings = get_settings()
    setup_registry(settings)
    model = get_model("support")

    print(f"планировщик на модели {settings.llm_support_model or settings.llm_model}\n")
    failures = 0

    for question, expected, note in CASES:
        plan = await plan_search(model, question, max_requests=10, retries=1)
        got = len(plan.subqueries)
        chat = plan.mode == "chat"

        ok = chat if expected == 0 else (not chat and got >= expected)

        mark = "ок " if ok else "!! "
        failures += 0 if ok else 1
        first = question.replace("\n", " / ")[:52]
        print(
            f"[{mark}] ждали {expected}, получили {got if not chat else 'chat'}  {note}"
        )
        print(f"       запрос: {first}")
        for sub in plan.subqueries:
            shown = sub.replace("\n", " ⏎ ")
            print(f"         → {shown[:88]}")
        print()

    print(f"неверно разобрано: {failures} из {len(CASES)}")
    return 1 if failures else 0


def main() -> int:
    """Точка входа.

    Returns:
        int: Код возврата процесса.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
