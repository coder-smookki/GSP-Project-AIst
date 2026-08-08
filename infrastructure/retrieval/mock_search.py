"""Синтетический поиск для отладки интерфейса и тестов.

Реализация лежит в рантайме, а не в `tests/`, потому что на неё ссылается
фабрика поиска: прод-код не должен импортировать тестовый пакет — при сборке
образа его там просто нет.

Использовать только когда индекс недоступен: корпус синтетический, из
четырёх чанков. Выбор этой реализации логируется предупреждением на каждом
старте — «демо на заглушке» не должно быть тихим состоянием.
"""

from __future__ import annotations

import logging

from infrastructure.retrieval.search import Chunk, Search

logger = logging.getLogger(__name__)


class MockSearch(Search):
    """Поиск по совпадению слов в синтетическом корпусе."""

    _STOPWORDS = frozenset(
        {
            "и",
            "в",
            "на",
            "по",
            "с",
            "со",
            "от",
            "об",
            "о",
            "к",
            "у",
            "для",
            "как",
            "что",
            "это",
            "или",
            "не",
            "а",
            "их",
            "же",
            "при",
            "ещё",
        }
    )

    _CORPUS: list[Chunk] = [  # noqa: RUF012
        Chunk(
            chunk_id="R-U2-23#c001",
            doc_code="R-U2-23",
            doc_title="Проведение специальной оценки условий труда",
            section="3.7",
            section_type="procedura",
            clause="3.7",
            text=(
                "Специальная оценка условий труда проводится на рабочем месте "
                "не реже одного раза в пять лет. Внеплановая оценка проводится "
                "по перечню установленных случаев."
            ),
        ),
        Chunk(
            chunk_id="R-U2-23#c2",
            doc_code="R-U2-23",
            doc_title="Проведение специальной оценки условий труда",
            section="2.2",
            section_type="sokrashcheniya",
            clause="",
            text="СОУТ — специальная оценка условий труда.",
        ),
        Chunk(
            chunk_id="R-U2-59#c1",
            doc_code="R-U2-59",
            doc_title="Идентификация опасностей и оценка рисков",
            section="4.1",
            section_type="procedura",
            clause="4.1",
            text=(
                "Величина риска рассчитывается по матрице как произведение "
                "вероятности события и тяжести последствий."
            ),
        ),
        Chunk(
            chunk_id="R-U2-80#c1",
            doc_code="R-U2-80",
            doc_title="Обучение по гражданской обороне и чрезвычайным ситуациям",
            section="5.1",
            section_type="procedura",
            clause="5.1",
            text="Обучение по гражданской обороне проводится по ежегодному плану.",
        ),
    ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Разбивает текст на значимые слова.

        Args:
            text: Исходный текст.

        Returns:
            set[str]: Слова в нижнем регистре без стоп-слов и односимвольных.
        """
        for char in "?!.,:;()[]{}«»\"'—–-":
            text = text.replace(char, " ")
        return {
            word
            for word in text.lower().split()
            if word and word not in MockSearch._STOPWORDS and len(word) > 1
        }

    async def query(
        self,
        query: str,
        sections: list[str] | None = None,
        limit: int | None = None,
        areas: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Ищет по пересечению слов запроса и чанка.

        Args:
            query: Текст запроса.
            sections: Типы разделов для фильтрации.
            limit: Число результатов; ``None`` — из настроек.
            areas: Предметные области. Заглушка их не различает — параметр
                принимается ради совместимости с боевой реализацией.

        Returns:
            list[tuple[Chunk, float]]: Пары «чанк, доля совпавших слов»
            по убыванию.
        """
        wanted = self._tokens(query)
        if not wanted:
            return []

        scored: list[tuple[Chunk, float]] = []
        for chunk in self._CORPUS:
            if sections and chunk.section_type not in sections:
                continue
            overlap = wanted & self._tokens(chunk.text)
            if overlap:
                scored.append((chunk, len(overlap) / len(wanted)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: limit or 8]
