from infrastructure.retrieval.search import Chunk, Search


class MockSearch(Search):
    """Тестовая реализация Search: keyword-overlap по синтетическому корпусу.

    Заменяет реальный слой (Qdrant, ingest, реранк) пока его не написал
    владелец ретрива. Используется только в тестах (tests/).
    """

    _STOPWORDS = {  # noqa: RUF012
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
        words = text.lower().replace(",", " ").replace(".", " ").split()
        return {w for w in words if w not in MockSearch._STOPWORDS and len(w) > 1}

    async def query(
        self,
        query: str,
        sections: list[str] | None = None,
        limit: int = 8,
    ) -> list[tuple[Chunk, float]]:
        q_tokens = self._tokens(query)
        if not q_tokens:
            return []

        scored: list[tuple[float, Chunk]] = []
        for chunk in self._CORPUS:
            if sections and chunk.section_type not in sections:
                continue
            common = q_tokens & self._tokens(chunk.text)
            if not common:
                continue
            scored.append((float(len(common)), chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(chunk, score) for score, chunk in scored[:limit]]
