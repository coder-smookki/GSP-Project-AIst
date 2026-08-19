from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_code: str
    doc_title: str
    section: str
    section_type: str
    clause: str | None
    text: str
    contains_table: bool = False


@dataclass(frozen=True)
class Citation:
    n: int
    doc_code: str
    doc_title: str
    clause: str | None
    section_no: str
    section_title: str
    chunk_id: str
    score: float
    snippet: str
    text: str = ""
    url: str | None = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "doc_code": self.doc_code,
            "doc_title": self.doc_title,
            "clause": self.clause,
            "section_no": self.section_no,
            "section_title": self.section_title,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "snippet": self.snippet,
            "text": self.text,
            "url": self.url,
        }


def to_citation(chunk: Chunk, n: int, score: float = 0.0) -> Citation:
    return Citation(
        n=n,
        doc_code=chunk.doc_code,
        doc_title=chunk.doc_title,
        clause=None if chunk.contains_table else chunk.clause,
        section_no=chunk.section,
        section_title=chunk.section,
        chunk_id=chunk.chunk_id,
        score=score,
        snippet=chunk.text[:400],
        text=chunk.text,
    )


class Search:
    async def query(
        self,
        query: str,
        sections: list[str] | None = None,
        limit: int | None = None,
        areas: list[str] | None = None,
        docs: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Ищет фрагменты базы знаний под запрос.

        Args:
            query: Текст поискового запроса.
            sections: Типы разделов (``section_type``); пусто - все.
            limit: Сколько фрагментов вернуть; пусто - настройка по умолчанию.
            areas: Области знаний (``area``); пусто - все.
            docs: Коды документов (``doc_code``); пусто - все.

        Returns:
            list[tuple[Chunk, float]]: Пары «фрагмент, оценка» по убыванию.
        """
        raise NotImplementedError
