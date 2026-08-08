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
    """Цитата на источник.

    Args:
        n: Номер источника, совпадает с маркером [n] в тексте ответа.
        doc_code: Код документа.
        doc_title: Название документа.
        clause: Номер пункта; ``None`` для табличных зон.
        section_no: Номер раздела.
        section_title: Заголовок раздела.
        chunk_id: Идентификатор чанка.
        score: Счёт ранжирования.
        snippet: Короткая выдержка для панели источников.
        text: Полный текст чанка — уходит в контекст модели.
        url: Ссылка на источник, когда она появится.
    """

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
        # Два разных поля с разным назначением. `snippet` короткий: он идёт
        # в панель источников, где важно, чтобы карточка не разрасталась.
        # `text` — полный текст чанка, и именно он уходит в контекст модели.
        # Раньше в промпт попадал только `snippet`, и при медиане чанка в
        # 822 символа модель не видела больше половины найденного пункта —
        # включая числовые требования, которые обычно стоят во второй
        # половине абзаца.
        snippet=chunk.text[:400],
        text=chunk.text,
    )


class Search:
    async def query(
        self,
        query: str,
        sections: list[str] | None = None,
        limit: int = 8,
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError
