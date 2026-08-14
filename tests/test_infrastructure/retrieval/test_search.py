import pytest

from infrastructure.retrieval.search import Chunk, Citation, to_citation
from tests.fakes.fake_search import MockSearch


class TestMockSearch:
    @pytest.mark.asyncio
    async def test_returns_hits_for_known_query(self):
        search = MockSearch()
        hits = await search.query("как часто проводится соут")
        assert hits
        assert hits[0][0].doc_code == "R-U2-23"

    @pytest.mark.asyncio
    async def test_empty_query_returns_nothing(self):
        search = MockSearch()
        assert await search.query("") == []
        assert await search.query("   ") == []

    @pytest.mark.asyncio
    async def test_unrelated_query_returns_nothing(self):
        search = MockSearch()
        assert await search.query("какой сегодня курс доллара") == []

    @pytest.mark.asyncio
    async def test_sections_filter(self):
        search = MockSearch()
        hits = await search.query("обучение", sections=["procedura"])
        assert all(h[0].section_type == "procedura" for h in hits)

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        search = MockSearch()
        hits = await search.query("проводится", limit=1)
        assert len(hits) == 1

    @pytest.mark.asyncio
    async def test_sorted_by_score_desc(self):
        search = MockSearch()
        hits = await search.query("проводится оценка условий труда")
        scores = [score for _, score in hits]
        assert scores == sorted(scores, reverse=True)


class TestToCitation:
    def test_regular_chunk(self):
        chunk = Chunk(
            chunk_id="R-U2-23#c001",
            doc_code="R-U2-23",
            doc_title="Проведение специальной оценки условий труда",
            section="3.7",
            section_type="procedura",
            clause="3.7",
            text="Текст документа для ответа.",
        )
        citation = to_citation(chunk, n=1, score=2.0)
        assert isinstance(citation, Citation)
        assert citation.to_dict() == {
            "n": 1,
            "doc_code": "R-U2-23",
            "doc_title": "Проведение специальной оценки условий труда",
            "clause": "3.7",
            "section_no": "3.7",
            "section_title": "3.7",
            "chunk_id": "R-U2-23#c001",
            "score": 2.0,
            "snippet": "Текст документа для ответа.",
            # Полный текст чанка: он уходит в контекст модели, тогда как
            # snippet обрезан до 400 символов ради панели источников.
            "text": "Текст документа для ответа.",
            "url": None,
        }

    def test_table_chunk_has_null_clause(self):
        chunk = Chunk(
            chunk_id="R-U2-59#c1",
            doc_code="R-U2-59",
            doc_title="Идентификация опасностей и оценка рисков",
            section="4.1",
            section_type="procedura",
            clause="4.1",
            text="таблица",
            contains_table=True,
        )
        citation = to_citation(chunk, n=1)
        assert citation.clause is None

    def test_snippet_truncated_to_400(self):
        chunk = Chunk(
            chunk_id="c",
            doc_code="D",
            doc_title="T",
            section="1",
            section_type="procedura",
            clause=None,
            text="x" * 500,
        )
        citation = to_citation(chunk, n=1)
        assert len(citation.snippet) == 400
        assert citation.to_dict()["snippet"] == "x" * 400
