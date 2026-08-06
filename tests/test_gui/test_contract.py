from gui.contract import build_input, decode_aist_envelope, format_sources


class TestBuildInput:
    def test_without_sections(self):
        assert build_input("привет") == {
            "messages": [{"type": "human", "content": "привет"}]
        }

    def test_with_sections(self):
        payload = build_input("привет", sections=["procedura"])
        assert payload == {
            "messages": [{"type": "human", "content": "привет"}],
            "sections": ["procedura"],
        }

    def test_empty_sections_omitted(self):
        assert "sections" not in build_input("привет", [])


class TestDecodeAistEnvelope:
    def test_v1_direct_key(self):
        chunk = {"aist": {"v": 1, "kind": "sources", "payload": {"items": []}}}
        env = decode_aist_envelope(chunk)
        assert env is not None
        assert env["kind"] == "sources"

    def test_v2_nested_params(self):
        chunk = {
            "params": {"data": {"aist": {"v": 1, "kind": "followups", "payload": {}}}}
        }
        env = decode_aist_envelope(chunk)
        assert env is not None
        assert env["kind"] == "followups"

    def test_empty_data_returns_none(self):
        assert decode_aist_envelope({}) is None

    def test_plain_custom_data_returns_none(self):
        assert decode_aist_envelope({"kind": "trace"}) is None

    def test_fallback_to_params_if_top_aist_empty(self):
        chunk = {
            "aist": None,
            "params": {"data": {"aist": {"v": 1, "kind": "related", "payload": {}}}},
        }
        env = decode_aist_envelope(chunk)
        assert env is not None
        assert env["kind"] == "related"


class TestFormatSources:
    def test_full_citation(self):
        items = [
            {
                "n": 1,
                "doc_code": "R-U2-23",
                "doc_title": "Проведение специальной оценки условий труда",
                "clause": "3.7",
                "snippet": "Синтетический пример.",
            }
        ]
        out = format_sources(items)
        assert "[1]" in out
        assert "R-U2-23" in out
        assert "п. 3.7" in out
        assert "Синтетический пример." in out

    def test_null_clause_uses_section(self):
        items = [
            {
                "n": 2,
                "doc_code": "R-U2-59",
                "doc_title": "Идентификация опасностей и оценка рисков",
                "clause": None,
                "section_no": "4.1",
                "section_title": "4.1",
            }
        ]
        out = format_sources(items)
        assert "п." not in out
        assert "R-U2-59" in out

    def test_empty_items(self):
        assert format_sources([]) == ""

    def test_full_markdown_join(self):
        items = [
            {
                "n": 1,
                "doc_code": "A",
                "doc_title": "Док 1",
                "clause": "1",
                "snippet": "s",
            },
            {
                "n": 2,
                "doc_code": "B",
                "doc_title": "Док 2",
                "clause": "2",
                "snippet": "t",
            },
        ]
        out = format_sources(items)
        assert out != ""
