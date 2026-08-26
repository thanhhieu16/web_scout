from types import SimpleNamespace

from app.nodes.parsing import (
    extract_url_citations,
    find_unknown_refs,
    map_refs_to_urls,
    parse_findings_block,
)

SAMPLE = """Nghiên cứu tổng hợp.

LangGraph chạy vòng lặp state machine còn Temporal chạy workflow durable.

## FINDINGS
- [S1] LangGraph is a state-machine orchestration library | confidence: high
- [S2] Temporal provides durable execution with automatic replay | confidence: medium
"""


def test_parse_findings_block_extracts_and_strips():
    parsed = parse_findings_block(SAMPLE)
    assert len(parsed.findings) == 2
    assert parsed.findings[0]["claim"].startswith("LangGraph is")
    assert parsed.refs[0] == ["S1"]
    assert parsed.findings[0]["confidence"] == "high"
    assert "FINDINGS" not in parsed.narrative
    assert parsed.narrative.strip().startswith("Nghiên cứu")


def test_parse_findings_block_absent():
    parsed = parse_findings_block("Chỉ là văn bản thường.")
    assert parsed.findings == []
    assert parsed.narrative == "Chỉ là văn bản thường."


def test_parse_findings_bad_confidence_skipped():
    text = "## FINDINGS\n- [S1] claim one | confidence: very-high\n"
    assert parse_findings_block(text).findings == []


def _msg(annotations):
    return SimpleNamespace(
        annotations=annotations,
        additional_kwargs={},
        response_metadata={},
    )


def test_extract_url_citations_from_annotations_attr():
    m = _msg(
        [
            {
                "type": "url_citation",
                "url_citation": {"url": "https://a.dev", "title": "A", "content": "ca"},
            },
            {"type": "other"},
        ]
    )
    cites = extract_url_citations(m)
    assert cites == [{"url": "https://a.dev", "title": "A", "content": "ca"}]


def test_extract_url_citations_from_additional_kwargs():
    m = SimpleNamespace(
        annotations=None,
        additional_kwargs={
            "annotations": [
                {"url_citation": {"url": "https://b.dev", "title": "B", "content": ""}}
            ]
        },
        response_metadata={},
    )
    assert len(extract_url_citations(m)) == 1


def test_find_unknown_refs_flags_missing_citation():
    findings = [
        {"claim": "x", "source_urls": ["https://a.dev"], "confidence": "high"},
        {"claim": "y", "source_urls": ["https://ghost.dev"], "confidence": "low"},
    ]
    citations = [
        {"url": "https://a.dev", "title": "A", "content": "long" * 300},
        {"url": "https://a.dev", "title": "A dup", "content": ""},
    ]
    assert find_unknown_refs(findings, citations) == ["https://ghost.dev"]


def test_find_unknown_refs_flags_unresolved_prefix():
    findings = [{"claim": "g", "source_urls": ["unresolved:S9"], "confidence": "low"}]
    assert find_unknown_refs(findings, []) == ["unresolved:S9"]


def test_parse_maps_source_refs_to_urls():
    parsed = parse_findings_block(SAMPLE)
    citations = [
        {"url": "https://langchain.ai", "title": "LG docs", "content": ""},
        {"url": "https://temporal.io", "title": "Temporal docs", "content": ""},
    ]
    mapped = map_refs_to_urls(parsed.findings, parsed.refs, citations)
    assert mapped[0]["source_urls"] == ["https://langchain.ai"]
    assert mapped[1]["source_urls"] == ["https://temporal.io"]


def test_map_refs_unresolved_marker():
    parsed = parse_findings_block("## FINDINGS\n- [S9] Ghost | confidence: low\n")
    mapped = map_refs_to_urls(
        parsed.findings, parsed.refs, [{"url": "https://y.dev", "title": "Y", "content": ""}]
    )
    assert mapped[0]["source_urls"] == ["unresolved:S9"]


def test_parse_reports_dropped_lines():
    text = (
        "Body.\n\n## FINDINGS\n"
        "- [S1] good claim | confidence: high\n"
        "- [S2] bad claim | confidence: very-high\n"
        "just some prose\n"
    )
    parsed = parse_findings_block(text)
    assert len(parsed.findings) == 1
    assert parsed.dropped == [
        "- [S2] bad claim | confidence: very-high",
    ]
    assert parsed.block_found is True


def test_parse_reports_missing_block():
    parsed = parse_findings_block("No contract here at all.")
    assert parsed.block_found is False
    assert parsed.findings == []
    assert parsed.dropped == []
    assert parsed.narrative == "No contract here at all."


def test_parse_ignores_trailing_prose_after_block():
    text = (
        "Body.\n\n## FINDINGS\n"
        "- [S1] good claim | confidence: high\n\n"
        "Hope this helps! Let me know if you want more detail.\n"
    )
    parsed = parse_findings_block(text)
    assert len(parsed.findings) == 1
    assert parsed.dropped == []


def test_parse_reports_dropped_line_with_no_ref_at_all():
    text = "## FINDINGS\n- Claim without any ref | confidence: high\n"
    parsed = parse_findings_block(text)
    assert parsed.findings == []
    assert parsed.dropped == ["- Claim without any ref | confidence: high"]


def test_parse_still_ignores_plain_trailing_prose():
    text = "## FINDINGS\n- [S1] good claim | confidence: high\n\nHope this helps!\n"
    parsed = parse_findings_block(text)
    assert len(parsed.findings) == 1
    assert parsed.dropped == []


def test_parse_accepts_multiple_refs_per_claim():
    text = "## FINDINGS\n- [S1][S2] Cross-checked claim | confidence: high\n"
    parsed = parse_findings_block(text)
    assert parsed.refs == [["S1", "S2"]]
    assert parsed.findings[0]["claim"] == "Cross-checked claim"


def test_parse_accepts_comma_separated_refs():
    text = "## FINDINGS\n- [S1], [S3] Another claim | confidence: medium\n"
    assert parse_findings_block(text).refs == [["S1", "S3"]]


def test_multi_ref_maps_to_multiple_urls():
    parsed = parse_findings_block(
        "## FINDINGS\n- [S1][S2] Cross-checked | confidence: high\n"
    )
    citations = [{"url": "https://a.dev"}, {"url": "https://b.dev"}]
    mapped = map_refs_to_urls(parsed.findings, parsed.refs, citations)
    assert mapped[0]["source_urls"] == ["https://a.dev", "https://b.dev"]
