from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from app.agent import RESEARCH_SYSTEM_PROMPT
from app.config import Settings
from app.nodes.answer import render_findings
from app.nodes.parsing import collect_source_texts, ground_findings, parse_findings_block
from app.nodes.research import make_research_node
from app.state import merge_findings
from evals.evaluators import grounding_evaluator

PAGE = (
    "Navigation | Home | About\n\n"
    "The prohibitions on unacceptable-risk AI systems apply from 2 February 2025, "
    "six months after entry into force."
)


def _finding(claim="c", urls=("https://a.dev",), confidence="high"):
    return {"claim": claim, "source_urls": list(urls), "confidence": confidence}


def _fetch_transcript(final: str):
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "web_fetch", "args": {"url": "https://a.dev"}, "id": "f1"}],
        ),
        ToolMessage(content=f"[S1] https://a.dev\n\n{PAGE}", tool_call_id="f1"),
        AIMessage(content=final),
    ]


def test_prompt_asks_for_verbatim_quote():
    assert '| quote: "' in RESEARCH_SYSTEM_PROMPT
    assert "word for word" in RESEARCH_SYSTEM_PROMPT


def test_parse_reads_quote_and_strips_marks():
    parsed = parse_findings_block(
        "## FINDINGS\n"
        '- [S1] A | confidence: high | quote: "apply from 2 February 2025"\n'
        "- [S2] B | confidence: low | quote: “curly marks too”\n"
        "- [S3] C | confidence: medium\n"
    )
    assert [f["claim"] for f in parsed.findings] == ["A", "B", "C"]
    assert parsed.quotes == ["apply from 2 February 2025", "curly marks too", ""]
    assert parsed.dropped == []


def test_collect_source_texts_strips_fetch_header_and_keeps_search_excerpts():
    messages = _fetch_transcript("done") + [
        AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "s1"}],
        ),
        ToolMessage(
            content="SEARCH_RESULTS (1 results, 1 search executed):\n\n"
            "[S2] https://b.dev | B\nEXCERPT: search snippet for b",
            tool_call_id="s1",
        ),
    ]
    texts = collect_source_texts(messages)
    assert texts["https://a.dev"] == PAGE
    assert texts["https://b.dev"] == "search snippet for b"


def test_quote_found_despite_case_punctuation_and_ellipsis():
    findings, weak = ground_findings(
        [_finding()],
        ["THE prohibitions on unacceptable risk AI systems ... six months after entry"],
        {"https://a.dev": PAGE},
    )
    assert findings[0]["grounded"] is True
    assert findings[0]["confidence"] == "high"
    assert weak == []


def test_missing_or_too_short_quote_caps_confidence_at_medium():
    findings, weak = ground_findings(
        [_finding(), _finding(confidence="low"), _finding()],
        ["", "", "2 February 2025"],
        {"https://a.dev": PAGE},
    )
    assert [f["confidence"] for f in findings] == ["medium", "low", "medium"]
    assert not any(f["grounded"] for f in findings)
    assert weak == []


def test_quote_absent_from_cited_source_drops_to_low_and_is_reported():
    findings, weak = ground_findings(
        [_finding(claim="made up")],
        ["the act was repealed in its entirety last year"],
        {"https://a.dev": PAGE},
    )
    assert findings[0]["grounded"] is False
    assert findings[0]["confidence"] == "low"
    assert weak == ["quote not found in cited source(s): made up"]


def test_quote_only_counts_in_a_source_the_finding_cites():
    findings, _ = ground_findings(
        [_finding(urls=("https://b.dev",))],
        ["apply from 2 February 2025, six months"],
        {"https://a.dev": PAGE, "https://b.dev": "unrelated text"},
    )
    assert findings[0]["grounded"] is False


def test_research_node_grounds_findings_and_uses_quote_as_excerpt():
    final = (
        "Body.\n\n## FINDINGS\n"
        '- [S1] Bans apply from Feb 2025 | confidence: high | quote: "apply from '
        '2 February 2025, six months after entry into force"\n'
        '- [S1] Invented | confidence: high | quote: "this sentence is not on the page"\n'
    )

    class Agent:
        def invoke(self, payload, config=None):
            return {"messages": _fetch_transcript(final)}

    delta = make_research_node(Agent(), Settings(_env_file=None))(  # type: ignore[call-arg]
        {"question": "q", "iteration": 0}
    )
    good, bad = delta["findings"]
    assert good["grounded"] is True and good["confidence"] == "high"
    assert bad["grounded"] is False and bad["confidence"] == "low"
    assert "quote not found in cited source(s): Invented" in delta["weak_claims"]
    # The page's first 500 chars start with navigation; the quote replaces them.
    assert delta["sources"][0]["excerpt"] == good["quote"]


def test_merge_findings_adopts_a_later_grounded_quote():
    first = dict(_finding(claim="X"), quote="", grounded=False)
    second = dict(_finding(claim="x"), quote="the supporting words here", grounded=True)
    merged = merge_findings([first], [second])
    assert merged[0]["grounded"] is True
    assert merged[0]["quote"] == "the supporting words here"


def test_render_findings_numbers_refs_and_separates_uncertain():
    sources = [{"url": "https://a.dev"}, {"url": "https://b.dev"}]
    findings = [
        dict(_finding("solid", ["https://b.dev", "https://a.dev"]), quote="q1", grounded=True),
        _finding("shaky", ["https://a.dev"], confidence="low"),
        _finding("orphan", ["unresolved:S9"]),
    ]
    text = render_findings(findings, sources)
    assert '- (high) solid [1][2]\n  quote: "q1"' in text
    assert "Uncertain findings" in text
    assert text.index("shaky") > text.index("Uncertain findings")
    assert "orphan" not in text
    assert "https://" not in text


def test_grounding_evaluator_scores_citation_health():
    run = SimpleNamespace(
        outputs={
            "answer": "Claim [1] and [3].",
            "sources": [{"url": "https://a.dev"}, {"url": "https://b.dev"}],
            "findings": [
                dict(_finding(), grounded=True),
                dict(_finding(urls=("unresolved:S4",)), grounded=False),
            ],
        }
    )
    results = {r.key: r.score for r in grounding_evaluator(run, None)["results"]}
    assert results == {"answer_refs_valid": 0.5, "refs_resolved": 0.5, "findings_grounded": 0.5}


def test_grounding_evaluator_omits_finding_keys_without_findings():
    run = SimpleNamespace(outputs={"answer": "No citations.", "sources": []})
    results = {r.key: r.score for r in grounding_evaluator(run, None)["results"]}
    assert results == {"answer_refs_valid": 0.0}
