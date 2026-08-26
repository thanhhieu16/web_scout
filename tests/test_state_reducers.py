from app.state import merge_findings, merge_sources, merge_weak_claims


def _src(url, title="T"):
    return {"url": url, "title": title, "source_type": "secondary", "excerpt": ""}


def _find(claim, urls, conf):
    return {"claim": claim, "source_urls": list(urls), "confidence": conf}


def test_merge_sources_dedupes_by_url_and_keeps_first_order():
    prior = [_src("https://a.dev", "A"), _src("https://b.dev", "B")]
    new = [_src("https://b.dev", "B again"), _src("https://c.dev", "C")]
    merged = merge_sources(prior, new)
    assert [s["url"] for s in merged] == [
        "https://a.dev",
        "https://b.dev",
        "https://c.dev",
    ]
    assert merged[1]["title"] == "B"


def test_merge_findings_unions_urls_and_keeps_lower_confidence():
    prior = [_find("Claim one", ["https://a.dev"], "high")]
    new = [_find("  claim ONE  ", ["https://b.dev"], "low")]
    merged = merge_findings(prior, new)
    assert len(merged) == 1
    assert merged[0]["claim"] == "Claim one"
    assert merged[0]["source_urls"] == ["https://a.dev", "https://b.dev"]
    assert merged[0]["confidence"] == "low"


def test_merge_findings_appends_distinct_claims():
    merged = merge_findings(
        [_find("one", ["https://a"], "high")],
        [_find("two", ["https://b"], "medium")],
    )
    assert [f["claim"] for f in merged] == ["one", "two"]


def test_merge_weak_claims_dedupes_preserving_order():
    assert merge_weak_claims(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_reducers_do_not_mutate_prior():
    prior = [_src("https://a.dev")]
    merge_sources(prior, [_src("https://b.dev")])
    assert len(prior) == 1
