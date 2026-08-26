from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def test_skill_md_frontmatter_valid():
    text = (ROOT / "skills" / "web-research" / "SKILL.md").read_text(encoding="utf-8")
    front = text.split("---")[1]
    meta = yaml.safe_load(front)
    assert meta["name"] == "web-research"
    assert "when" in meta["description"].lower()
    assert len(text.split("---", 2)[2]) > 500


def test_skill_body_has_methodology_sections():
    body = (ROOT / "skills" / "web-research" / "SKILL.md").read_text(encoding="utf-8")
    for heading in (
        "Source priority",
        "Verify important claims",
        "Handle conflicts",
        "Citation integrity",
    ):
        assert heading.lower() in body.lower()
