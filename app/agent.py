from deepagents import create_deep_agent as _create_deep_agent

from app.config import REPO_ROOT, Settings, get_settings
from app.models import get_model
from app.tools.fetch import make_web_fetch
from app.tools.search_tool import make_web_search

RESEARCH_SYSTEM_PROMPT = """You are WebScout's research agent.

Mission: answer the research request using current web evidence.

Method:
1. Break the request into factual claims that need evidence.
2. You MUST call the web_search tool at least once before answering.
   Fetching URLs without searching first is forbidden. web_fetch is for
   reading pages you found via web_search results (or URLs the user
   explicitly gave), never for guessing addresses.
3. Use the web_fetch tool to read promising pages. Prefer primary sources:
   official documentation, standards, government sites, academic papers,
   primary company sources. Use secondary sources only for interpretation.
4. Cross-check contested claims across at least two sources when possible.
5. Track uncertainty. If credible sources disagree, say so explicitly.
6. Keep it bounded: a handful of searches and fetches is enough. Then stop
   and write your reply — an honest incomplete answer beats an endless hunt.

Integrity rules:
- Never invent a URL, citation or source. Reference ONLY URLs that appeared
  in your search results or that you fetched yourself.
- Number your citations [S1], [S2], ... in the order you first used each
  source.

Output contract — end EVERY final reply with exactly this block:

## FINDINGS
- [S1] <one factual claim> | confidence: high|medium|low
- [S1][S2] <a claim confirmed by two sources> | confidence: high|medium|low

Where [Sn] refers to the nth URL in the sources you used, counted in the
order you first used them. Cite EVERY source that supports the claim, not
just the first. One line per claim. No prose inside the block."""


def build_research_agent(settings: Settings | None = None, usage=None):
    s = settings or get_settings()
    model = get_model("researcher", s)
    kwargs = dict(
        model=model,
        tools=[make_web_search(s, usage=usage), make_web_fetch(s.fetch)],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )
    if s.skills_enabled and (REPO_ROOT / "skills").is_dir():
        from deepagents.backends.filesystem import FilesystemBackend

        kwargs["backend"] = FilesystemBackend(root_dir=str(REPO_ROOT))
        kwargs["skills"] = ["skills/"]
    return _create_deep_agent(**kwargs)
