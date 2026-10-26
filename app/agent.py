from deepagents import create_deep_agent as _create_deep_agent

from app.config import REPO_ROOT, Settings, get_settings
from app.models import get_model
from app.tools.cache import fetch_cache, search_cache
from app.tools.fetch import make_web_fetch
from app.tools.search_tool import make_web_search
from app.tools.sources import SourceRegistry

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
- Every web_search result and every web_fetch page is labeled by the tool
  with a source ID such as [S3]. Cite a source ONLY by the exact ID the tool
  gave it. Never renumber IDs and never cite an ID you did not see in a tool
  result.

Output contract — end EVERY final reply with exactly this block:

## FINDINGS
- [S1] <one factual claim> | confidence: high|medium|low | quote: "<exact words from S1>"
- [S1][S2] <claim backed by two sources> | confidence: high|medium|low | quote: "<exact words>"

Where [Sn] is the source ID the tool printed next to that source's URL.
Cite EVERY source that supports the claim, not just the first. The quote is
a short passage (one sentence is enough) copied word for word from the text a
tool returned for one of the cited sources — never paraphrase it, translate
it or write it yourself; quotes are checked against the source text, and a
claim whose quote is not found there is downgraded. One line per claim. No
prose inside the block."""

# Attempts per researcher model call when the provider rate-limits. Applied per
# call (inside the agent loop) so a 429 retries one step, not the whole run.
RESEARCHER_RATE_LIMIT_ATTEMPTS = 5


def _skills_backend():
    """Expose only skills/ to the agent, read-only; everything else is in-memory.

    deepagents hands the agent read/write/edit/delete file tools over its
    backend. Rooting a FilesystemBackend at the repo would let a prompt-injected
    web page read `.env` (then leak it through a web_fetch URL) or rewrite code.
    Unrouted paths land in StateBackend — per-run scratch that never touches
    disk — and the returned permissions deny writes under /skills/.
    """
    from deepagents import FilesystemPermission
    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": FilesystemBackend(
                root_dir=str(REPO_ROOT / "skills"), virtual_mode=True
            )
        },
    )
    permissions = [
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")
    ]
    return backend, permissions


def build_research_agent(settings: Settings | None = None, usage=None, registry=None):
    s = settings or get_settings()
    model = get_model("researcher", s)
    model.rate_limit_attempts = RESEARCHER_RATE_LIMIT_ATTEMPTS
    # One registry for both tools, so a URL found by web_search and then
    # fetched keeps the same [Sn] ID.
    registry = registry or SourceRegistry()
    kwargs = dict(
        model=model,
        tools=[
            make_web_search(s, usage=usage, cache=search_cache, registry=registry),
            make_web_fetch(s.fetch, cache=fetch_cache, registry=registry),
        ],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )
    if s.skills_enabled and (REPO_ROOT / "skills").is_dir():
        kwargs["backend"], kwargs["permissions"] = _skills_backend()
        kwargs["skills"] = ["/skills/"]
    return _create_deep_agent(**kwargs)
