# WebScout
## Mini Deep-Research Agent — Technical Idea & Architecture

**Status:** Initial design  
**Primary stack:** LangGraph + LangChain + Deep Agents + LangSmith + ACP + OpenRouter  
**Language:** Python  
**Project type:** Web research / question-answering agent

---

# 1. Project vision

**WebScout** is a small research agent that receives a user question, searches the web, reads relevant sources, checks whether the evidence is sufficient, and returns a concise answer with citations.

Example:

> LangGraph khác Temporal ở điểm nào? Tìm thông tin mới nhất và dẫn nguồn.

Desired behavior:

```text
User question
    ↓
Understand question
    ↓
Research web
    ↓
Collect sources
    ↓
Verify evidence
    ↓
Enough?
 ┌──┴───┐
 No    Yes
 │       │
Research │
more     │
 └───────┘
    ↓
Write answer
    ↓
Answer + citations
```

The project is intentionally small.

The goal is to learn how these components work together without rebuilding an agent harness:

- Deep Agents
- LangGraph
- LangChain
- OpenRouter
- LangSmith
- ACP
- Skills

---

# 2. Core design principle

The most important rule is:

> **Deep Agents own the internal tool-use loop. LangGraph owns the research-quality loop.**

There are two different loops.

## Agent loop

```text
LLM
 ↓
search
 ↓
observe
 ↓
fetch
 ↓
observe
 ↓
search
 ↓
...
 ↓
research result
```

Do **not** build this loop manually.

Deep Agents already provide the agent harness.

---

## Product loop

```text
research
   ↓
verify
   │
   ├── insufficient → research
   │
   └── sufficient   → answer
```

This loop belongs to WebScout.

LangGraph controls it.

---

# 3. High-level architecture

```text
                       User
                        │
                       ACP
                        │
                        ▼
                  ┌───────────┐
                  │ LangGraph │
                  └─────┬─────┘
                        │
               ┌────────▼────────┐
               │ Research Agent  │
               │   Deep Agent    │
               │                 │
               │ web-research    │
               │     skill       │
               └────────┬────────┘
                        │
                ┌───────┴───────┐
                ▼               ▼
             search           fetch
                │               │
                └───────┬───────┘
                        ▼
                       Web
                        │
                        ▼
                    Verifier
                    LangChain
                        │
                 enough evidence?
                   /          \
                 no            yes
                 │              │
                 └─ research    ▼
                             Answer
                                │
                                ▼
                              User

                     OpenRouter
                         ↑
                  every model call

                     LangSmith
                         ↑
                 traces + evaluations
```

---

# 4. Technology responsibilities

## 4.1 Deep Agents

Deep Agents provide the research-agent harness.

Responsibilities:

- tool calling,
- iterative search,
- iterative fetch,
- planning,
- context management,
- subagents when needed,
- skill discovery,
- memory later if needed.

WebScout should not reimplement:

```text
LLM → tool → observation → LLM
```

---

## 4.2 LangGraph

LangGraph controls the product workflow.

Initial graph:

```text
START
  ↓
research
  ↓
verify
  ↓
sufficient?
  ├── no  → research
  └── yes → answer
              ↓
             END
```

Responsibilities:

- global state,
- research iteration count,
- conditional routing,
- loop limits,
- persistence later,
- human interruption later if required.

---

## 4.3 LangChain

LangChain is used for bounded model interactions.

Good uses in WebScout:

- structured verifier output,
- final answer generation,
- schemas,
- model abstraction,
- tools,
- prompts.

Example:

```text
Deep Agent
= open-ended research

LangChain
= bounded verification / answer generation
```

---

## 4.4 OpenRouter

OpenRouter is the model gateway.

Architecture:

```text
WebScout
   ↓
LangChain / Deep Agents
   ↓
OpenRouter
   ↓
 ┌─────────┬─────────┐
 ▼         ▼         ▼
Claude    GPT      Gemini
```

Benefits:

- one API gateway,
- model switching,
- different models per role,
- easier experimentation,
- fallback models,
- cost optimization.

---

## 4.5 LangSmith

LangSmith observes and evaluates the whole workflow.

Example trace:

```text
Question
│
├── research #1
│   ├── search
│   ├── search
│   ├── fetch
│   └── fetch
│
├── verifier #1
│   └── insufficient
│
├── research #2
│   ├── search
│   └── fetch
│
├── verifier #2
│   └── sufficient
│
└── answer
```

Metrics:

```text
correctness
citation quality
source quality
search calls
fetch calls
iterations
tokens
cost
latency
```

---

## 4.6 ACP

ACP is the external client interface.

Initial development can use CLI.

Later:

```text
Zed / compatible client
          │
         ACP
          │
          ▼
       WebScout
```

ACP can expose:

- current research stage,
- search activity,
- page-fetch activity,
- verifier status,
- final answer.

Example:

```text
✓ Understanding question
✓ Searching official documentation
✓ Reading source 1
✓ Reading source 2
◉ Verifying evidence
○ Writing answer
```

---

# 5. Research state

Keep the state small.

```python
from typing import TypedDict


class ResearchState(TypedDict):
    question: str

    research: str
    sources: list[dict]

    gaps: list[str]
    weak_claims: list[str]

    sufficient: bool

    iteration: int
    max_iterations: int

    answer: str
```

Example:

```json
{
  "question": "LangGraph khác Temporal như thế nào?",
  "research": "...",
  "sources": [
    {
      "title": "LangGraph documentation",
      "url": "...",
      "source_type": "primary"
    },
    {
      "title": "Temporal documentation",
      "url": "...",
      "source_type": "primary"
    }
  ],
  "gaps": [
    "Need better evidence about replay semantics"
  ],
  "weak_claims": [],
  "sufficient": false,
  "iteration": 1,
  "max_iterations": 3
}
```

---

# 6. Research node

The `research` node invokes a Deep Agent.

Input:

```text
Question
+
Previous research
+
Verifier gaps
```

First iteration:

```text
Question:
Compare LangGraph and Temporal.
```

Second iteration:

```text
Original question:
Compare LangGraph and Temporal.

Existing research:
...

Missing evidence:
- Temporal replay semantics
- LangGraph durability details

Research only the missing points.
```

The research agent may:

```text
search
 ↓
fetch
 ↓
search
 ↓
fetch
 ↓
compare
 ↓
return findings
```

The graph does not micromanage these steps.

---

# 7. Research-agent responsibilities

Research Agent should:

1. Understand the question.
2. Break it into claims/questions that need evidence.
3. Search for authoritative sources.
4. Read relevant pages.
5. Prefer primary sources.
6. Cross-check important claims.
7. Track uncertainty.
8. Return source information with findings.
9. Avoid inventing citations.

The research agent should not decide when the overall product workflow is finished.

That belongs to the verifier + LangGraph.

---

# 8. Web tools

V1 needs only two conceptual tools.

## Search

```python
web_search(query)
```

Returns:

```text
title
url
snippet
source/domain
```

---

## Fetch

```python
web_fetch(url)
```

Returns normalized page content.

---

The exact implementation can use:

- OpenRouter server-side web tools,
- custom search API,
- custom fetch implementation,
- another search provider.

Keep the rest of WebScout independent from the specific provider.

---

# 9. Web research skill

Initial skill:

```text
skills/
└── web-research/
    └── SKILL.md
```

Example:

```markdown
---
name: web-research
description: >
  Use when researching factual questions using web sources.
---

# Research methodology

## 1. Understand the question

Identify:
- factual claims needed,
- date sensitivity,
- comparison dimensions,
- likely authoritative sources.

## 2. Source priority

Prefer:

1. official documentation
2. standards / specifications
3. government sources
4. academic papers
5. primary company sources

Use secondary sources for:

- interpretation,
- comparisons,
- community views.

## 3. Verify important claims

For important or contested claims:

- check multiple sources,
- prefer primary evidence,
- record uncertainty.

## 4. Track evidence

For every important finding capture:

- claim
- source
- publication/update date when relevant
- confidence

## 5. Handle conflicts

If credible sources disagree:

- report the disagreement,
- do not silently choose one,
- explain which source is stronger and why.

## 6. Citation integrity

Never invent a URL, citation or source.
```

---

# 10. Verifier node

The verifier is not a full Deep Agent.

Use a bounded LangChain model call with structured output.

Schema:

```python
from pydantic import BaseModel


class VerificationResult(BaseModel):
    sufficient: bool
    missing_information: list[str]
    weak_claims: list[str]
    contradictory_claims: list[str]
```

Input:

```text
Original question
+
Research findings
+
Sources
```

Example output:

```json
{
  "sufficient": false,
  "missing_information": [
    "Need primary evidence for Temporal replay behavior"
  ],
  "weak_claims": [
    "Scaling comparison relies only on a secondary blog"
  ],
  "contradictory_claims": []
}
```

---

# 11. Verification policy

The verifier checks:

```text
Did research answer every part of the question?

Are important claims supported?

Are citations tied to actual sources?

Are source dates appropriate?

Are current claims based on current information?

Are there contradictory claims?

Are primary sources available?

Is uncertainty clearly represented?
```

If not sufficient:

```text
verify
 ↓
research again
```

---

# 12. Research loop limit

Never allow unlimited research.

Example:

```python
MAX_ITERATIONS = 3
```

Routing:

```text
if sufficient:
    answer

elif iteration < MAX_ITERATIONS:
    research

else:
    answer_with_uncertainty
```

This protects:

- latency,
- cost,
- runaway tool calls.

---

# 13. Answer node

The final answer node should be bounded.

Input:

```text
Question
+
Verified findings
+
Sources
```

Requirements:

```text
answer directly
cite important claims
distinguish fact from inference
state uncertainty
do not add unsupported claims
```

Do not make the answer node perform new research.

If new research is needed, routing should return to the research node.

---

# 14. Model roles

Use different models by role when useful.

Example configuration:

```yaml
models:

  researcher:
    provider: openrouter
    model: <research-capable-model>

  verifier:
    provider: openrouter
    model: <independent-reasoning-model>

  answer:
    provider: openrouter
    model: <fast-answer-model>
```

Suggested philosophy:

```text
Researcher
= strong tool use + research

Verifier
= independent reasoning

Answer
= fast + good writing
```

---

# 15. OpenRouter model factory

Keep provider configuration centralized.

Concept:

```python
import os
from langchain_openai import ChatOpenAI


def get_model(
    model: str,
    temperature: float = 0,
):
    return ChatOpenAI(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )
```

Do not instantiate provider configuration throughout the application.

---

# 16. LangGraph routing

Conceptual graph:

```python
START
  ↓
research
  ↓
verify
  ↓
route_after_verify
```

Routing:

```python
def route_after_verify(state):
    if state["sufficient"]:
        return "answer"

    if state["iteration"] >= state["max_iterations"]:
        return "answer"

    return "research"
```

Then:

```text
answer
 ↓
END
```

---

# 17. Minimal graph

WebScout V1 should have only:

```text
research
verify
answer
```

Do not add:

```text
planner agent
query agent
source agent
citation agent
memory agent
search agent
fetch agent
summary agent
```

unless evidence later shows they are needed.

---

# 18. Why no separate planner initially

Deep Agents already have internal planning behavior.

For a simple research project:

```text
Question
 ↓
Research Agent
```

is enough.

A dedicated planner becomes useful only if:

- research tasks become large,
- user approves plan before search,
- multiple specialized research workers are used,
- explicit decomposition needs to be inspectable.

Not needed in V1.

---

# 19. Why no separate search agent

Search is a tool, not necessarily an agent.

Bad:

```text
Research Agent
   ↓
Search Agent
   ↓
Search Tool
```

Better:

```text
Research Agent
   ↓
Search Tool
```

Only create another agent if it requires independent:

- context,
- model,
- permissions,
- objective,
- lifecycle.

---

# 20. Why no A2A

All components are inside one application.

```text
LangGraph
 ├── Research Agent
 ├── Verifier
 └── Answer
```

A2A is unnecessary.

Rule:

> **Use LangGraph internally. Add A2A only at independent service boundaries.**

---

# 21. Memory strategy

Do not add long-term memory in V1.

Web facts become stale quickly.

The source of truth is the current web.

Possible future memory:

```text
user preferences
previous research topics
trusted domains
preferred answer style
saved research sessions
```

Do not treat remembered web facts as authoritative.

Rule:

> **Current web evidence beats memory.**

---

# 22. Skills vs memory

For WebScout:

```text
Skill
= how to research well

Memory
= what the user/project previously learned or preferred
```

Example:

```text
web-research skill:
"Prefer primary sources."

memory:
"User prefers official docs over blog posts."
```

---

# 23. LangSmith tracing

Enable tracing early.

Useful trace:

```text
Question
│
├── Research #1
│   ├── model
│   ├── search
│   ├── search
│   ├── fetch
│   └── fetch
│
├── Verify #1
│   └── model
│
├── Research #2
│   ├── search
│   └── fetch
│
├── Verify #2
│   └── model
│
└── Answer
    └── model
```

This makes it easy to find:

- poor search queries,
- unnecessary searches,
- weak source choices,
- verifier loops,
- expensive runs,
- latency bottlenecks.

---

# 24. Evaluation dataset

Create a small benchmark.

Example categories:

```text
current technology
historical fact
product comparison
technical documentation
scientific question
multi-source synthesis
conflicting evidence
recent news
```

Start with 20–30 questions.

Later expand to 100+.

---

# 25. Evaluation metrics

## Answer correctness

Does the answer actually answer the question correctly?

---

## Citation correctness

Does each citation support the claim?

---

## Citation completeness

Are important factual claims cited?

---

## Source quality

Did the agent prefer authoritative sources?

---

## Freshness

For current questions, are sources current enough?

---

## Research efficiency

Measure:

```text
number of searches
number of fetches
number of research iterations
tokens
latency
cost
```

---

# 26. Example experiment

Compare:

```text
Agent A
1 research iteration

Agent B
up to 3 research iterations + verifier
```

Measure:

```text
                     A        B

Correctness          78%      89%
Citation quality     74%      93%
Avg searches         3.2      6.1
Latency              8s       17s
Cost                 $X       $Y
```

This lets LangSmith show whether the verification loop is actually worth the extra cost.

---

# 27. ACP integration

Start with CLI.

Example:

```text
$ webscout

> What is the difference between ACP and A2A?

Researching...
Verifying evidence...

Answer:
...
```

Then add ACP:

```text
Zed
 │
ACP
 │
WebScout
```

Possible streamed status:

```text
✓ Question understood
✓ Found official ACP source
✓ Found official A2A source
◉ Cross-checking claims
○ Writing answer
```

---

# 28. Project structure

Recommended:

```text
webscout/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── models.py
│   ├── agent.py
│   │
│   ├── state.py
│   ├── graph.py
│   │
│   ├── nodes/
│   │   ├── research.py
│   │   ├── verify.py
│   │   └── answer.py
│   │
│   ├── tools/
│   │   ├── search.py
│   │   └── fetch.py
│   │
│   └── schemas.py
│
├── skills/
│   └── web-research/
│       └── SKILL.md
│
├── evals/
│   ├── dataset.json
│   └── evaluators.py
│
├── tests/
│
├── pyproject.toml
├── .env.example
└── README.md
```

---

# 29. V0

Goal:

```text
Question
 ↓
Deep Agent
 ↓
Web tools
 ↓
Answer
```

Build only:

- OpenRouter model,
- Research Deep Agent,
- search/fetch,
- CLI.

No LangGraph quality loop yet.

---

# 30. V1

Add LangGraph:

```text
research
 ↓
verify
 ↓
answer
```

And:

```text
verify insufficient
 ↓
research again
```

Maximum 3 iterations.

---

# 31. V2

Add the `web-research` skill.

Measure whether it improves:

- source quality,
- citation quality,
- correctness.

---

# 32. V3

Add LangSmith tracing and evals.

Dataset:

```text
20–30 research questions
```

Track:

```text
quality
cost
latency
search count
```

---

# 33. V4

Add ACP integration.

```text
Zed
 ↓
WebScout
```

Stream research progress into the client.

---

# 34. V5

Optional improvements:

```text
session persistence
user preferences
research history
trusted-source profiles
domain-specific skills
parallel research subagents
```

Do not add these until V1–V4 are reliable.

---

# 35. Possible future domain skills

Later:

```text
skills/
├── web-research/
├── technical-research/
├── scientific-research/
├── news-research/
├── product-comparison/
└── source-verification/
```

Each skill should remain focused.

---

# 36. Parallel research — future only

For harder questions:

```text
                 Research Lead
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Primary    Academic    Community
       Sources     Sources      Sources
          └──────────┼──────────┘
                     ▼
                  Synthesis
```

Deep Agent subagents can handle this later.

Do not start here.

---

# 37. Reliability rules

## Rule 1

Never invent citations.

## Rule 2

For current claims, use current sources.

## Rule 3

Prefer primary sources.

## Rule 4

Important claims should have strong evidence.

## Rule 5

Conflicting credible sources should be surfaced.

## Rule 6

Research loops must have a hard limit.

## Rule 7

The answer node does not perform hidden new research.

## Rule 8

Memory is never stronger than current web evidence.

---

# 38. What WebScout actually builds

Do not rebuild:

| Component | Build? |
|---|---:|
| Tool-calling loop | No |
| Agent harness | No |
| Context management | No |
| Basic planning | No |
| Skill loading mechanism | No |
| LangGraph runtime | No |
| OpenRouter API gateway | No |
| LangSmith tracing platform | No |
| ACP protocol | No |

Build:

| Component | Build? |
|---|---:|
| Research workflow | Yes |
| Research skill | Yes |
| Search/fetch adapters | Yes |
| Verification schema | Yes |
| Verification policy | Yes |
| Source handling | Yes |
| Answer policy | Yes |
| Evaluation dataset | Yes |
| ACP entrypoint/config | Yes |

---

# 39. Mental model

```text
Deep Agents
= research harness

LangGraph
= research control plane

LangChain
= structured LLM primitives

Skill
= research methodology

OpenRouter
= model gateway

LangSmith
= quality + observability

ACP
= interaction layer
```

---

# 40. Final V1 architecture

```text
                 USER
                  │
                  ▼
              CLI / ACP
                  │
                  ▼
              LangGraph
                  │
                  ▼
          Research Deep Agent
                  │
           web-research skill
                  │
          ┌───────┴───────┐
          ▼               ▼
       Search            Fetch
          │               │
          └───────┬───────┘
                  ▼
                 WEB
                  │
                  ▼
              Verifier
             LangChain
                  │
             sufficient?
             /         \
           no           yes
           │             │
           └─ Research   ▼
                      Answer
                         │
                         ▼
                       USER

                  OpenRouter
                      ↑
               model gateway

                  LangSmith
                      ↑
             trace + evaluation
```

---

# 41. Recommended first milestone

Build exactly this:

```text
question
 ↓
research Deep Agent
 ↓
web search/fetch
 ↓
verification
 ↓
if needed research one more time
 ↓
answer with sources
```

Definition of done:

1. User can ask a factual web question.
2. Agent searches at least one authoritative source.
3. Agent can fetch and inspect source content.
4. Verifier identifies missing evidence.
5. Graph can loop back to research.
6. Maximum research iterations are enforced.
7. Final answer contains source references.
8. All model calls use OpenRouter.
9. Run is visible in LangSmith.

Do not add memory, A2A, multiple specialist agents or a complex UI until this loop is reliable.

---

# 42. Project philosophy

WebScout is not intended to be a large agent platform.

It is a small project for learning the correct boundaries between:

```text
agent harness
workflow
skills
models
observability
client integration
```

The project succeeds if it demonstrates one reliable loop:

> **Research → Verify → Research if necessary → Answer.**

That loop is small enough to understand end-to-end, but rich enough to teach the architecture needed for larger systems such as RepoSmith.
