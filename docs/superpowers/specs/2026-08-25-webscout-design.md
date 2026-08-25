# WebScout — Design Spec (V0 → V3)

**Ngày:** 2026-08-25
**Trạng thái:** Đã duyệt qua brainstorming, chờ review
**Nguồn ý tưởng:** `webscout_idea.md` (gốc tiếng Anh)
**Môi trường dev:** Windows, PowerShell, Python ≥ 3.12, uv

---

## 1. Tóm tắt

WebScout là research agent nhỏ: nhận câu hỏi → search web → đọc nguồn → kiểm tra bằng chứng đủ chưa → trả lời kèm citations. Mục tiêu của dự án là **học ranh giới đúng giữa các thành phần**, không phải xây nền tảng agent lớn:

```text
Deep Agents = research harness (agent loop)
LangGraph   = research control plane (product loop)
LangChain   = structured LLM primitives
Skill       = research methodology
OpenRouter  = model gateway
LangSmith   = observability + evals
```

## 2. Quyết định đã chốt với stakeholder

| Hạng mục | Quyết định |
|---|---|
| Phạm vi kế hoạch đầu tiên | V0 → V3 |
| Cơ chế search | Server tool `openrouter:web_search` (plugin `:online`/`web` đã bị OpenRouter đánh dấu deprecated) |
| Fetch trang | Tool `web_fetch` tự viết: httpx + trafilatura |
| Ngôn ngữ trả lời | Theo ngôn ngữ của câu hỏi |
| Models | `stealth/ox-alpha` cho mọi vai trò, toàn bộ qua OpenRouter (API key cấp sau) |
| Kiến trúc | Hướng A: Deep Agent là node bên trong graph LangGraph |

## 3. Nguyên tắc kiến trúc

1. **Deep Agents sở hữu agent loop** — không tự viết vòng `LLM → tool → observation → LLM`.
2. **LangGraph sở hữu product loop** — vòng `research → verify → answer`, routing, giới hạn số vòng.
3. **Chỉ 3 node**: `research`, `verify`, `answer`. Không planner/search-agent/citation-agent riêng.
4. **Verifier và Answer là bounded calls** — một model call có cấu trúc, không tool, không loop nội bộ.
5. **Không A2A, không memory dài hạn** trong phạm vi V0–V3 (ACP là V4, ngoài spec này).

## 4. Kiến trúc tổng thể

```text
USER → CLI
         ↓
     LangGraph graph
   ┌─────┴──────────────────────────┐
   │ research (node)                │
   │   = Deep Agent (ox-alpha)      │
   │   tools: openrouter:web_search │ ← server tool OpenRouter
   │          web_fetch             │ ← @tool tự viết (httpx+trafilatura)
   │   skills: web-research (V2)    │
   ├────────────────────────────────┤
   │ verify (node)                  │
   │   = ChatOpenAI(ox-alpha)       │
   │     .with_structured_output()  │
   ├────────────────────────────────┤
   │ answer (node)                  │
   │   = ChatOpenAI(ox-alpha)       │
   └────────────────────────────────┘

routing sau verify:
  sufficient                      → answer
  iteration < max_iterations      → research
  else                            → answer (kèm uncertainty)
```

Mọi model call đi qua OpenRouter (`https://openrouter.ai/api/v1`). Mọi run được trace lên LangSmith.

## 5. State và giao thức bàn giao

### 5.1 Schema state

Điểm mới so với ý tưởng gốc: thay trường tự do `research: str` bằng findings có cấu trúc, tạo "hợp đồng" giữa Deep Agent và phần còn lại.

```python
class Source(TypedDict):
    url: str
    title: str
    source_type: str        # "primary" | "secondary"
    excerpt: str            # đoạn trích dẫn chứng

class Finding(TypedDict):
    claim: str              # một phát hiện factual
    source_urls: list[str]  # tham chiếu vào sources
    confidence: str         # "high" | "medium" | "low"

class ResearchState(TypedDict):
    question: str
    findings: list[Finding]
    sources: list[Source]
    gaps: list[str]
    weak_claims: list[str]
    contradictory_claims: list[str]
    sufficient: bool
    iteration: int
    max_iterations: int     # mặc định 3, config qua env
    answer: str
    answer_language: str    # rỗng = theo ngôn ngữ câu hỏi
```

### 5.2 Giao thức bàn giao Deep Agent → state (research node)

1. Gọi deep agent với input: câu hỏi (+ gaps từ verifier nếu là vòng lặp sau).
2. System prompt **ép agent kết thúc bằng khối Findings có cấu trúc**: mỗi dòng một claim + tham chiếu `[S1]`, `[S2]...` + confidence.
3. Node parse khối Findings thành `list[Finding]`.
4. Đồng thời trích **annotations `url_citation`** từ response của OpenRouter để dựng `sources` — đây là danh sách nguồn chính thức. **URL do agent tự gõ trong Findings mà không khớp annotation nào sẽ bị đánh dấu vào `weak_claims`** (chống bịa citation ở mức cơ chế).

## 6. Các node

### 6.1 research

- Wrap `create_deep_agent()` từ thư viện `deepagents`.
- Model: researcher role; đính kèm server tool search (mục 7.1) + tool `web_fetch`.
- Input vòng đầu: chỉ câu hỏi. Vòng lặp sau: câu hỏi + findings hiện có + gaps ("chỉ nghiên cứu các điểm còn thiếu").
- Không micromanage các bước search/fetch bên trong.

### 6.2 verify

- Một call duy nhất: `question + findings + sources` → Pydantic:

```python
class VerificationResult(BaseModel):
    sufficient: bool
    missing_information: list[str]
    weak_claims: list[str]
    contradictory_claims: list[str]
```

- Kết quả merge vào state (`missing_information` → `gaps`). Temperature 0.
- Nguồn của `weak_claims`: research node đánh dấu claim có URL không khớp annotation (mục 5.2); verifier bổ sung các claim yếu về mặt ngữ nghĩa (ví dụ chỉ dựa vào một blog secondary). Cả hai cộng dồn vào cùng một trường.

### 6.3 answer

- Bounded call: `question + findings + sources` (+ cờ hết-vòng nếu chưa sufficient).
- Viết đáp án theo ngôn ngữ câu hỏi, trích dẫn `[n]` trỏ vào `sources`, phân biệt fact/inference, nêu uncertainty khi chưa sufficient.
- **Không có tool** — nếu cần thêm bằng chứng thì routing phải quay về `research`.

## 7. Web tools

### 7.1 Search — server tool `openrouter:web_search`

Không viết code search. Đính kèm vào mọi model call của research agent qua `extra_body`:

```python
{
    "type": "openrouter:web_search",
    "parameters": {
        "max_results": 5,
        "max_uses": 4,           # tối đa 4 lần search / 1 request
        "max_characters": 4000,  # giới hạn nội dung mỗi kết quả
    },
}
```

- Engine `auto`: native search nếu model hỗ trợ, fallback Exa (~$0.007/lần).
- Lưu ý: deep agent chạy nhiều request trong một lần research (mỗi bước LLM là một request) → tổng search cộng dồn; chặn gián tiếp bằng `max_uses`/request + `recursion_limit` của graph.
- Số lần search đọc từ `usage.server_tool_use.web_search_requests` → log console + LangSmith.

### 7.2 Fetch — `@tool web_fetch(url)`

- `httpx`: timeout 15s, user-agent riêng.
- `trafilatura` trích nội dung chính; thất bại thì fallback sang text thô.
- Trả tối đa ~20.000 ký tự.
- Lỗi (403/timeout/PDF/không phải HTML) → trả về thông báo lỗi dạng text để agent tự bỏ nguồn đó và thử nguồn khác.

## 8. Model factory và cấu hình

```python
ROLES = {"researcher", "verifier", "answer"}   # mặc định cùng một model

def get_model(role: str = "researcher") -> BaseChatModel:
    ...  # ChatOpenAI(base_url="https://openrouter.ai/api/v1", model=<từ config>, ...)
```

- `config.yaml` (pydantic-settings) + `.env`:
  - model từng role — mặc định đều `stealth/ox-alpha`
  - temperature: researcher 0.2, verifier 0, answer 0.3
  - `MAX_ITERATIONS=3`
  - tham số search (mục 7.1)
- Đổi model = sửa YAML, không đụng code.
- `.env.example` chứa `OPENROUTER_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.

## 9. Skill `web-research` (V2)

- `skills/web-research/SKILL.md` theo Agent Skills spec, load qua `create_deep_agent(skills=["skills/"])` + `FilesystemBackend`.
- Nội dung: phương pháp nghiên cứu từ ý tưởng mục 9 (ưu tiên official docs → standards → chính phủ → học thuật → công ty gốc; xử lý mâu thuẫn; track evidence; cấm bịa citation).
- **Phân công tránh trùng lặp**: format khối Findings nằm ở system prompt của research node (có từ V0); skill chỉ bổ sung *phương pháp*.
- V2 đo A/B bật/tắt skill trên eval dataset.

## 10. LangSmith và Evals

- **Tracing bật từ V0**: env `LANGSMITH_TRACING=true` — mọi call LangChain/LangGraph/Deep Agents trace phân tầng tự động.
- **V3 thêm evals**:
  - `evals/dataset.json`: 20–30 câu theo 8 nhóm của ý tưởng mục 24 (current tech, historical, comparison, documentation, science, multi-source synthesis, conflicting evidence, recent news).
  - `evals/run_evals.py`: chạy graph trên dataset, đẩy experiments lên LangSmith.
  - `evals/evaluators.py`: correctness + citation-support (LLM-as-judge); số liệu deterministic: số lần search, tokens, chi phí, latency.
  - Experiment tiêu chuẩn: skill on/off × iterations 1/3.

## 11. Cấu trúc dự án

```text
webscout/
├── app/
│   ├── main.py          # CLI: webscout "câu hỏi" (one-shot) + chế độ tương tác
│   ├── config.py        # pydantic-settings đọc config.yaml + .env
│   ├── models.py        # model factory
│   ├── agent.py         # create_deep_agent + system prompt + findings format
│   ├── state.py         # ResearchState
│   ├── graph.py         # build graph + route_after_verify
│   ├── schemas.py       # Source, Finding, VerificationResult
│   ├── nodes/{research,verify,answer}.py
│   └── tools/{search.py,fetch.py}
├── skills/web-research/SKILL.md
├── evals/{dataset.json,evaluators.py,run_evals.py}
├── tests/
├── pyproject.toml       # uv, python >= 3.12
├── .env.example
└── README.md
```

Phụ thuộc: `deepagents`, `langchain-openai`, `langgraph`, `trafilatura`, `httpx`, `pydantic-settings`, `langsmith`, `pytest` (dev). LangGraph đi kèm deepagents nhưng khai báo rõ ràng vì dùng trực tiếp.

## 12. Kiểm thử

| Lớp | Nội dung | Network |
|---|---|---|
| Unit | Parser khối Findings (pure function + fixture text), đối chiếu URL↔annotations, `route_after_verify`, config loading | Không |
| Unit | `web_fetch` trích HTML từ file fixture cục bộ | Không |
| Integration (`@pytest.mark.integration`) | 1 câu end-to-end: có sources, answer chứa citation marker | Cần key |
| Evals (V3) | Chất lượng trên dataset qua LangSmith | Cần key |

Integration test chạy thủ công trước khi đóng từng mốc.

## 13. Lộ trình và Definition of Done

| Mốc | Nội dung | DoD |
|---|---|---|
| **M0 spike** (~nửa ngày) | Script thỏa thuận xác minh R1, R2, R3 (mục 14) | Go/no-go; code mầm cho `models.py` + `tools/search.py`; script là throwaway |
| **V0** | Deep Agent + 2 web tools + CLI one-shot | Hỏi→đáp có citations; thấy trace LangSmith |
| **V1** | Graph LangGraph research→verify→answer, MAX_ITERATIONS, nhánh answer-with-uncertainty | Đủ 9 tiêu chí mục 41 ý tưởng gốc |
| **V2** | Skill `web-research` | A/B bật/tắt chạy được trên dataset nhỏ |
| **V3** | Eval dataset + evaluators + experiments | Báo cáo số liệu trên LangSmith |

## 14. Rủi ro và phương án

| # | Rủi ro | Phương án |
|---|---|---|
| R1 | Inject server tool `openrouter:web_search` chung mảng `tools` với function tool qua langchain-openai có thể xung đột merge | Spike M0 thử `extra_body`; fallback: wrapper ChatOpenAI thêm tool ở lớp request |
| R2 | Annotations `url_citation` chưa chắc nằm ở field nào của AIMessage (langchain-core) | Spike M0; fallback parse `additional_kwargs` / response metadata thô |
| R3 | ox-alpha qua OpenRouter: chất lượng tool-calling chưa kiểm chứng thực tế | Spike M0 smoke test; nếu kém, đổi model researcher qua config.yaml (không đụng code) |

## 15. Quy tắc độ tin cậy (thừa kế ý tưởng mục 37)

1. Không bao giờ bịa citation — nguồn chính thức lấy từ annotations, không tin URL agent tự gõ.
2. Claim hiện tại cần nguồn hiện tại.
3. Ưu tiên nguồn primary.
4. Claim quan trọng cần bằng chứng mạnh.
5. Nguồn đáng tin mâu thuẫn phải được nêu ra.
6. Vòng research có hard limit.
7. Answer node không research ngầm.
8. Memory không mạnh hơn bằng chứng web hiện tại (không áp dụng trong V0–V3 vì chưa có memory).

## 16. Ngoài phạm vi (V0–V3)

ACP integration (V4), session persistence, user preferences, research history, subagents song song, domain skills mở rộng, UI đồ họa, A2A.

## 17. Tiêu chí thành công của dự án

Một vòng lặp đáng tin cậy: **Research → Verify → Research if necessary → Answer**, hiểu được end-to-end qua trace LangSmith, đo được chất lượng qua evals — nền móng cho hệ thống lớn hơn (RepoSmith).
