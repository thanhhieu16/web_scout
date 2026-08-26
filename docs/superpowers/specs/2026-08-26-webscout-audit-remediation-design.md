# WebScout — Audit Remediation Design

**Ngày:** 2026-08-26
**Trạng thái:** Đã duyệt qua brainstorming, chờ review
**Nguồn:** Kiểm toán toàn bộ `app/`, `tests/`, `evals/`, CI (26/08/2026) — 16 phát hiện
**Tiền đề:** [2026-08-25-webscout-design.md](2026-08-25-webscout-design.md) (V0–V3)

---

## 1. Tóm tắt

Kiểm toán codebase sau khi V3 hoàn tất tìm ra 16 vấn đề, chia thành sáu nhóm công việc. Spec này thiết kế cách xử lý **toàn bộ** sáu nhóm.

Ba vấn đề nghiêm trọng nhất đều là **im lặng** — không crash, không log, chỉ trả kết quả sai:

1. Vòng lặp research thứ hai **xoá sạch bằng chứng của vòng một** (`findings` và `sources` bị ghi đè, state không có reducer).
2. Dòng METRICS **báo cáo sai**: `est_cost` gần như luôn `$0.0000`, và toàn bộ token/chi phí của tool `web_search` không được đếm.
3. Mọi lời gọi không-tool (verify, answer, judge) gửi `"tools": []` lên wire — model hiện tại chấp nhận, model khác trả 400.

## 2. Quyết định đã chốt với stakeholder

| Hạng mục | Quyết định |
|---|---|
| Mục đích repo | **Tool dùng thật**, chạy thường xuyên, có khi chạy dài |
| Phạm vi | Cả sáu nhóm A–F |
| Breaking changes | **Được phép đổi thoải mái** — prompt, regex, `ResearchState`, chữ ký node |
| Baseline eval cũ | Bỏ; chạy lại sau khi A–E xong |
| Mô hình state | **Reducers của LangGraph** (không phải helper `merge_evidence` cộng tay) |
| Kênh usage của `web_search` | **`UsageCollector` truyền vào factory** (không phải nhét vào text rồi parse) |

## 3. Sáu nhóm công việc

| | Nhóm | Phát hiện | Bản chất |
|---|---|---|---|
| **A** | Evidence integrity | 1, 6, 7, 9 | Hợp đồng parse → state → answer |
| **B** | Usage accounting | 2, 3 | Kênh usage xuyên module |
| **C** | Portability | 4, 10 (+ `config.yaml` theo CWD) | Hỏng khi đổi model hoặc đổi thư mục chạy |
| **D** | Network trust boundary | 5, 8 | SSRF guard + retry search |
| **E** | Repo hygiene | 11–15 | Cơ học, không có quyết định thiết kế |
| **F** | Eval credibility | 16 | Đổi ý nghĩa của các con số |

## 4. Nhóm A — Evidence integrity

### 4.1 Vấn đề

[`app/nodes/research.py`](../../../app/nodes/research.py) trả `findings` và `sources` **của riêng lần chạy đó**. `ResearchState` là `TypedDict` trần, không reducer. Prompt vòng 2 lại ra lệnh `"research ONLY these points"`. Hệ quả dây chuyền:

```
vòng 1: findings = [f1, f2, f3]   sources = [s1, s2]
vòng 2: findings = [f4]           sources = [s3]        ← f1..f3, s1, s2 biến mất
answer: chỉ thấy f4 / s3, đánh số [1] cho s3
```

Ba lỗi phụ cùng vùng: dòng FINDINGS sai định dạng bị `continue` im lặng (#6); mỗi claim chỉ trích được **một** nguồn trong khi prompt yêu cầu đối chiếu hai nguồn (#7); `weak_claims` nhân bản qua các vòng (#9).

### 4.2 Thiết kế — `ResearchState` có reducer

```python
# app/state.py
import operator
from typing import Annotated, TypedDict

class ResearchState(TypedDict, total=False):
    question: str
    max_iterations: int
    answer_language: str

    findings: Annotated[list, merge_findings]
    sources: Annotated[list, merge_sources]
    weak_claims: Annotated[list, merge_weak_claims]

    iteration: Annotated[int, operator.add]
    search_calls: Annotated[int, operator.add]
    total_tokens: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]

    # verdict hiện tại — ghi đè là đúng, không reducer
    gaps: list
    contradictory_claims: list
    sufficient: bool
    answer: str
```

| Reducer | Quy tắc |
|---|---|
| `merge_sources` | Dedupe theo `url`, giữ thứ tự xuất hiện lần đầu → đánh số `[Sn]` **ổn định qua các vòng** |
| `merge_findings` | Key = `claim` đã normalize (casefold + strip khoảng trắng). Trùng → hợp `source_urls` (giữ thứ tự), `confidence` lấy **mức thấp hơn** |
| `merge_weak_claims` | Dedupe theo chuỗi nguyên văn, giữ thứ tự |

**Vì sao confidence lấy mức thấp hơn:** vòng 2 bị ra lệnh chỉ đào gaps, nên việc nó nhắc lại một claim cũ **không phải** xác nhận độc lập. Công cụ bằng chứng không được để phép lặp thổi phồng độ tin cậy.

**Vì sao `gaps` không có reducer:** đó là phán quyết hiện tại của verifier, không phải sổ tích luỹ. Tích luỹ `gaps` sẽ khiến vòng 3 đi đào lại thứ vòng 2 đã tìm ra.

### 4.3 Hệ quả lên các node

Cả ba node chỉ còn trả delta của chính nó. Mọi biểu thức `state.get(k, 0) + delta` bị xoá:

```python
# research — trước
"search_calls": state.get("search_calls", 0) + count_total_searches(messages),
# research — sau
"search_calls": count_total_searches(messages),
```

`verify` không còn phải tự gộp `weak_claims` bằng `dict.fromkeys`; nó trả `result.weak_claims` thô, reducer lo phần còn lại. Việc làm tròn `total_cost` chuyển ra một chỗ duy nhất ở [`app/main.py`](../../../app/main.py) — hết cộng dồn sai số làm tròn qua từng node.

`route_after_verify` không đổi: nó đọc `state["iteration"]` sau khi reducer đã chạy.

### 4.4 Hợp đồng parse mới

`parse_findings_block` trả `NamedTuple` thay vì tuple 3 phần tử:

```python
class FindingsParse(NamedTuple):
    findings: list[Finding]
    refs: list[list[str]]
    narrative: str
    dropped: list[str]     # dòng trong block không khớp regex
    block_found: bool      # có gặp "## FINDINGS" hay không
```

Research node xử lý hai tín hiệu mới:

- `dropped` không rỗng → mỗi dòng thành một `weak_claim` (`"unparseable FINDINGS line: ..."`) + cảnh báo stderr.
- `block_found is False` → agent bỏ qua hoàn toàn hợp đồng đầu ra. Thành `weak_claim` riêng + cảnh báo stderr. **Hôm nay trường hợp này im lặng tuyệt đối** — đúng chế độ hỏng nguy hiểm nhất của pipeline.

Đổi tuple 3 → NamedTuple 5 làm mọi chỗ unpack cũ ném `ValueError` ngay lập tức. Cố ý: hai call site ([`main.py`](../../../app/main.py), [`nodes/research.py`](../../../app/nodes/research.py)) phải được cập nhật, không có đường trôi qua âm thầm.

### 4.5 Multi-ref (#7)

```python
_LINE_RE = re.compile(
    r"^-\s*(?P<refs>(?:\[S\d+\][\s,]*)+)\s*(?P<claim>.+?)"
    r"\s*\|\s*confidence:\s*(?P<conf>high|medium|low)\s*$",
    re.IGNORECASE,
)
# refs = re.findall(r"S\d+", m.group("refs"))
```

`map_refs_to_urls` **không đổi** — nó vốn đã lặp qua list refs của mỗi finding, chỉ là chưa bao giờ nhận quá một phần tử.

`RESEARCH_SYSTEM_PROMPT` đổi kèm: ví dụ đầu ra thành `- [S1][S2] <claim> | confidence: high`, thêm câu "cite every source that supports the claim". Ba thứ này (**prompt, `_LINE_RE`, `map_refs_to_urls`**) là một khối, đã ghi trong [CLAUDE.md](../../../CLAUDE.md); commit phải đụng cả ba.

### 4.6 Guard hồi quy

Test cấp graph, hai vòng, fake agent + fake verifier, khẳng định findings và sources của vòng 1 vẫn còn khi answer node chạy. **Hiện chưa có test nào chạm tới đường này** — đó là lý do lỗi #1 sống sót qua V0–V3.

## 5. Nhóm B — Usage accounting

### 5.1 Vấn đề

`est_cost` gần như luôn bằng `$0.0000`: [`sum_usage`](../../../app/nodes/parsing.py) đọc `token_usage.cost`, nhưng OpenRouter chỉ trả `usage.cost` khi request mang `usage: {"include": true}` — không chỗ nào đặt cờ đó.

Đồng thời [`app/tools/search_tool.py`](../../../app/tools/search_tool.py) tự gọi `/chat/completions` riêng. Token và chi phí của lời gọi đó **không bao giờ** vào `total_tokens`/`total_cost`: chỉ có phần text quay về message stream. Mỗi lần search là một lời gọi LLM đầy đủ bị bỏ sổ.

### 5.2 Thiết kế — `app/usage.py`

```python
class UsageCollector:
    """Thu usage từ các lời gọi HTTP nằm ngoài message stream của LangChain."""
    def __init__(self) -> None: ...
    def add(self, tokens: int, cost: float, searches: int = 0) -> None: ...
    def drain(self) -> tuple[int, float, int]: ...   # trả về rồi reset
```

- **`drain()` reset là bắt buộc.** Research node đọc theo từng vòng; không reset thì vòng 2 đếm lại vòng 1.
- **`threading.Lock`** quanh `add`/`drain`: LangGraph có thể chạy nhiều tool call song song trong một turn.

### 5.3 Luồng nối dây

```
build_graph
  └─ UsageCollector()
       ├─→ build_research_agent(s, usage=collector)
       │     └─→ make_web_search(s, transport=None, usage=collector)
       └─→ make_research_node(agent, s, usage=collector)
             └─ drain() sau agent.invoke()
```

Thêm một tham số vào hai factory đã có sẵn seam `transport=` — đúng khuôn mẫu tiêm phụ thuộc hiện hành của repo.

`run_question` (đường agent-only trong `main.py`, không qua graph) tự tạo collector riêng.

### 5.4 Bỏ kênh stringly-typed

`count_total_searches` hiện đếm search bằng **hai** đường: `usage.server_tool_use*` của AIMessage, **và** regex `(\d+) search executed` bới trong text ToolMessage. Khi collector vào, nhánh regex bị xoá — nếu không sẽ đếm đôi.

Sau thay đổi:
- `count_total_searches` chỉ đọc usage của server tool (đường server-tool thuần, hiện chưa dùng nhưng giữ lại).
- Collector là nguồn duy nhất cho search count của tool phía client.
- Header `SEARCH_RESULTS (2 results, 1 search executed)` **giữ nguyên** — nó có ích cho model đọc; chỉ là không ai parse nó nữa.

### 5.5 Bật usage accounting (#2)

- [`get_model`](../../../app/models.py): `model_kwargs={"usage": {"include": True}}`
- Body của `web_search` tool: thêm `"usage": {"include": True}`

Thêm một **integration test** khẳng định `cost` thật sự về trong `response_metadata` sau một lời gọi thật. Hiện chưa ai chứng minh được điều đó — nếu OpenRouter vốn đã trả `cost` mặc định thì cờ này vô hại, và test là thứ phân xử.

## 6. Nhóm C — Portability

### 6.1 `"tools": []` (#4)

Đã xác minh: model không tool nào → payload chứa `tools: []`. Backend tương thích OpenAI trả 400 cho mảng tools rỗng. Model hiện tại chịu được; README lại quảng cáo đổi model là sửa một dòng `config.yaml` — đó chính là quả mìn.

```python
def _get_request_payload(self, *args, **kwargs):
    payload = super()._get_request_payload(*args, **kwargs)
    tools = list(payload.get("tools") or []) + list(self.server_tools)
    if tools:
        payload["tools"] = tools
    else:
        payload.pop("tools", None)
    return payload
```

Test: model trần → `"tools" not in payload`.

### 6.2 Đường dẫn theo CWD (#10 + `config.yaml`)

`Path("skills").is_dir()` trong [`app/agent.py`](../../../app/agent.py) tính theo CWD → chạy ngoài repo root là skills tắt im lặng. Sửa bằng `_REPO_ROOT = Path(__file__).resolve().parent.parent`, dùng cho cả guard lẫn `root_dir` của `FilesystemBackend`.

Cùng loại bệnh, sẽ lộ ra ngay khi §8.2 làm `webscout` chạy được từ thư mục bất kỳ: `Settings` nạp `yaml_file="config.yaml"` theo CWD. Sửa thành **`./config.yaml` nếu có, không thì fallback về `config.yaml` ở repo root**. Thứ tự đó giữ nguyên [`test_config.py`](../../../tests/test_config.py) (vốn `chdir` sang `tmp_path` rồi viết `config.yaml` ở đó) và đồng thời làm console script dùng được.

## 7. Nhóm D — Network trust boundary

### 7.1 SSRF guard cho `web_fetch`

URL đưa vào `web_fetch` đến từ LLM và từ nội dung web. Hiện `follow_redirects=True`, không có gì chặn `127.0.0.1`, dải RFC1918, hay `169.254.169.254`. Chạy local thì vô hại; chạy trong CI hoặc trên máy chủ thì không.

1. Scheme phải thuộc `{http, https}` — kiểm tra tường minh. Hôm nay test `ftp://` chỉ pass **nhờ may mắn**: httpx tình cờ ném `UnsupportedProtocol`.
2. Resolve hostname; chặn nếu bất kỳ IP nào là private / loopback / link-local / reserved / multicast / unspecified.
3. `follow_redirects=False`, tự lặp tối đa `fetch.max_redirects` (mặc định 5), **chạy lại guard ở mỗi hop** — chặn redirect-vào-nội-bộ.
4. Config thêm `fetch.allow_private_hosts: bool = False` để mở khi dev cần.

**Seam test:** `make_web_fetch(cfg, transport=None, resolve=_default_resolve)` — resolver tiêm được. Bắt buộc: không có nó, `getaddrinfo("hostile.example")` sẽ phá vỡ cam kết "suite offline không chạm mạng". Đổi lại, guard test được trực tiếp: resolver trả `127.0.0.1` → kỳ vọng `FETCH_ERROR`.

**Ngoài phạm vi (ghi rõ):** DNS rebinding — cửa sổ TOCTOU giữa lúc resolve và lúc connect. Chặn được nó cần pin IP đã resolve rồi connect thẳng vào IP kèm override header `Host`. Guard này lo redirect-vào-nội-bộ và địa chỉ private trực tiếp, **không** lo rebinding.

### 7.2 Tổng quát hoá backoff (#8)

`call_with_backoff` hiện chỉ bắt `OpenAIRateLimitError` — exception của OpenAI SDK. `web_search` dùng httpx thuần nên không dùng lại được.

```python
def call_with_backoff(fn, *args, attempts=5, base_delay=20.0,
                      retry_on=(OpenAIRateLimitError,), **kwargs):
    """retry_on: tuple kiểu exception HOẶC predicate (exc) -> bool."""
```

Mặc định giữ y nguyên → hành vi của các lời gọi LLM không đổi, [`test_backoff.py`](../../../tests/test_backoff.py) đứng nguyên.

`web_search` truyền predicate (`TransportError`, hoặc `HTTPStatusError` với status 408/409/429/500/502/503/504) kèm **`attempts=3, base_delay=2.0`**. Nhịp 20s·n là dành cho LLM rate limit; áp lên search sẽ treo agent gần một phút vì một lỗi mạng thoáng qua.

## 8. Nhóm E — Repo hygiene

### 8.1 Dependency group (#11)

`uv run pytest` như README hướng dẫn **thất bại** khi dev extras chưa sync: uv rơi về pytest hệ thống, cho 17 `ImportError`. Đã dính đúng lỗi này trong lúc kiểm toán.

Chuyển `pytest` sang `[dependency-groups] dev` (PEP 735) — uv sync nhóm dev mặc định, `uv run pytest` chạy đúng ngay. Bỏ hẳn `[project.optional-dependencies]` để không có hai nguồn sự thật. Cập nhật README, CLAUDE.md, CI.

### 8.2 Build system (#12)

Thêm hatchling + `[tool.hatch.build.targets.wheel] packages = ["app"]`. Console script `webscout` chạy được; xoá hai đoạn workaround trong README và CLAUDE.md.

Kiểm tra kèm: với editable install, `_REPO_ROOT` tính từ `__file__` vẫn trỏ vào cây nguồn nên `skills/` và `config.yaml` vẫn resolve đúng (xem §6.2).

### 8.3 Lint (#13)

ruff vào nhóm dev. `[tool.ruff] line-length = 100`, `lint.select = ["E", "F", "I", "UP", "B"]`. Thêm step CI. `except Exception` ở [`app/models.py`](../../../app/models.py) là cố ý (lift annotations không được phép làm hỏng lời gọi) → `# noqa` kèm lý do.

### 8.4 CI matrix (#14)

`os: [ubuntu-latest, windows-latest]` × `python: ["3.12", "3.13"]`. Suite chạy 9 giây; bốn job vẫn rẻ. Venv dev hiện tại là **3.13 trên Windows** — tổ hợp CI chưa từng chạy lần nào.

### 8.5 Code chết (#15)

- Xoá `attach_server_tools` (không ai gọi).
- `map_refs_to_urls`: bỏ giá trị trả về thứ hai (luôn là `[]`).
- `reconcile_sources` → `find_unknown_refs(findings, citations) -> list[str]`. Giá trị trả về thứ nhất chưa từng được đọc ở bất kỳ đâu.
- `build_search_spec` **giữ lại** — `search_tool.py` đang dùng.

## 9. Nhóm F — Eval credibility

- Thêm role `judge` vào `config.yaml` và `Settings`. `correctness_evaluator` hiện chấm bằng chính model verifier — tự chấm bài mình.
- `citation_support_evaluator` chấm **mọi** ref có excerpt, score = tỉ lệ được support. Hiện một excerpt đúng là cả câu trả lời được 1.0.
- Chạy lại baseline sau khi A–E xong, ghi vào `evals/runs/`. Baseline [`skill-ab-v2.md`](../../../evals/runs/skill-ab-v2.md) cũ không còn so sánh được (prompt đổi ở §4.5).

## 10. Thứ tự thi hành

| # | Bước | Vì sao ở vị trí này |
|---|---|---|
| 1 | E-packaging (§8.1, §8.2, §8.3, §8.4) | Không đổi hành vi; gỡ chướng ngại chạy test trước khi đụng code |
| 2 | C (§6) | Nhỏ, độc lập, không phụ thuộc ai |
| 3 | A (§4) | Lớn nhất; đổi chữ ký return của node |
| 4 | B (§5) | Phụ thuộc chữ ký return mới của node |
| 5 | D (§7) | Độc lập, nhưng đổi seam test của `web_fetch` |
| 6 | E-deadcode (§8.5) | Sau cùng để không xung đột với A và B |
| 7 | F (§9) + chạy lại eval | Cần A–E đã ổn định mới đo có nghĩa |

## 11. Chiến lược kiểm thử

Suite offline **giữ nguyên cam kết không chạm mạng**.

| Vùng | Test mới |
|---|---|
| A | Graph 2 vòng: bằng chứng vòng 1 sống sót (guard hồi quy cho #1) |
| A | Reducer: `merge_sources` dedupe theo url; `merge_findings` lấy confidence thấp hơn; `merge_weak_claims` dedupe |
| A | Parse: multi-ref `[S1][S2]`; `dropped` bắt dòng sai định dạng; `block_found=False` khi thiếu `## FINDINGS` |
| B | `UsageCollector.drain()` reset; `web_search` ghi nhận token/cost/searches |
| B | Integration: `cost` xuất hiện trong `response_metadata` |
| C | Model trần → payload không có key `tools` |
| C | `skills_enabled=True` vẫn nạp được skill khi CWD ở nơi khác |
| D | Resolver tiêm trả `127.0.0.1` → `FETCH_ERROR`; redirect sang host private → `FETCH_ERROR` |
| D | Backoff: predicate `retry_on` retry đúng 429, không retry 404 |

Test cũ sẽ phải sửa: `test_node_verify.py` (verify hết tự gộp `weak_claims`), `test_fetch.py` (thêm resolver), các chỗ unpack `parse_findings_block`.

## 12. Cập nhật tài liệu

- **CLAUDE.md** — viết lại mục "State accumulation is manual" thành mô tả reducer; cập nhật mục hợp đồng FINDINGS (multi-ref); xoá workaround không-build-system; sửa lệnh test.
- **README.md** — xoá workaround console script; cập nhật lệnh test; thêm ghi chú về giới hạn an toàn của `web_fetch`.

## 13. Ngoài phạm vi

- DNS rebinding protection (§7.1).
- Đường server tool `openrouter:web_search` thuần — giữ nguyên, không kích hoạt lại.
- Bất kỳ tính năng mới nào. Spec này chỉ sửa những gì kiểm toán tìm ra.

## 14. Định nghĩa hoàn thành

1. Suite offline xanh trên cả bốn tổ hợp CI, vẫn không chạm mạng.
2. Có test cấp graph chứng minh bằng chứng tích luỹ qua các vòng.
3. Dòng METRICS báo `est_cost` khác 0 trong một lần chạy thật, và bao gồm chi phí của `web_search`.
4. `webscout "câu hỏi"` chạy được sau `uv sync`, từ thư mục bất kỳ.
5. `web_fetch` từ chối địa chỉ private, kể cả khi đến qua redirect.
6. `ruff check .` sạch.
7. Baseline eval mới ghi vào `evals/runs/`.
