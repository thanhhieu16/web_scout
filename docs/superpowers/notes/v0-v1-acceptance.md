# Acceptance notes — V0 & V1

Ngày: 2026-08-26 · Branch feature/webscout-v0-v3 · Model stealth/ox-alpha qua OpenRouter
Câu hỏi smoke chính: "ACP va A2A khac nhau the nao?" (chạy nhiều lần trong quá trình gỡ lỗi live)

## V0 — Definition of Done

| Tiêu chí | Bằng chứng |
|---|---|
| Hỏi→đáp có citations | `smoke_out.txt`: ANSWER với [n] + SOURCES list (12 nguồn) — EXIT=0 |
| Integration test | `uv run pytest -m integration` → `test_cli_real_roundtrip PASSED` (76s) |
| Trace LangSmith | LANGSMITH_TRACING=true + key present; project `webscout` |

Kết luận: **V0 PASS**

## V1 — 9 tiêu chí (spec mục 41 ý tưởng gốc)

| # | Tiêu chí | Bằng chứng |
|---|---|---|
| 1 | Hỏi được câu factual web | Nhiều run end-to-end EXIT=0 |
| 2 | ≥1 nguồn authoritative | SOURCES gồm agentcommunicationprotocol.dev, lfaidata.foundation, research.ibm.com, a2a-protocol.org |
| 3 | Fetch và đọc nội dung nguồn | diag_run.py log: web_fetch calls + ToolMessage nội dung trang |
| 4 | Verifier tìm thiếu bằng chứng | Run đầu: [verify] insufficient → research lại |
| 5 | Graph loop về research | Run đầu: research→verify→research→verify |
| 6 | MAX_ITERATIONS thi hành | Run 3-iteration: sau vòng 3 → answer_with_uncertainty ("Quá trình xác minh đã hết ngân sách...") |
| 7 | Answer có source refs | [1]..[7] trỏ đúng SOURCES; kết luận ACP merged into A2A (8/2025) có [3][7] |
| 8 | Mọi model call qua OpenRouter | base_url cố định https://openrouter.ai/api/v1 (models.py); spike usage metadata `provider_name: Stealth` |
| 9 | Run hiển thị LangSmith | Tracing bật từ .env; xem project webscout trên LangSmith UI |

Kết luận: **V1 PASS** (tiêu chí 9 cần xác nhận trực quan trên UI bởi stakeholder)

## Phát hiện live đã fix trong batch (spike-driven)

1. `use_responses_api=False` bắt buộc trên ResearchChatOpenAI — tool type khác function tự động route sang Responses API và crash (commit b6bda27)
2. Annotations url_citation bị langchain-openai drop → lift qua override `_create_chat_result` (b6bda27)
4. Field đếm search thật: `server_tool_use_details` (không phải `server_tool_use` như docs)
5. 429 pool stealth: thêm SDK retries (max_retries=4/timeout 180s) + application backoff (call_with_backoff) (1c2e9f4, 1a11d8a)
6. Sources rỗng khi agent fetch thay vì search → derive sources từ fetch thành công (a04c02a); lọc search-engine result pages (269217d)

## Còn lại

- Skill A/B live + experiments 4-arm: chạy tiếp trong batch này (xem v3-experiments.md)
