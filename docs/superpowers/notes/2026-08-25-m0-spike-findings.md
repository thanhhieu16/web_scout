# M0 Spike findings

Ngày chạy: 2026-08-26 · Model: stealth/ox-alpha qua OpenRouter · Chi phí ~$0.03 cho 4 probe

## R1 - inject server tool

- Phương án A (extra_body={"tools":[spec]} trên ChatOpenAI): LOẠI từ review Task 8 — extra_body.tools ghi đè top-level tools của bind_tools trên wire.
- Phương án B' (ResearchChatOpenAI: `_get_request_payload` extend tools + `use_responses_api=False`): **CHỐT NHẬN**. Spike xác nhận:
  - Không set `use_responses_api=False` → langchain-openai 1.6 auto-route sang Responses API khi payload có tool type khác `function` (`_use_responses_api`: uses_builtin_tools) → crash `responses.create(messages=...)`.
  - Set False → đi chat/completions, OpenRouter thực thi server tool, model search đúng lúc.
- Quyết định: **Method B' đã trong code** (app/models.py, app/tools/search.py).

## R2 - vị trí url_citation annotations trong AIMessage

- langchain-openai 1.6 DROP field `annotations` tùy chỉnh của OpenRouter (response_metadata chỉ có token_usage...).
- Fix: override `_create_chat_result` trong ResearchChatOpenAI — lift `choices[0].message.annotations` vào `additional_kwargs["annotations"]`, normalize SDK objects qua `.model_dump()`.
- Spike xác nhận end-to-end: AK ANNOTATIONS chứa url_citation đầy đủ (url/title/content) ✓
- Lưu ý count: field thật là `usage.server_tool_use_details.web_search_requests` (docs nói `server_tool_use`) — `count_web_searches` đã đọc cả hai key + fallback `token_usage`.

## R3 - tool-calling chất lượng ox-alpha

- Model gọi search đúng lúc trong mọi probe; trả lời grounded kèm citations.
- Go/No-go: **GO**

## Hệ quả kế thừa

- Spike findings đã code hóa: models.py (+use_responses_api=False, +_create_chat_result lift), tools/search.py (+details key, +token_usage fallback), tests (5 mới).
- Batch acceptance phía sau chạy trên nền đã verify wire.
