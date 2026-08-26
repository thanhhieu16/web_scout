# V3 Experiments — ma trận skill × iterations trên LangSmith

Ngày: 2026-08-26 · Dataset `webscout-evals-v1` (24 câu) · Mỗi arm chạy `--limit 3` (câu đầu dataset) · Judge = verifier role (ox-alpha)

## Experiments trên LangSmith (project `webscout`, dataset 1d457114)

| Arm | n | correctness ↑ | citation_support ↑ | num_sources |
|---|---|---|---|---|
| skill-on × iters3 | 3 | **0.93** | 0.67 | **5.0** |
| skill-off × iters3 | 3 | **1.00** | 0.50 | 4.3 |
| skill-on × iters1 | 3 | 0.87 | 0.50 | 4.3 |
| skill-off × iters1 | 3 | 0.60 | 0.83 | 2.7 |

Links (compare views nằm trong logs `evals/runs/arm*.log`, ví dụ experiment `skill-on-iters3-05d1604e`).

## Đọc số liệu

1. **Vòng verify đáng giá**: iters1 correctness thấp hơn rõ (0.87 / 0.60 so với 0.93 / 1.00) — đúng giả thuyết thiết kế của vòng Research→Verify.
2. **Skill ổn định hóa cận dưới**: với budget 1 vòng, skill giữ correctness ở 0.87 trong khi không-skill rơi xuống 0.60; skill cũng kéo num_sources lên (5.0 vs 4.3 ở iters3; 4.3 vs 2.7 ở iters1).
3. **skill-off × iters3 đạt 1.00 nhưng citation_support thấp nhất nhóm iters3 (0.50)** — answer đúng nhưng gắn nguồn yếu hơn; sample quá nhỏ để coi là vượt skill-on.
4. **citation_support thấp chung (0.50–0.83)**: judge chấm theo tiêu chí mọi [n] phải resolve + excerpt hỗ trợ — điểm cải thiện hàng đầu của sản phẩm.

## Finding kiến trúc quan trọng

**`search_calls = 0` ở CẢ 4 arm** — qua trace: ox-alpha trong deepagents loop *không bao giờ* gọi server tool `openrouter:web_search`; nó discovery hoàn toàn bằng `web_fetch` (kể cả fetch trang kết quả DuckDuckGo). Spike raw đã chứng minh server tool hoạt động khi model chịu gọi — vấn đề là xu hướng chọn tool của model này khi biết trước domain. Hệ quả: cơ chế sources-from-fetch (a04c02a) mới là đường cấp nguồn chính thức hiện tại; muốn bắt buộc dùng search server-side cần hard-prompt thêm hoặc đổi model researcher (chỉnh 1 dòng config.yaml).

## Giới hạn

n=3/arm là smoke của harness eval, không phải kết luận thống kê. Chạy lại với full dataset (`--limit 0`) khi cần quyết định dựa trên số liệu.
