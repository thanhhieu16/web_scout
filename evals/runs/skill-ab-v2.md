# Skill A/B — V2 acceptance (mini, live)

Ngày: 2026-08-26

| Lượt | skills_enabled | Câu hỏi | Kết quả |
|---|---|---|---|
| ON | true (config gốc) | "ACP va A2A khac nhau the nao?" | EXIT=0; 1 iteration; verify sufficient; answer chi tiết có [n] citations; agent đọc SKILL.md ở bước 1 (diag_run.py log) |
| OFF | false | "WebSockets khac HTTP polling o diem nao?" | EXIT=0; 1 iteration; verify sufficient; answer + sources |

Nhận xét:
- Toggle hoạt động đúng cả 2 hướng; agent ON đọc `/skills/web-research/SKILL.md` như thiết kế progressive disclosure.
- Hai lượt dùng câu hỏi khác nhau nên chỉ chứng minh cơ chế, không so chất lượng định lượng.
- Đo định lượng: xem `v3-experiments.md` (ma trận skill × iterations trên LangSmith).
