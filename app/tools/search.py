from typing import Any

from app.config import SearchConfig


def build_search_spec(search: SearchConfig) -> dict:
    return {
        "type": "openrouter:web_search",
        "parameters": {
            "max_results": search.max_results,
            "max_uses": search.max_uses,
            "max_characters": search.max_characters,
        },
    }


def _usage_of(result: Any) -> dict:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result.get("usage") or {}
    meta = getattr(result, "response_metadata", None) or {}
    return meta.get("usage") or {}


def count_web_searches(result: Any) -> int:
    usage = _usage_of(result)
    stu = usage.get("server_tool_use") if isinstance(usage, dict) else None
    value = stu.get("web_search_requests", 0) if isinstance(stu, dict) else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def attach_server_tools(model, specs: list[dict]):
    return model.model_copy(update={"extra_body": {"tools": specs}})
