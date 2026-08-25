from langchain_openai import ChatOpenAI

from app.config import RoleConfig, get_settings

ROLES = ("researcher", "verifier", "answer")


class ResearchChatOpenAI(ChatOpenAI):
    server_tools: list = []

    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)
        tools = list(payload.get("tools") or [])
        tools.extend(self.server_tools)
        payload["tools"] = tools
        return payload


def get_model(role: str = "researcher", settings=None):
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    s = settings or get_settings()
    cfg: RoleConfig = getattr(s, role)
    return ResearchChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        api_key=s.openrouter_api_key or "not-set",
        base_url=s.openrouter_base_url,
    )
