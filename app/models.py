from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.config import RoleConfig, Settings, get_settings

ROLES = ("researcher", "verifier", "answer")


class ResearchChatOpenAI(ChatOpenAI):
    server_tools: list = []
    use_responses_api: bool = False

    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)
        tools = list(payload.get("tools") or []) + list(self.server_tools)
        if tools:
            payload["tools"] = tools
        else:
            payload.pop("tools", None)
        return payload

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        try:
            if isinstance(response, dict):
                message = (response.get("choices") or [{}])[0].get("message") or {}
                annotations = message.get("annotations")
            else:
                choices = getattr(response, "choices", None) or []
                annotations = getattr(choices[0].message, "annotations", None) if choices else None
        except Exception:
            annotations = None
        if annotations:
            normalized = []
            for ann in annotations:
                normalized.append(ann.model_dump() if hasattr(ann, "model_dump") else ann)
            gen_message = result.generations[0].message
            if isinstance(gen_message, AIMessage):
                existing = gen_message.additional_kwargs.get("annotations") or []
                gen_message.additional_kwargs["annotations"] = existing + normalized
        return result


def get_model(
    role: str = "researcher", settings: Settings | None = None
) -> ResearchChatOpenAI:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    s = settings or get_settings()
    cfg: RoleConfig = getattr(s, role)
    return ResearchChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        api_key=s.openrouter_api_key or "not-set",
        base_url=s.openrouter_base_url,
        max_retries=4,
        timeout=180,
        # OpenRouter only returns usage.cost when the request asks for accounting.
        # It must ride in extra_body: the OpenAI SDK's create() has a closed
        # signature and rejects a top-level `usage` keyword.
        extra_body={"usage": {"include": True}},
    )
