from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.backoff import call_with_backoff
from app.config import ROLE_NAMES, RoleConfig, Settings, get_settings

ROLES = ROLE_NAMES


class ResearchChatOpenAI(ChatOpenAI):
    use_responses_api: bool = False
    # >1 wraps each call in call_with_backoff. Only the researcher sets it: its
    # calls run inside the agent loop, where an outer retry would replay the
    # whole run. Other roles keep call_with_backoff at their call sites.
    rate_limit_attempts: int = 1

    def _generate(self, *args, **kwargs):
        if self.rate_limit_attempts <= 1:
            return super()._generate(*args, **kwargs)
        return call_with_backoff(
            super()._generate, *args, attempts=self.rate_limit_attempts, **kwargs
        )

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
