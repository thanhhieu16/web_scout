import sys
import time

from langchain_openai.chat_models.base import OpenAIRateLimitError


def call_with_backoff(fn, *args, attempts: int = 5, base_delay: float = 20.0, **kwargs):
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except OpenAIRateLimitError:
            if attempt == attempts - 1:
                raise
            wait = base_delay * (attempt + 1)
            print(
                f"[warn] provider rate-limited, retry {attempt + 1}/{attempts - 1} "
                f"in {wait:.0f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
