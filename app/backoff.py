import sys
import time

from langchain_openai.chat_models.base import OpenAIRateLimitError


def _should_retry(exc: BaseException, retry_on) -> bool:
    # A bare exception class is callable, so it would otherwise be mistaken for a
    # predicate — and `bool(SomeError(exc))` is truthy, silently retrying everything.
    if isinstance(retry_on, type) and issubclass(retry_on, BaseException):
        retry_on = (retry_on,)
    if isinstance(retry_on, tuple):
        return isinstance(exc, retry_on)
    return bool(retry_on(exc))


def call_with_backoff(
    fn,
    *args,
    attempts: int = 5,
    base_delay: float = 20.0,
    retry_on=(OpenAIRateLimitError,),
    **kwargs,
):
    """Linear backoff. `retry_on` is a tuple of exception types or a predicate.

    The default keeps the original LLM behavior: retry provider rate limits only,
    20s * attempt. Callers with a different failure profile (the web_search HTTP
    call) pass their own predicate and a much shorter base_delay.
    """
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _should_retry(exc, retry_on) or attempt == attempts - 1:
                raise
            wait = base_delay * (attempt + 1)
            print(
                f"[warn] retrying after {type(exc).__name__}, "
                f"attempt {attempt + 1}/{attempts - 1} in {wait:.0f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
