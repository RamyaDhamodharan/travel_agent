"""
Multi-provider fallback chain: Groq -> OpenAI -> OpenRouter -> Cerebras.

Falls through to the next provider on rate limits, quota, auth errors,
timeouts, server errors, 400s (e.g. Groq structured-output failures), and
schema-validation failures (a provider returning too few/too many items for
a constrained field, e.g. fewer than the required meal options). Providers
with no API key configured are skipped. Any other exception is treated as a
real bug and raised immediately.
"""
import asyncio
import logging
import random
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Fail fast on 4xx (bad request / auth / not-found): retrying those just
# burns time before we fall through to the next provider anyway. But 5xx /
# timeouts ARE worth a couple of quick retries against the SAME provider
# first, since those are often transient (e.g. Gemini 504 DEADLINE_EXCEEDED).
TIMEOUT = 30
MAX_RETRIES = 0

# Transient-error retry (same provider) before moving on to the next one.
TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_BASE = 1.5  # seconds; exponential: 1.5, 3, 6...
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_TRANSIENT_WORDS = (
    "timeout", "timed out", "deadline_exceeded", "deadline exceeded",
    "service unavailable", "resource_exhausted", "rate limit", "ratelimit",
    "connection reset", "connection aborted",
)

# How many times to retry the SAME provider with a corrective follow-up
# message before moving on to the next provider, when the result parses
# fine but fails a caller-supplied post-hoc check (e.g. wrong item counts).
SAME_PROVIDER_REPAIR_ATTEMPTS = 1


def _is_timeout_type(e: Exception) -> bool:
    """Catches httpx.ReadTimeout / ConnectTimeout / PoolTimeout / WriteTimeout
    and asyncio.TimeoutError, all of which commonly carry an EMPTY message
    (e.g. `ReadTimeout('')`), so a text-based check on str(e) misses them
    entirely -- that's what let a bare ReadTimeout crash all the way up to
    a 500 instead of being retried/falling through."""
    exc_type = type(e).__name__.lower()
    return "timeout" in exc_type


def _is_transient(e: Exception) -> bool:
    if _is_timeout_type(e):
        return True
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    msg = str(e).lower()
    return any(w in msg for w in _TRANSIENT_WORDS)

_clients: dict[str, object] = {}
_structured: dict[tuple[str, str], object] = {}

# HTTP statuses where trying another provider makes sense.
_FALLTHROUGH_STATUS = {400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504}
_FALLTHROUGH_WORDS = (
    "rate limit", "ratelimit", "quota", "resource_exhausted",
    "timeout", "timed out", "service unavailable",
    "invalid_api_key", "authentication", "unauthorized", "permission",
    "api key not valid", "workspace", "invalid_request",
    "tool_use_failed", "tool choice", "did not call a tool",
    "model_not_found", "does not exist", "not_found", "not found", "404",
)


def _should_fall_through(e: Exception) -> bool:
    # A pydantic ValidationError means the model's response didn't satisfy
    # the schema constraints -- worth trying another provider.
    if isinstance(e, ValidationError):
        return True
    if _is_timeout_type(e):
        return True
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(status, int) and status in _FALLTHROUGH_STATUS:
        return True
    exc_type = type(e).__name__.lower()
    if any(w in exc_type for w in ("notfound", "ratelimit", "quota", "unauthorized", "badrequest")):
        return True
    msg = str(e).lower()
    return any(w in msg for w in _FALLTHROUGH_WORDS)


# Explicit output cap. The Itinerary schema is large (multi-day, 4 meals/day,
# 5 options/meal, each with name+distance+specialty) -- for a 3+ day trip
# that easily runs past most providers' *default* output token limit, which
# silently truncates the JSON mid-object. That truncation is what shows up
# as Groq's "Failed to parse tool call arguments as JSON" / tool_use_failed
# -- it is NOT transient, so retrying the same provider won't help; only a
# high enough max_tokens fixes it.
MAX_OUTPUT_TOKENS = 8192


def _get_groq():
    if "groq" not in _clients:
        from langchain_groq import ChatGroq
        _clients["groq"] = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    return _clients["groq"]


def _get_openai():
    if "openai" not in _clients:
        from langchain_openai import ChatOpenAI
        _clients["openai"] = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    return _clients["openai"]


def _get_openrouter():
    if "openrouter" not in _clients:
        from langchain_openai import ChatOpenAI
        _clients["openrouter"] = ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    return _clients["openrouter"]


def _get_cerebras():
    if "cerebras" not in _clients:
        from langchain_openai import ChatOpenAI
        _clients["cerebras"] = ChatOpenAI(
            model=settings.cerebras_model,
            api_key=settings.cerebras_api_key,
            base_url="https://api.cerebras.ai/v1",
            temperature=0,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    return _clients["cerebras"]


# (name, getter, settings attribute holding its API key)
# Gemini and Anthropic removed from the chain.
_PROVIDER_CHAIN = [
    ("groq", _get_groq, "groq_api_key"),
    ("openai", _get_openai, "openai_api_key"),
    ("openrouter", _get_openrouter, "openrouter_api_key"),
    ("cerebras", _get_cerebras, "cerebras_api_key"),
]


def _get_llm():
    return _PROVIDER_CHAIN[0][1]()


async def _invoke_with_transient_retry(structured_llm, messages, provider_name: str):
    """Call one provider, retrying a few times (with backoff) if the error
    looks transient (timeout / 5xx / rate limit), before giving up on this
    provider and letting the caller fall through to the next one."""
    attempt = 0
    while True:
        try:
            return await structured_llm.ainvoke(messages)
        except Exception as e:
            if attempt >= TRANSIENT_RETRIES or not _is_transient(e):
                raise
            wait = (TRANSIENT_BACKOFF_BASE ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "%s transient error (%s: %s), retry %d/%d in %.1fs",
                provider_name, type(e).__name__, str(e)[:150],
                attempt + 1, TRANSIENT_RETRIES, wait,
            )
            await asyncio.sleep(wait)
            attempt += 1


def _get_structured(name: str, getter, schema: Type[T]):
    key = (name, schema.__name__)
    if key not in _structured:
        _structured[key] = getter().with_structured_output(schema)
    return _structured[key]


async def call_structured(
    system_prompt: str,
    user_content: str,
    schema: Type[T],
    validate_fn: Optional[Callable[[T], Optional[str]]] = None,
) -> T:
    """Call the provider chain for a structured (pydantic-schema) response.

    validate_fn: optional extra check run on a successfully-parsed result,
    beyond what pydantic itself enforces. Return None if the result is
    acceptable, or a short string describing what's wrong if not (e.g.
    "Breakfast only has 2 options, need exactly 5"). A non-None result is
    treated like a recoverable failure: the same provider gets one
    corrective retry with that message appended to the prompt, then we
    move on to the next provider. If every provider's result fails the
    check, the LAST (best-effort) result is returned rather than raising,
    so the app degrades gracefully instead of erroring out.
    """
    last_error: Exception | None = None
    best_effort_result: Optional[T] = None
    best_effort_issue: Optional[str] = None

    for name, getter, key_attr in _PROVIDER_CHAIN:
        if not getattr(settings, key_attr, None):
            logger.info("Skipping %s: no API key configured", name)
            continue

        attempt_user_content = user_content
        for attempt in range(SAME_PROVIDER_REPAIR_ATTEMPTS + 1):
            try:
                structured_llm = _get_structured(name, getter, schema)
                result = await _invoke_with_transient_retry(
                    structured_llm,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": attempt_user_content},
                    ],
                    provider_name=name,
                )
            except Exception as e:
                last_error = e
                if _should_fall_through(e):
                    logger.warning("%s failed (%s: %s), trying next provider",
                                   name, type(e).__name__, str(e)[:200])
                    break  # move on to next provider
                raise

            if not validate_fn:
                return result

            issue = validate_fn(result)
            if issue is None:
                return result

            logger.warning("%s produced a result failing validation: %s", name, issue)
            best_effort_result = result
            best_effort_issue = issue
            if attempt < SAME_PROVIDER_REPAIR_ATTEMPTS:
                attempt_user_content = (
                    f"{user_content}\n\nYour previous response was rejected: {issue}. "
                    f"Fix this exactly and return a complete, corrected response."
                )
                continue
            # exhausted repair attempts on this provider, try next provider
            break

    if best_effort_result is not None:
        logger.error(
            "All providers exhausted with an unresolved validation issue (%s); "
            "returning best-effort result instead of failing the request.",
            best_effort_issue,
        )
        return best_effort_result

    raise last_error or RuntimeError("No LLM provider configured")