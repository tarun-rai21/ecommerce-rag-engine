import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from groq import APIConnectionError, AsyncGroq, RateLimitError
from loguru import logger

from config.settings import settings

REFUSAL_MESSAGE = "I cannot find a matching product matching those exact parameters in our catalog."
UNAVAILABLE_MESSAGE = "I was unable to process your request at this time. Please try again later."

SYSTEM_PROMPT = f"""You are an expert e-commerce product auditor. Your single task is to answer the user's query using ONLY the verified product context provided below.

CRITICAL RULES:
1. Base every single fact, price, and feature directly on the provided context.
2. If none of the products is the kind of item the user asked for, state exactly: "{REFUSAL_MESSAGE}" A product that is only related to the request, such as storage or an accessory for the item, is not the kind of item asked for. Do not speculate or invent models. If the products are the right kind of item but a requested detail, such as a size or color, is not listed for one, you may still show it and say that detail is not listed. If a size is listed for a product, only say it comes in the sizes listed.
3. Never mention details or features not explicitly listed in the context. Describe a product only with attributes the context states; do not add qualities such as "comfortable", "durable" or "classic" unless the context says so.
4. The context holds only the few products retrieved for this query, not the whole catalog. Never claim they are the only, all, or best matching products, and do not end with a statement about how complete the list is.
5. Speak to the shopper directly. Never mention "the context", "the provided data" or how the products were retrieved."""


# Some reasoning models (e.g. Qwen3) can emit a <think>…</think> block before the answer.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


# Shared with src/evaluation/generation_eval.py so the eval scores exactly what
# production sends.
GENERATION_PARAMS = {"temperature": 0.2, "max_tokens": 512}


def answer_fingerprint() -> str:
    """Changes whenever the model, prompt or generation settings change; part of the answer
    cache key, so a new prompt never serves answers written under the old one."""
    blob = json.dumps([settings.LLM_MODEL, settings.LLM_REASONING_EFFORT, SYSTEM_PROMPT, GENERATION_PARAMS],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def build_messages(query: str, context: str) -> list[dict]:
    user = f"VERIFIED PRODUCT CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# Five retrieved products can't support a claim about the whole catalog ("these are the
# only ones in our catalog"). Rule 4 forbids it, but the model still wrote one in about
# 1 answer in 10 once rug sizes were added, so such sentences are removed in code.
CATALOG_CLAIM = re.compile(
    r"\b(?:the only (?:ones?|products?|items?|options?)\b"
    r"|only (?:ones?|products?|items?|options?) (?:in|available in|that)"
    r"|the only [\w\s/‑-]{0,60}?(?:in|available in) (?:our|the|your) catalog"
    r"|(?:these|this|they) (?:are|is) (?:all )?the [\w\s/‑-]{0,60}?(?:in|available in|listed in) (?:our|the|your) catalog"
    r"|all (?:of )?the [\w\s/‑-]{0,60}?(?:in|available in|listed in) (?:our|the|your) catalog)",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def drop_catalog_claims(text: str) -> str:
    """Remove sentences that claim to cover the whole catalog. Table rows are left
    alone, and an answer made only of such sentences is returned unchanged."""
    lines = []
    for line in text.split("\n"):
        if CATALOG_CLAIM.search(line) and not line.lstrip().startswith("|"):
            line = " ".join(s for s in _SENTENCE_END.split(line) if not CATALOG_CLAIM.search(s))
            if not line.strip():
                continue
        lines.append(line)
    kept = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return kept or text.strip()


def clean_output(text: str | None) -> str:
    return drop_catalog_claims(_THINK_BLOCK.sub("", text or "").strip())


@dataclass(frozen=True)
class LLMResult:
    answer: str
    ok: bool  # False when generation failed and `answer` is a fallback message
    # Billed tokens of the call (completion includes reasoning tokens); None if not reported.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # The model hit max_tokens (which includes gpt-oss reasoning): the answer may stop
    # mid-sentence. It is still shown, but never cached.
    truncated: bool = False


@dataclass
class StreamUsage:
    """Filled in by LLMService.stream_answer once the stream ends."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None  # "length" = cut off at max_tokens

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMUnavailable(Exception):
    """The LLM could not be reached (after retries) before any text was produced."""


ATTEMPTS = 3  # rate limits and dropped connections are retried; anything else is not


class LLMService:
    def __init__(self, async_client: AsyncGroq | None = None) -> None:
        # One async client for both endpoints: a blocking client would run in the default
        # thread pool, which query embedding also uses, so slow LLM calls (up to a minute
        # with retries) could hold up searches. max_retries=0: the loop in _open is the
        # single retry layer.
        self.async_client = async_client or AsyncGroq(
            api_key=settings.GROQ_API_KEY,
            max_retries=0,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        self.model_name = settings.LLM_MODEL
        self.extra_params: dict = (
            {"reasoning_effort": settings.LLM_REASONING_EFFORT} if settings.LLM_REASONING_EFFORT else {}
        )
        logger.info(f"LLMService initialized with Groq {self.model_name}.")

    async def _open(self, messages: list[dict], **kwargs):
        """chat.completions.create with retries (2 s, 4 s) on rate limits and dropped connections."""
        for attempt in range(1, ATTEMPTS + 1):
            try:
                return await self.async_client.chat.completions.create(
                    model=self.model_name, messages=messages, **GENERATION_PARAMS, **self.extra_params, **kwargs,
                )
            except (RateLimitError, APIConnectionError):
                if attempt == ATTEMPTS:
                    raise
                await asyncio.sleep(min(2 ** attempt, 16))

    async def generate_answer(self, query: str, context: str) -> LLMResult:
        try:
            response = await self._open(build_messages(query, context))
            choice = response.choices[0]
            content = clean_output(choice.message.content)
            if not content:
                raise ValueError("LLM returned an empty response")
            truncated = getattr(choice, "finish_reason", None) == "length"
            if truncated:
                logger.warning(f"LLM answer cut off at max_tokens for query: '{query}'")
            usage = getattr(response, "usage", None)
            logger.info(f"LLM response generated for query: '{query}'")
            return LLMResult(answer=content, ok=True, truncated=truncated,
                             prompt_tokens=getattr(usage, "prompt_tokens", None),
                             completion_tokens=getattr(usage, "completion_tokens", None))
        except Exception as e:
            logger.error(f"LLM generation failed after retries: {e!r}")
            return LLMResult(answer=UNAVAILABLE_MESSAGE, ok=False)

    async def stream_answer(self, query: str, context: str, usage: StreamUsage) -> AsyncIterator[str]:
        """Yields raw text deltas as the model writes them; fills `usage` at the end.

        Opening the stream is retried like generate_answer (rate limits, dropped
        connections); once text has been produced, a failure is raised to the caller,
        which has already shown part of the answer. Raises LLMUnavailable if no stream
        could be opened. If the caller stops early (client disconnected), the upstream
        request is closed instead of running to completion."""
        try:
            stream = await self._open(build_messages(query, context), stream=True)
        except Exception as e:
            raise LLMUnavailable(repr(e)) from e
        try:
            async for chunk in stream:
                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    if getattr(chunk.choices[0], "finish_reason", None):
                        usage.finish_reason = chunk.choices[0].finish_reason
                reported = chunk.usage or getattr(getattr(chunk, "x_groq", None), "usage", None)
                if reported:
                    usage.prompt_tokens, usage.completion_tokens = reported.prompt_tokens, reported.completion_tokens
        finally:
            close = getattr(stream, "close", None)
            if close:
                await close()
