import asyncio
import re
from dataclasses import dataclass
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from groq import Groq, RateLimitError, APIConnectionError

from config.settings import settings

REFUSAL_MESSAGE = "I cannot find a matching product matching those exact parameters in our catalog."
UNAVAILABLE_MESSAGE = "I was unable to process your request at this time. Please try again later."

SYSTEM_PROMPT = f"""You are an expert e-commerce product auditor. Your single task is to answer the user's query using ONLY the verified product context provided below.

CRITICAL RULES:
1. Base every single fact, price, and feature directly on the provided context.
2. If the text context does not contain enough information to answer the query, state exactly: "{REFUSAL_MESSAGE}" Do not speculate or invent models.
3. Never mention details or features not explicitly listed in the context."""


# Some reasoning models (e.g. Qwen3) can emit a <think>…</think> block before the answer.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class LLMResult:
    answer: str
    ok: bool  # False when generation failed and `answer` is a fallback message


class LLMService:
    def __init__(self, client: Groq | None = None) -> None:
        # max_retries=0: tenacity below is the single retry layer (the SDK's own
        # retries would otherwise multiply with it).
        self.client = client or Groq(
            api_key=settings.GROQ_API_KEY,
            max_retries=0,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        self.model_name = settings.LLM_MODEL
        self.extra_params: dict = (
            {"reasoning_effort": settings.LLM_REASONING_EFFORT} if settings.LLM_REASONING_EFFORT else {}
        )
        logger.info(f"LLMService initialized with Groq {self.model_name}.")

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            f"VERIFIED PRODUCT CONTEXT:\n"
            f"{context}\n\n"
            f"USER QUESTION:\n"
            f"{query}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True,
    )
    def _call_groq_sync(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            **self.extra_params,
        )
        content = _THINK_BLOCK.sub("", response.choices[0].message.content or "")
        if not content or not content.strip():
            raise ValueError("LLM returned an empty response")
        return content.strip()

    async def generate_answer(self, query: str, context: str) -> LLMResult:
        prompt = self._build_prompt(query, context)
        try:
            answer = await asyncio.to_thread(self._call_groq_sync, prompt)
            logger.info(f"LLM response generated for query: '{query}'")
            return LLMResult(answer=answer, ok=True)
        except Exception as e:
            logger.error(f"LLM generation failed after retries: {e!r}")
            return LLMResult(answer=UNAVAILABLE_MESSAGE, ok=False)
