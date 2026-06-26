import asyncio
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from groq import Groq, RateLimitError, APIConnectionError

from config.settings import settings

SYSTEM_PROMPT = """You are an expert e-commerce product auditor. Your single task is to answer the user's query using ONLY the verified product context provided below.

CRITICAL RULES:
1. Base every single fact, price, and feature directly on the provided context.
2. If the text context does not contain enough information to answer the query, state exactly: "I cannot find a matching product matching those exact parameters in our catalog." Do not speculate or invent models.
3. Never mention details or features not explicitly listed in the context."""


class LLMService:
    def __init__(self) -> None:
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model_name = "llama-3.1-8b-instant"
        logger.info("LLMService initialized with Groq llama-3.1-8b-instant.")

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
        )
        return response.choices[0].message.content

    async def generate_answer(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        try:
            answer = await asyncio.to_thread(self._call_groq_sync, prompt)
            logger.info(f"LLM response generated for query: '{query}'")
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed after retries: {e}")
            return "I was unable to process your request at this time. Please try again later."