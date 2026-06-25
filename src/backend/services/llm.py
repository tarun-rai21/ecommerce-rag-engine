import asyncio
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from google import genai
from google.genai import types

from config.settings import settings

SYSTEM_PROMPT = """You are an expert e-commerce product auditor. Your single task is to answer the user's query using ONLY the verified product context provided below.

CRITICAL RULES:
1. Base every single fact, price, and feature directly on the provided context.
2. If the text context does not contain enough information to answer the query, state exactly: "I cannot find a matching product matching those exact parameters in our catalog." Do not speculate or invent models.
3. Never mention details or features not explicitly listed in the context."""


class LLMService:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = "models/gemini-2.0-flash-lite"
        logger.info("LLMService initialized with Gemini 2.0 Flash.")

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
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _call_gemini_sync(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        return response.text

    async def generate_answer(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        try:
            answer = await asyncio.to_thread(self._call_gemini_sync, prompt)
            logger.info(f"LLM response generated for query: '{query}'")
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed after retries: {e}")
            return "I was unable to process your request at this time. Please try again later."