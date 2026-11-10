"""Shopper feedback (thumbs up/down) joined to exactly what the shopper saw.

Every answered /ask records an *interaction* under its request id: the query, how it was
interpreted, the products shown, the answer, its source and the model. A rating sent
later with that request id is stored together with the interaction, so each piece of
feedback can be analysed against its query, retrieval and answer.

Storage is Redis (Upstash in the live demo, which persists): interactions expire after
FEEDBACK_RETENTION_DAYS; ratings go to a capped list (newest first) that
deploy/export_feedback.py reads. Recording an interaction is best effort and can't slow
or fail an answer; storing a rating reports failures, because the caller is waiting.
"""
import asyncio
import json
import time
from collections.abc import Callable

from loguru import logger

from config.settings import settings

INTERACTION_KEY = "rag:interaction:{}"
RATED_KEY = "rag:feedback:rated:{}"
FEEDBACK_LIST = "rag:feedback"


class UnknownRequest(Exception):
    """No recorded answer for this request id (never answered here, or expired)."""


class AlreadyRated(Exception):
    pass


class FeedbackUnavailable(Exception):
    pass


class FeedbackStore:
    def __init__(self, client, op_timeout: float = 0.25, write_timeout: float = 2.0,
                 available: Callable[[], bool] = lambda: True) -> None:
        self.client = client
        self.op_timeout, self.write_timeout = op_timeout, write_timeout
        # The cache's circuit breaker: while it has Redis marked down, recording is skipped,
        # so an outage doesn't add a timed-out write to every answer.
        self.available = available

    @property
    def ttl_seconds(self) -> int:
        return settings.FEEDBACK_RETENTION_DAYS * 86400

    async def record_interaction(self, request_id: str | None, record: dict) -> None:
        if not request_id or not self.available():
            return
        try:
            await asyncio.wait_for(
                self.client.set(INTERACTION_KEY.format(request_id), json.dumps(record), ex=self.ttl_seconds),
                timeout=self.op_timeout,
            )
        except Exception as e:  # best effort: the answer matters more than its record
            logger.warning(f"Could not record interaction: {type(e).__name__}")

    async def add(self, request_id: str, rating: str, comment: str | None) -> None:
        try:
            raw = await asyncio.wait_for(self.client.get(INTERACTION_KEY.format(request_id)),
                                         timeout=self.write_timeout)
            if raw is None:
                raise UnknownRequest(request_id)
            # NX: one rating per answer; a second click can't stuff the numbers.
            first = await asyncio.wait_for(
                self.client.set(RATED_KEY.format(request_id), rating, ex=self.ttl_seconds, nx=True),
                timeout=self.write_timeout,
            )
            if not first:
                raise AlreadyRated(request_id)
            entry = {"request_id": request_id, "rating": rating, "comment": comment,
                     "rated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "interaction": json.loads(raw)}
            await asyncio.wait_for(self.client.lpush(FEEDBACK_LIST, json.dumps(entry)), timeout=self.write_timeout)
            await asyncio.wait_for(self.client.ltrim(FEEDBACK_LIST, 0, settings.FEEDBACK_MAX_ITEMS - 1),
                                   timeout=self.write_timeout)
        except (UnknownRequest, AlreadyRated):
            raise
        except Exception as e:
            raise FeedbackUnavailable(type(e).__name__) from e
