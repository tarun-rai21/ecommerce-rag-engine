"""/ask/stream (Server-Sent Events) and the streaming answer cleaner."""
import json
import random

import pytest

from config.settings import settings
from src.backend.api.security import rate_limiter
from src.backend.services.llm import UNAVAILABLE_MESSAGE, LLMUnavailable, clean_output
from src.backend.services.stream_cleaner import StreamCleaner
from tests.fakes import FakeLLM, FakeRedis, FakeSearch


@pytest.fixture(autouse=True)
def open_api(monkeypatch):
    for name, value in (("API_KEYS", ""), ("RATE_LIMIT_PER_MINUTE", 0), ("ASK_RATE_LIMIT_PER_MINUTE", 0)):
        monkeypatch.setattr(settings, name, value)
    rate_limiter.reset()


class StreamingLLM(FakeLLM):
    """Streams `text` in small pieces; can fail before or part-way through."""
    def __init__(self, text="Here are two rugs. Both are grey.", fail_before=False, fail_after=None):
        super().__init__()
        self.text, self.fail_before, self.fail_after, self.streams = text, fail_before, fail_after, 0

    async def stream_answer(self, query, context, usage):
        self.streams += 1
        if self.fail_before:
            raise LLMUnavailable("rate limited")
        for i in range(0, len(self.text), 4):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("connection dropped")
            yield self.text[i:i + 4]
        usage.prompt_tokens, usage.completion_tokens = 900, 150


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def answer_of(evts) -> str:
    return "".join(d["text"] for e, d in evts if e == "token")


def test_stream_event_order_and_content(make_client):
    c = make_client(llm=StreamingLLM())
    r = c.post("/api/v1/ask/stream", json={"q": "grey rug under $300", "limit": 2}, headers={"X-Request-ID": "s-1"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evts = events(r)
    assert [e for e, _ in evts][0] == "meta" and evts[-1][0] == "done"
    meta, done = evts[0][1], evts[-1][1]
    assert meta["request_id"] == "s-1" and meta["source"] == "llm" and len(meta["retrieved_products"]) == 2
    assert meta["interpreted_query"] == "grey rug" and meta["applied_filters"]["max_price"] == 300
    assert answer_of(evts) == "Here are two rugs. Both are grey."
    assert done["source"] == "llm" and done["usage"] == {"prompt_tokens": 900, "completion_tokens": 150}
    assert done["time_to_first_token_ms"] is not None


def test_streamed_answer_is_cached_for_both_endpoints(make_client):
    llm = StreamingLLM()
    c = make_client(llm=llm, redis=FakeRedis())
    c.post("/api/v1/ask/stream", json={"q": "rug"})
    again = events(c.post("/api/v1/ask/stream", json={"q": "rug"}))
    assert again[0][1]["source"] == "cache" and answer_of(again) == "Here are two rugs. Both are grey."
    plain = c.post("/api/v1/ask", json={"q": "rug"}).json()
    assert plain["source"] == "cache" and plain["answer"] == "Here are two rugs. Both are grey."
    assert llm.streams == 1 and llm.calls == 0


def test_catalog_claim_never_reaches_the_client(make_client):
    text = "Here is a rug. This is the only rug in our catalog that matches. It is grey."
    evts = events(make_client(llm=StreamingLLM(text)).post("/api/v1/ask/stream", json={"q": "rug"}))
    assert answer_of(evts) == "Here is a rug. It is grey."
    assert all("only rug" not in d["text"] for e, d in evts if e == "token")


def test_llm_unavailable_streams_the_fallback(make_client):
    c = make_client(llm=StreamingLLM(fail_before=True), redis=FakeRedis())
    evts = events(c.post("/api/v1/ask/stream", json={"q": "rug"}))
    assert answer_of(evts) == UNAVAILABLE_MESSAGE and evts[-1][1]["source"] == "llm_error"
    assert events(c.post("/api/v1/ask/stream", json={"q": "rug"}))[0][1]["source"] == "llm"  # not cached


def test_failure_mid_stream_sends_error_event(make_client):
    evts = events(make_client(llm=StreamingLLM(fail_after=8)).post("/api/v1/ask/stream", json={"q": "rug"}))
    names = [e for e, _ in evts]
    assert "error" in names and names[-1] == "done" and evts[-1][1]["source"] == "llm_error"
    assert evts[names.index("error")][1]["code"] == "llm_failed"


def test_no_results_and_search_outage(make_client):
    evts = events(make_client(search=FakeSearch(n_results=0)).post("/api/v1/ask/stream", json={"q": "zzz"}))
    assert evts[0][1]["source"] == "no_results" and evts[-1][1]["source"] == "no_results"

    class Down(FakeSearch):
        async def search(self, *a, **kw):
            raise ConnectionError("qdrant down")
    r = make_client(search=Down()).post("/api/v1/ask/stream", json={"q": "rug"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "search_unavailable"  # before streaming


def test_stream_needs_key_and_shares_the_ask_rate_limit(make_client, monkeypatch):
    c = make_client(llm=StreamingLLM())
    monkeypatch.setattr(settings, "API_KEYS", "k")
    assert c.post("/api/v1/ask/stream", json={"q": "rug"}).status_code == 401
    monkeypatch.setattr(settings, "ASK_RATE_LIMIT_PER_MINUTE", 1)
    h = {"X-API-Key": "k"}
    assert c.post("/api/v1/ask", json={"q": "a"}, headers=h).status_code == 200
    assert c.post("/api/v1/ask/stream", json={"q": "b"}, headers=h).status_code == 429


# ── StreamCleaner ─────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Here are three rugs.\n\nThese are the shag rugs in your catalog that match the 8 x 10 size.",
    "**Brandt rug**\n- Size: 5' x 8'\n\nIt suits a living room. This rug is the only product in the catalog that matches.",
    "| Rug | Note |\n|---|---|\n| a | the only one in the catalog |",
    "<think>plan the answer</think>\nHere is a lamp. It costs $45.99.",
    "These are the only ones.",
    "Price: $97.99. Rating 4.5/5. Done.",
])
def test_cleaner_matches_clean_output_for_any_chunking(text):
    rng = random.Random(1)
    for _ in range(20):
        cleaner, i, shown = StreamCleaner(), 0, ""
        while i < len(text):
            n = rng.randint(1, 9)
            shown += cleaner.feed(text[i:i + n])
            i += n
        shown += cleaner.finish()
        assert shown.strip() == clean_output(text) == cleaner.text


def test_cleaner_holds_text_until_the_sentence_is_complete():
    c = StreamCleaner()
    assert c.feed("This is the only rug in our cat") == ""
    assert c.feed("alog. Here is ") == ""          # the claim is dropped once complete
    assert c.feed("another. And") == "Here is another. "


class TruncatingLLM(StreamingLLM):
    async def stream_answer(self, query, context, usage):
        async for piece in super().stream_answer(query, context, usage):
            yield piece
        usage.finish_reason = "length"


def test_answer_cut_off_at_max_tokens_is_shown_but_not_cached(make_client):
    llm = TruncatingLLM()
    c = make_client(llm=llm, redis=FakeRedis())
    first = events(c.post("/api/v1/ask/stream", json={"q": "rug"}))
    assert answer_of(first) == "Here are two rugs. Both are grey." and first[-1][1]["source"] == "llm"
    second = events(c.post("/api/v1/ask/stream", json={"q": "rug"}))
    assert second[0][1]["source"] == "llm" and llm.streams == 2  # generated again, not served from cache


def test_empty_stream_gets_the_fallback_message(make_client):
    c = make_client(llm=StreamingLLM(text=""), redis=FakeRedis())
    evts = events(c.post("/api/v1/ask/stream", json={"q": "rug"}))
    assert answer_of(evts) == UNAVAILABLE_MESSAGE and evts[-1][1]["source"] == "llm_error"


def test_cache_key_separates_parsing_on_and_off(make_client):
    # Same words, same (empty) filters, but parsing turns "8x10" into a size preference,
    # so retrieval differs and the two must not share an answer.
    llm = StreamingLLM()
    c = make_client(llm=llm, redis=FakeRedis())
    c.post("/api/v1/ask/stream", json={"q": "grey rug 8x10"})
    other = events(c.post("/api/v1/ask/stream", json={"q": "grey rug 8x10", "parse_filters": False}))
    assert other[0][1]["source"] == "llm" and llm.streams == 2
