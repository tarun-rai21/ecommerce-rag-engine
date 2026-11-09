import itertools
import json
import os

import requests
import streamlit as st
from components.diagnostics import render_diagnostics

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# Sent server-side only; the browser never sees it.
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def error_message(response: requests.Response) -> str:
    """The backend's error envelope message, with its request id for support."""
    try:
        err = response.json()["error"]
        return f"{err['message']} (request id {err.get('request_id')})"
    except Exception:
        return f"The backend returned HTTP {response.status_code}."


# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Search Engine",
    page_icon="🛍️",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────
st.title("🛍️ E-Commerce Semantic Search")
st.caption("Powered by Hybrid Dense + Sparse Retrieval · Reciprocal Rank Fusion · LLM Generation")
st.caption(
    "Catalog: Wayfair WANDS (~43k products). Prices are synthetic — WANDS publishes none — "
    "and exist only to demonstrate price filtering."
)
st.divider()

# ── Session State Init ────────────────────────────────────
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_source" not in st.session_state:
    st.session_state.last_source = None
if "last_timing" not in st.session_state:
    st.session_state.last_timing = {}
if "last_products" not in st.session_state:
    st.session_state.last_products = []
if "last_interpretation" not in st.session_state:
    st.session_state.last_interpretation = None
if "last_request_id" not in st.session_state:
    st.session_state.last_request_id = None


@st.cache_data(ttl=600, show_spinner=False)
def load_categories() -> list[str]:
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/facets", params={"key": "category"}, headers=HEADERS, timeout=5)
        r.raise_for_status()
        return [v["value"] for v in r.json()["values"]]
    except Exception:
        return []  # filters still work without the dropdown


def describe_filters(f: dict) -> str:
    parts = []
    if f.get("min_price") is not None and f.get("max_price") is not None:
        parts.append(f"${f['min_price']:,.0f}–${f['max_price']:,.0f}")
    elif f.get("max_price") is not None:
        parts.append(f"≤ ${f['max_price']:,.0f}")
    elif f.get("min_price") is not None:
        parts.append(f"≥ ${f['min_price']:,.0f}")
    if f.get("categories"):
        parts.append(", ".join(f["categories"]))
    if f.get("min_rating") is not None:
        parts.append(f"rating ≥ {f['min_rating']:g}★")
    return " · ".join(parts)


# ── Search Input ──────────────────────────────────────────
query = st.text_input(
    label="Ask about a product",
    placeholder="e.g. grey area rug under $150, mid-century coffee table between $200 and $400, desk lamp 4.5+ stars",
    max_chars=200,
)
st.caption("Price and rating limits written in the query (\"under $300\", \"4+ stars\") are applied as filters.")

with st.expander("Filters"):
    c1, c2, c3 = st.columns(3)
    min_price = c1.number_input("Min price ($)", min_value=0, value=None, step=25)
    max_price = c2.number_input("Max price ($)", min_value=0, value=None, step=25)
    min_rating = c3.selectbox("Min rating", [None, 3.0, 3.5, 4.0, 4.5],
                              format_func=lambda v: "Any" if v is None else f"{v:g}★ & up")
    categories = st.multiselect("Categories", load_categories())

search_clicked = st.button("Search", type="primary")

# ── Query Execution ───────────────────────────────────────
def _clear_result() -> None:
    st.session_state.last_answer = None
    st.session_state.last_source = None
    st.session_state.last_timing = {}
    st.session_state.last_products = []
    st.session_state.last_interpretation = None
    st.session_state.last_request_id = None


def stream_answer(body: dict, state: dict):
    """Yields answer text from /ask/stream as it arrives; stores the `meta`, `done` and
    `error` events in `state`. Connection: 15 s; each read: 90 s, because a free-tier
    backend that has slept takes about a minute to send its first byte."""
    with requests.post(f"{BACKEND_URL}/api/v1/ask/stream", json=body, headers=HEADERS,
                       stream=True, timeout=(15, 90)) as response:
        if not response.ok:
            raise RuntimeError(error_message(response))
        response.encoding = "utf-8"
        event = None
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
                if event == "token":
                    yield data["text"]
                else:
                    state[event] = data


def interpretation(meta: dict) -> str:
    applied = describe_filters(meta.get("applied_filters") or {})
    return (f"Searched for “{meta.get('interpreted_query', '')}”" + (f" · Filters: {applied}" if applied else "")
            + (f" · {meta['preferred_size']} sizes first" if meta.get("preferred_size") else ""))


if search_clicked and query.strip() and min_price is not None and max_price is not None and min_price > max_price:
    st.warning("Min price must not exceed max price.")
elif search_clicked and query.strip():
    _clear_result()  # never show a previous answer under a new query
    explicit = {k: v for k, v in {"min_price": min_price, "max_price": max_price,
                                  "min_rating": min_rating, "categories": categories}.items() if v not in (None, [])}
    body = {"q": query.strip(), "limit": 5, **({"filters": explicit} if explicit else {})}
    state: dict = {}
    try:
        with st.spinner("Searching... The first search after a quiet spell can take up to a minute "
                        "while the free-tier backend wakes up."):
            tokens = stream_answer(body, state)
            first = next(tokens, None)  # wait for the first sentence under the spinner
        st.subheader("💬 Answer")
        answer = st.write_stream(itertools.chain([first], tokens)) if first is not None else ""
        meta, done = state.get("meta", {}), state.get("done", {})
        st.session_state.last_answer = answer if isinstance(answer, str) else "".join(answer)
        st.session_state.last_source = done.get("source", meta.get("source"))
        st.session_state.last_timing = done.get("timing", {})
        st.session_state.last_products = meta.get("retrieved_products", [])
        st.session_state.last_interpretation = interpretation(meta)
        st.session_state.last_request_id = meta.get("request_id")
        if "error" in state:
            st.session_state.last_error = state["error"]["message"]
        st.rerun()  # redraw as a finished answer, with badges and diagnostics
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure the FastAPI server is running.")
    except requests.exceptions.Timeout:
        st.error("The request timed out. If the backend was asleep it should be awake now, "
                 "so please try again.")
    except RuntimeError as e:
        st.error(str(e))

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")

# ── Answer Display ────────────────────────────────────────
if st.session_state.get("last_error"):
    st.error(st.session_state.pop("last_error"))

if st.session_state.last_answer:
    st.subheader("💬 Answer")

    source_badge = {
        "cache": "⚡ Served from Cache",
        "llm": "🤖 Generated by LLM",
        "no_results": "❌ No Results Found",
        "llm_error": "⚠️ LLM unavailable — showing retrieved products only",
    }.get(st.session_state.last_source, "")

    st.caption(source_badge)
    if st.session_state.last_interpretation:
        st.caption(st.session_state.last_interpretation)

    # Markdown, so the model's tables and lists render. st.markdown never renders raw HTML
    # unless unsafe_allow_html=True, so a steered answer can't inject markup or scripts.
    with st.container(border=True):
        st.markdown(st.session_state.last_answer)

    # Thumbs up/down, joined server-side to this exact answer through its request id.
    request_id = st.session_state.last_request_id
    if request_id and st.session_state.last_source != "no_results":
        def send_feedback() -> None:
            choice = st.session_state.get(f"fb_{request_id}")
            if choice is None:
                return
            try:
                r = requests.post(f"{BACKEND_URL}/api/v1/feedback", headers=HEADERS, timeout=10,
                                  json={"request_id": request_id, "rating": "up" if choice == 1 else "down"})
                st.toast("Thanks for the feedback!" if r.ok else error_message(r))
            except requests.exceptions.RequestException:
                st.toast("Feedback couldn't be sent right now.")

        st.caption("Was this answer helpful?")
        st.feedback("thumbs", key=f"fb_{request_id}", on_change=send_feedback)

# ── Diagnostics Sidebar ───────────────────────────────────
if st.session_state.last_timing:
    render_diagnostics(
        timing=st.session_state.last_timing,
        retrieved_products=st.session_state.last_products,
    )
