import html
import os
import requests
import streamlit as st
from components.diagnostics import render_diagnostics

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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


@st.cache_data(ttl=600, show_spinner=False)
def load_categories() -> list[str]:
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/facets", params={"key": "category"}, timeout=5)
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


if search_clicked and query.strip() and min_price is not None and max_price is not None and min_price > max_price:
    st.warning("Min price must not exceed max price.")
elif search_clicked and query.strip():
    _clear_result()  # never show a previous answer under a new query
    explicit = {k: v for k, v in {"min_price": min_price, "max_price": max_price,
                                  "min_rating": min_rating, "categories": categories}.items() if v not in (None, [])}
    body = {"q": query.strip(), "limit": 5, **({"filters": explicit} if explicit else {})}
    with st.spinner("Searching..."):
        try:
            response = requests.post(f"{BACKEND_URL}/api/v1/ask", json=body, timeout=30)
            response.raise_for_status()
            data = response.json()

            st.session_state.last_answer = data.get("answer", "")
            st.session_state.last_source = data.get("source", "llm")
            st.session_state.last_timing = data.get("timing", {})
            st.session_state.last_products = data.get("retrieved_products", [])
            applied = describe_filters(data.get("applied_filters") or {})
            st.session_state.last_interpretation = (
                f"Searched for “{data.get('interpreted_query', query)}”" + (f" · Filters: {applied}" if applied else "")
                + (f" · {data['preferred_size']} sizes first" if data.get("preferred_size") else "")
            )

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure the FastAPI server is running.")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The LLM may be under load.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")

# ── Answer Display ────────────────────────────────────────
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

    # The answer is model output and can be steered by the query: escape it before
    # embedding in HTML so it can't inject markup or scripts.
    safe_answer = html.escape(st.session_state.last_answer).replace("\n", "<br>")

    st.markdown(
        f"""
        <div style="
            background-color: #1e1e2e;
            border-left: 4px solid #7c3aed;
            padding: 1.2rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            line-height: 1.7;
            color: #e2e8f0;
        ">
            {safe_answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Diagnostics Sidebar ───────────────────────────────────
if st.session_state.last_timing:
    render_diagnostics(
        timing=st.session_state.last_timing,
        retrieved_products=st.session_state.last_products,
    )