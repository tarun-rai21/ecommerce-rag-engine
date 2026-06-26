import streamlit as st


def render_diagnostics(timing: dict, retrieved_products: list[dict]) -> None:
    with st.sidebar:
        st.title("⚙️ Engine Diagnostics")

        # ── Timing Breakdown ──────────────────────────────
        st.subheader("⏱️ Latency Breakdown")

        source = st.session_state.get("last_source", "llm")

        if source == "cache":
            st.success("⚡ Cache Hit — No search or LLM call made.")
            st.metric("Cache Lookup", f"{timing.get('cache_lookup_ms', 0):.2f} ms")
            st.metric("Total", f"{timing.get('total_ms', 0):.2f} ms")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cache Lookup", f"{timing.get('cache_lookup_ms', 0):.2f} ms")
                st.metric("Search (RRF)", f"{timing.get('search_ms', 0):.2f} ms")
            with col2:
                st.metric("LLM Generation", f"{timing.get('llm_ms', 0):.2f} ms")
                st.metric("Total", f"{timing.get('total_ms', 0):.2f} ms")

            # Bar chart
            st.subheader("📊 Stage Latency Chart")
            chart_data = {
                "Stage": ["Cache Lookup", "Search (RRF)", "LLM Generation"],
                "Latency (ms)": [
                    timing.get("cache_lookup_ms", 0),
                    timing.get("search_ms", 0),
                    timing.get("llm_ms", 0),
                ],
            }
            import pandas as pd
            df_chart = pd.DataFrame(chart_data)
            st.bar_chart(df_chart.set_index("Stage"))

        st.divider()

        # ── RRF Retrieval Results ─────────────────────────
        st.subheader("🔍 Retrieved Products (RRF Ranked)")

        if not retrieved_products:
            st.info("Cache hit — retrieved products not re-fetched.")
        else:
            import pandas as pd
            rows = []
            for rank, product in enumerate(retrieved_products, start=1):
                rows.append({
                    "Rank": rank,
                    "Title": product.get("title", ""),
                    "Category": product.get("category", ""),
                    "Price ($)": product.get("price", 0.0),
                    "RRF Score": round(product.get("rrf_score", 0.0), 6),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)