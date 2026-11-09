"""Generates observability/grafana/dashboards/rag-engine.json.

The dashboard is code: edit the panel list here and re-run
    python observability/build_dashboard.py
rather than editing the JSON by hand or saving changes from the Grafana UI.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "grafana" / "dashboards" / "rag-engine.json"
# A variable, not a fixed uid: the same JSON imports into local Grafana (uid "prometheus")
# and into Grafana Cloud, whose Prometheus data source has a different uid.
DS = {"type": "prometheus", "uid": "${datasource}"}

# Groq on-demand prices for openai/gpt-oss-20b, USD per 1M tokens (see benchmarks/cost.py).
PRICE_IN, PRICE_OUT = 0.075, 0.30


def target(expr: str, legend: str = "", ref: str = "A") -> dict:
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}


def timeseries(title: str, targets: list[dict], unit: str, x: int, y: int, w: int = 12, h: int = 8,
               description: str = "") -> dict:
    return {
        "type": "timeseries", "title": title, "description": description, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 10}}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
        "targets": targets,
    }


def stat(title: str, expr: str, unit: str, x: int, y: int, w: int = 4, description: str = "",
         thresholds: list[tuple[str, float | None]] | None = None) -> dict:
    steps = [{"color": c, "value": v} for c, v in (thresholds or [("green", None)])]
    return {
        "type": "stat", "title": title, "description": description, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": 4},
        "fieldConfig": {"defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": steps}},
                        "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"},
        "targets": [target(expr)],
    }


def p(q: float, metric: str, by: str) -> str:
    return f"histogram_quantile({q}, sum by (le, {by}) (rate({metric}_bucket[5m])))"


panels = [
    # Row 1: headline numbers
    stat("Requests / s", "sum(rate(rag_requests_total[5m]))", "reqps", 0, 0),
    # "or vector(0)": with no 5xx at all the numerator has no series; show 0%, not "No data".
    stat("5xx error rate",
         '(sum(rate(rag_requests_total{status=~"5.."}[5m])) or vector(0)) / sum(rate(rag_requests_total[5m]))',
         "percentunit", 4, 0, thresholds=[("green", None), ("orange", 0.01), ("red", 0.05)]),
    stat("p95 /ask latency",
         'histogram_quantile(0.95, sum by (le) (rate(rag_request_duration_seconds_bucket{route="/api/v1/ask"}[5m])))',
         "s", 8, 0, thresholds=[("green", None), ("orange", 2), ("red", 5)]),
    stat("Cache hit ratio", 'sum(rate(rag_cache_lookups_total{result="hit"}[5m])) / sum(rate(rag_cache_lookups_total[5m]))',
         "percentunit", 12, 0),
    stat("Est. LLM cost / hour",
         f'(sum(rate(rag_llm_tokens_total{{type="prompt"}}[5m])) * {PRICE_IN} + '
         f'sum(rate(rag_llm_tokens_total{{type="completion"}}[5m])) * {PRICE_OUT}) / 1e6 * 3600',
         "currencyUSD", 16, 0, description="Token rate x Groq on-demand price for gpt-oss-20b"),
    stat("Cache breaker open", "max(rag_cache_breaker_open)", "none", 20, 0,
         description="1 while Redis is being skipped after failures",
         thresholds=[("green", None), ("red", 1)]),
    # Row 2: traffic and latency
    timeseries("Requests / s by route and status",
               [target("sum by (route, status) (rate(rag_requests_total[1m]))", "{{route}} {{status}}")],
               "reqps", 0, 4),
    timeseries("Request latency by route (p50, p95)",
               [target(p(0.5, "rag_request_duration_seconds", "route"), "p50 {{route}}", "A"),
                target(p(0.95, "rag_request_duration_seconds", "route"), "p95 {{route}}", "B")],
               "s", 12, 4),
    # Row 3: pipeline
    timeseries("Stage latency p95 (cache lookup, search, LLM, LLM first token)",
               [target(p(0.95, "rag_stage_duration_seconds", "stage"), "{{stage}}")], "s", 0, 12,
               description="Where /ask time goes; search includes query embedding"),
    timeseries("/ask answers by source",
               [target("sum by (source) (rate(rag_answers_total[5m]))", "{{source}}")], "reqps", 12, 12,
               description="llm_error = LLM failed or rate-limited; shoppers saw the fallback message"),
    # Row 4: cost and protection
    timeseries("LLM tokens / min",
               [target("sum by (type) (rate(rag_llm_tokens_total[5m])) * 60", "{{type}}")], "short", 0, 20),
    timeseries("Cache lookups / s by result",
               [target("sum by (result) (rate(rag_cache_lookups_total[5m]))", "{{result}}")], "reqps", 8, 20, w=8),
    timeseries("Shopper feedback (ratings / hour)",
               [target("sum by (rating) (increase(rag_feedback_total[1h]))", "{{rating}}")], "short", 0, 28,
               description="Thumbs up/down on answers; details via deploy/export_feedback.py"),
    timeseries("Refused requests / s",
               [target("sum by (scope) (rate(rag_rate_limited_total[5m]))", "rate limited: {{scope}}", "A"),
                target("sum(rate(rag_auth_failures_total[5m]))", "bad or missing API key", "B")],
               "reqps", 16, 20, w=8),
]
for i, panel in enumerate(panels, start=1):
    panel["id"] = i

dashboard = {
    "uid": "rag-engine",
    "title": "RAG engine",
    "description": "Traffic, latency by stage, cache, LLM tokens and cost, safeguards. Generated by observability/build_dashboard.py.",
    "tags": ["rag", "fastapi"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"},
    "templating": {"list": [{
        "name": "datasource", "label": "Data source", "type": "datasource", "query": "prometheus",
        "current": {}, "hide": 0, "refresh": 1, "includeAll": False, "multi": False,
    }]},
    "panels": panels,
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(panels)} panels)")
