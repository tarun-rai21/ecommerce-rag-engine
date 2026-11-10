import pandas as pd
import pytest

from src.backend.services.query_parser import SearchFilters, parse_query
from tests.conftest import REPO_ROOT

WANDS_QUERIES = REPO_ROOT / "data" / "wands" / "query.csv"


@pytest.mark.parametrize("query, text, min_price, max_price, min_rating", [
    ("grey area rug under $300", "grey area rug", None, 300, None),
    ("grey area rug under 300", "grey area rug", None, 300, None),
    ("coffee table less than $1,200", "coffee table", None, 1200, None),
    ("desk below $99.99", "desk", None, 99.99, None),
    ("sofa between $500 and $900", "sofa", 500, 900, None),
    ("sofa between 900 and 500", "sofa", 500, 900, None),
    ("lamp $40-$80", "lamp", 40, 80, None),
    ("lamp $40 to $80", "lamp", 40, 80, None),
    ("rug $200 or less", "rug", None, 200, None),
    ("bed frame over $300", "bed frame", 300, None, None),
    ("bed frame $300+", "bed frame", 300, None, None),
    ("chair 50 dollars max", "chair", None, 50, None),
    ("ottoman priced under $150", "ottoman", None, 150, None),
    ("desk lamp 4+ stars", "desk lamp", None, None, 4.0),
    ("desk lamp at least 4.5 stars", "desk lamp", None, None, 4.5),
    ("desk lamp rated 4+", "desk lamp", None, None, 4.0),
    ("patio set under $800 with 4 stars and up", "patio set", None, 800, 4.0),
    ("dining table up to $900", "dining table", None, 900, None),
    ("lamp max $60", "lamp", None, 60, None),
    ("rug within $200", "rug", None, 200, None),
    ("desk lamp 4 stars", "desk lamp", None, None, 4.0),
    ("desk lamp 4 star and up", "desk lamp", None, None, 4.0),
])
def test_parses_constraints(query, text, min_price, max_price, min_rating):
    parsed = parse_query(query)
    assert parsed.text == text
    assert parsed.filters.min_price == min_price
    assert parsed.filters.max_price == max_price
    assert parsed.filters.min_rating == min_rating


@pytest.mark.parametrize("query", [
    "bar stool under 30 inches",       # unit → size, not price (and no "under 3" backtracking)
    "table under 30 in",
    "twin over full bunk beds",        # positional words
    "end table between recliners",
    "above toilet cabinet",
    "pictures to hang over fireplace",
    "star wars rug",                   # 'star' without a number
    "48 inch bathroom vanity",
    "5 drawer dresser",
    "led 60",
    "zachary 72.5",
    "over 200 thread count sheets",    # 'over' needs a currency marker
    "wisdom stone river 3-3/4",
    # Found in the audit: weak price words with a bare number are capacities, not budgets.
    "dining table seats up to 8",
    "table for up to 8",
    "bookcase within 5",
    "lamp max 60",
    "5 star hotel towels",             # a style, not a rating filter
    "3 star",
])
def test_no_false_positives(query):
    parsed = parse_query(query)
    assert parsed.filters.is_empty(), parsed
    assert parsed.text == query


@pytest.mark.skipif(not WANDS_QUERIES.exists(), reason="WANDS not downloaded")
def test_no_constraints_extracted_from_any_real_wands_query():
    # None of the 480 real queries states a price or rating; any extraction is a false positive.
    queries = pd.read_csv(WANDS_QUERIES, sep="\t")["query"]
    false_positives = [(q, parse_query(q).filters) for q in queries if not parse_query(q).filters.is_empty()]
    assert false_positives == []


def test_filters_validation_and_merge():
    with pytest.raises(ValueError):
        SearchFilters(min_price=100, max_price=50)
    parsed = SearchFilters(max_price=300, min_rating=4)
    explicit = SearchFilters(max_price=200, categories=["Rugs"])
    merged = explicit.merged_over(parsed)
    assert merged.max_price == 200 and merged.min_rating == 4 and merged.categories == ["Rugs"]
    # explicit max below a parsed min: the explicit value wins, the conflicting min is dropped
    assert SearchFilters(max_price=50).merged_over(SearchFilters(min_price=100)).min_price is None


def test_sizes_are_never_read_as_prices():
    parsed = parse_query("rug 8x10 under 3x5")
    assert parsed.filters.is_empty() and parsed.size == "8x10"
    assert parse_query("carpet 5x6").filters.is_empty()
