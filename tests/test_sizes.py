"""Product sizes: parsing the catalog's messy values, search text, query form, LLM context."""
import pytest

from src.backend.api.v1.endpoints import build_context
from src.backend.services.llm import SYSTEM_PROMPT
from src.backend.services.query_parser import parse_query
from src.backend.services.search import size_first
from src.ingestion.loaders.wands import parse_dimensions, parse_rug_size, parse_sizes
from src.ingestion.schemas import build_search_text, size_token


@pytest.mark.parametrize("raw,expected", [
    ("rectangle 8 ' x 10'", "8' x 10'"),
    ("3 ' x 4 ' 6 ''", "3' x 4' 6\""),
    ("1'10 '' x 4'11 ''", "1' 10\" x 4' 11\""),
    ("4'-0 '' x 6'-0 ''", "4' x 6'"),
    ("30 '' x 46 ''", "2' 6\" x 3' 10\""),
    ("round 8'", "round 8'"),
    ("rectangle", None),
    ("1", None),
])
def test_parse_rug_size(raw, expected):
    assert parse_rug_size(raw) == expected


@pytest.mark.parametrize("size,token", [
    ("8' x 10'", "8x10"),
    ("7' 10\" x 10'", "8x10"),  # sold as an 8x10
    ("5' 3\" x 7' 6\"", "5x8"),
    ("round 8'", "round 8'"),
    ("queen", "queen"),
])
def test_size_token_rounds_to_nearest_foot(size, token):
    assert size_token(size) == token


def test_parse_sizes_keeps_every_size_once():
    raw = "rugsize : 8 ' x 10'|color : grey|rugsize : rectangle 8 ' x 10'|rugsize : 5 ' x 8'|mattresssize : Full / Double"
    assert parse_sizes(raw) == ["8' x 10'", "5' x 8'", "full"]
    assert parse_sizes(None) == [] and parse_sizes("color : grey") == []


def test_parse_sizes_splits_comma_separated_values():
    assert parse_sizes("rugsize : 2 ' x 3 ' , 5 ' x 7'") == ["2' x 3'", "5' x 7'"]


def test_parse_dimensions_shows_ranges_and_skips_bad_values():
    raw = "overallwidth-sidetoside:43.3|overallwidth-sidetoside:64.7|overallheight-toptobottom:39.6|overalldepth-fronttoback:n/a"
    assert parse_dimensions(raw) == {"width": "43.3-64.7", "height": "39.6"}


def test_sizes_stay_out_of_the_search_text():
    # In the index, a rare token like 8x10 outranked colour and style terms.
    assert "8x10" not in build_search_text({"title": "rug", "sizes": ["8' x 10'"]})


@pytest.mark.parametrize("query,text,size", [
    ("grey area rug 8 x 10", "grey area rug", "8x10"),
    ("rug 8' x 10'", "rug", "8x10"),
    ("rug 8ft by 10ft", "rug", "8x10"),
    ("8x10 rug", "rug", "8x10"),
    ("rug 10x8", "rug", "8x10"),
    ("8x10", "8x10", None),  # nothing else to search with
    ("black frame 22 x 36", "black frame 22 x 36", None),  # inches: stay in the text
    ("blk 18x18 seat cushions", "blk 18x18 seat cushions", None),
    ("side by side drawer", "side by side drawer", None),
])
def test_query_size_is_separated_from_text(query, text, size):
    parsed = parse_query(query)
    assert (parsed.text, parsed.size) == (text, size)


def test_size_query_keeps_price_filter():
    parsed = parse_query("rug 8 x 10 under $300")
    assert (parsed.text, parsed.size, parsed.filters.max_price) == ("rug", "8x10", 300)


def test_size_preference_moves_matches_first_and_keeps_order():
    a, b, c, d = ({"id": i, "sizes": s} for i, s in enumerate((["5' x 8'"], ["8' x 10'"], [], ["7' 10\" x 10'"])))
    assert size_first([a, b, c, d], "8x10", limit=3) == [b, d, a]
    assert size_first([a, c], "8x10", limit=2) == [a, c]  # nothing matches: order unchanged


def test_context_shows_sizes_dimensions_and_how_many_are_hidden():
    base = {"title": "t", "price": 1.0, "description": "d"}
    ctx = build_context([{**base, "sizes": ["8' x 10'", "queen"], "dimensions": {"width": "64.7", "height": "39.6"}}])
    assert "Available sizes: 8' x 10', queen" in ctx
    assert "Overall dimensions (inches): height 39.6, width 64.7" in ctx
    many = build_context([{**base, "sizes": [f"{i}' x {i + 1}'" for i in range(40)]}])
    assert "and 20 more" in many
    assert "Available sizes" not in build_context([base])


def test_prompt_allows_answering_when_only_a_detail_is_missing():
    assert "right kind of item" in SYSTEM_PROMPT and "not listed" in SYSTEM_PROMPT


def test_context_flags_sizes_matching_the_request():
    rug = {"title": "t", "price": 1.0, "description": "d", "sizes": ["5' x 8'", "7' 10\" x 10' 2\""]}
    assert "Matches the requested 8x10 size: 7' 10\" x 10' 2\"" in build_context([rug], prefer_size="8x10")
    assert "Matches the requested" not in build_context([rug], prefer_size="9x12")
    assert "Matches the requested" not in build_context([rug])
