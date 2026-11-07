"""Unit tests for DuckDB SQL queries and zero-GPU CLIP cosine visual search."""

import pytest

from listing_truth.config import CLIP_INDEX_NPZ_PATH, MULTIMODAL_PARQUET_PATH
from listing_truth.search_engine import MultimodalListingSearch
from listing_truth.warehouse_sql import ListingTruthWarehouseSQL


def test_duckdb_warehouse_queries_and_sql_guardrail() -> None:
    sql = ListingTruthWarehouseSQL(MULTIMODAL_PARQUET_PATH)
    catfish = sql.get_top_catfish_listings(limit=5)
    gems = sql.get_top_hidden_gems(limit=5)
    nb = sql.get_neighborhood_audit()

    assert len(catfish) == 5
    assert len(gems) == 5
    assert len(nb) >= 5
    assert catfish.iloc[0]["catfish_gap_score"] >= catfish.iloc[-1]["catfish_gap_score"]

    with pytest.raises(ValueError, match="read-only"):
        sql.run_custom_query("DELETE FROM listings;")


def test_clip_visual_search_returns_ranked_photos() -> None:
    engine = MultimodalListingSearch(MULTIMODAL_PARQUET_PATH, CLIP_INDEX_NPZ_PATH)
    prompt = engine.query_prompts[0]
    res = engine.search_by_visual_prompt(prompt, max_price_usd=600.0, top_k=8)
    assert len(res) > 0
    assert (res["price_usd"] <= 600.0).all()
    assert res.iloc[0]["clip_similarity"] >= res.iloc[-1]["clip_similarity"]
