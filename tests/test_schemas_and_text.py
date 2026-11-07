"""Unit tests for Pandera multimodal contracts and NLP marketing hype extraction."""

import pandas as pd
import pytest

from listing_truth.schemas import MultimodalListingRecord, validate_multimodal_df
from listing_truth.text_claims import (
    compute_catfish_and_gem_scores,
    extract_text_claims,
    strip_html,
)


def test_html_strip_and_hype_extraction() -> None:
    raw_html = "<b>Stunning Sun-Drenched Luxury Penthouse!</b><br/>Spacious spotless oasis!"
    assert "<b>" not in strip_html(raw_html)
    claims = extract_text_claims("Sunlit Luxury Loft!", raw_html)
    assert claims["claims_bright_light"] == 1
    assert claims["claims_spacious"] == 1
    assert claims["claims_luxury_modern"] == 1
    assert claims["claims_spotless_clean"] == 1
    assert claims["text_hype_score"] > 40.0


def test_pandera_rejects_corrupt_negative_price() -> None:
    valid = {
        "listing_id": "1001",
        "name": "Cozy Mission Studio",
        "neighbourhood": "Mission",
        "room_type": "Entire home/apt",
        "accommodates": 2,
        "bedrooms": 1.0,
        "beds": 1.0,
        "bathrooms": 1.0,
        "price_usd": 165.0,
        "number_of_reviews": 12,
        "review_scores_accuracy": 4.85,
        "picture_url": "https://a0.muscache.com/im/pictures/test.jpg",
        "listing_url": "https://www.airbnb.com/rooms/1001",
        "clean_description": "Nice bright studio apartment.",
        "word_count": 5,
        "claims_bright_light": 1,
        "claims_spacious": 0,
        "claims_luxury_modern": 0,
        "claims_spotless_clean": 0,
        "text_hype_score": 20.0,
        "vis_brightness": 72.0,
        "vis_spaciousness": 60.0,
        "vis_luxury_finish": 65.0,
        "vis_cleanliness": 78.0,
        "vis_composite_quality": 68.4,
        "catfish_gap_score": -12.5,
        "hidden_gem_score": 18.2,
    }
    rec = MultimodalListingRecord(**valid)
    assert rec.price_usd == 165.0

    bad = dict(valid, price_usd=-25.0)
    with pytest.raises(Exception):
        validate_multimodal_df(pd.DataFrame([bad]))


def test_catfish_gap_score_has_zero_price_leakage() -> None:
    """Guarantee `catfish_gap_score` is invariant to `price_usd` (zero target leakage)."""
    base_rows = pd.DataFrame(
        [
            {
                "room_type": "Entire home/apt",
                "price_usd": 120.0,
                "text_hype_score": 85.0,
                "claims_bright_light": 1,
                "claims_spacious": 1,
                "claims_luxury_modern": 1,
                "claims_spotless_clean": 1,
                "vis_brightness": 22.0,
                "vis_spaciousness": 28.0,
                "vis_luxury_finish": 25.0,
                "vis_cleanliness": 30.0,
                "vis_composite_quality": 26.0,
            },
            {
                "room_type": "Entire home/apt",
                "price_usd": 450.0,
                "text_hype_score": 15.0,
                "claims_bright_light": 0,
                "claims_spacious": 0,
                "claims_luxury_modern": 0,
                "claims_spotless_clean": 0,
                "vis_brightness": 88.0,
                "vis_spaciousness": 82.0,
                "vis_luxury_finish": 85.0,
                "vis_cleanliness": 90.0,
                "vis_composite_quality": 86.0,
            },
        ]
    )
    scored_a = compute_catfish_and_gem_scores(base_rows)
    flipped_price = base_rows.copy()
    flipped_price["price_usd"] = [1500.0, 60.0]
    scored_b = compute_catfish_and_gem_scores(flipped_price)

    pd.testing.assert_series_equal(scored_a["catfish_gap_score"], scored_b["catfish_gap_score"])
    assert scored_a.iloc[0]["catfish_gap_score"] > scored_a.iloc[1]["catfish_gap_score"]
