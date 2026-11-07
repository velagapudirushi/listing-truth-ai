"""NLP marketing-claim extractor and superlative hype scorer for host descriptions."""

from __future__ import annotations

import html
import re
from typing import Any

import numpy as np
import pandas as pd

BRIGHT_KEYWORDS = (
    "sunlit",
    "sun-drenched",
    "sun drenched",
    "sunny",
    "bright",
    "natural light",
    "flooded with light",
    "airy",
    "skylight",
    "floor-to-ceiling",
)
SPACIOUS_KEYWORDS = (
    "spacious",
    "expansive",
    "huge",
    "massive",
    "oversized",
    "generous space",
    "palatial",
    "wide open",
    "open-concept",
    "high ceilings",
)
LUXURY_KEYWORDS = (
    "luxury",
    "luxurious",
    "designer",
    "newly renovated",
    "remodeled",
    "upscale",
    "high-end",
    "high end",
    "penthouse",
    "oasis",
    "boutique",
    "gourmet",
    "custom",
    "stunning",
    "breathtaking",
)
CLEAN_KEYWORDS = (
    "spotless",
    "immaculate",
    "sparkling clean",
    "pristine",
    "hotel-grade",
    "sanitized",
    "freshly painted",
    "mint condition",
)
SUPERLATIVE_BUZZWORDS = (
    "dream",
    "paradise",
    "unbeatable",
    "spectacular",
    "gorgeous",
    "exquisite",
    "amazing",
    "perfect",
    "ultimate",
    "gem",
    "magnificent",
    "lavish",
    "chic",
)


def strip_html(raw: str | None) -> str:
    """Strip HTML tags (`<br/>`, `<b>`) and normalize whitespace in InsideAirbnb text."""
    if not raw or not isinstance(raw, str):
        return ""
    unescaped = html.unescape(raw)
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", no_tags).strip()


def extract_text_claims(title: str | None, description: str | None) -> dict[str, Any]:
    """Extract structured boolean claim flags and a 0–100 marketing hype intensity score."""
    clean_title = strip_html(title)
    clean_desc = strip_html(description)
    combined = f"{clean_title}. {clean_desc}".strip()
    lower = combined.lower()

    words = re.findall(r"[a-zA-Z']+", lower)
    word_count = len(words)

    def _count_hits(lexicon: tuple[str, ...]) -> int:
        return sum(1 for kw in lexicon if kw in lower)

    bright_hits = _count_hits(BRIGHT_KEYWORDS)
    spacious_hits = _count_hits(SPACIOUS_KEYWORDS)
    luxury_hits = _count_hits(LUXURY_KEYWORDS)
    clean_hits = _count_hits(CLEAN_KEYWORDS)
    buzz_hits = _count_hits(SUPERLATIVE_BUZZWORDS)
    exclamations = combined.count("!")

    total_claim_hits = bright_hits + spacious_hits + luxury_hits + clean_hits + buzz_hits
    # Normalize hype density + absolute claim count onto a smooth 0–100 scale
    density_per_100w = (total_claim_hits / max(word_count, 25)) * 100.0
    raw_hype = (
        11.5 * min(total_claim_hits, 6)
        + 2.8 * min(density_per_100w, 12.0)
        + 2.2 * min(exclamations, 5)
    )
    text_hype_score = round(float(max(0.0, min(100.0, raw_hype))), 2)

    return {
        "clean_description": combined[:900],
        "word_count": word_count,
        "exclamation_count": exclamations,
        "claims_bright_light": int(bright_hits > 0),
        "claims_spacious": int(spacious_hits > 0),
        "claims_luxury_modern": int(luxury_hits > 0),
        "claims_spotless_clean": int(clean_hits > 0),
        "total_hype_terms": total_claim_hits,
        "text_hype_score": text_hype_score,
    }


def compute_catfish_and_gem_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leakage-free `catfish_gap_score` (NLP vs CLIP) and consumer `hidden_gem_score`.

    Anti-Leakage Design:
    - `catfish_gap_score` uses ONLY host text claims & CLIP photo scores (zero `price_usd`),
      making it 100% leakage-free as an input feature for fair-price regression.
    - `hidden_gem_score` combines CLIP visual quality with room-type price rank for the
      consumer deal-finder leaderboard, and is strictly excluded from model input features.
    """
    out = df.copy()
    hype_pct = out["text_hype_score"].rank(pct=True) * 100.0
    vis_pct = out["vis_composite_quality"].rank(pct=True) * 100.0

    bright_pct = out["vis_brightness"].rank(pct=True) * 100.0
    space_pct = out["vis_spaciousness"].rank(pct=True) * 100.0
    lux_pct = out["vis_luxury_finish"].rank(pct=True) * 100.0
    clean_pct = out["vis_cleanliness"].rank(pct=True) * 100.0

    broken_promises = (
        out["claims_bright_light"] * np.maximum(0.0, 50.0 - bright_pct) * 2.0
        + out["claims_spacious"] * np.maximum(0.0, 50.0 - space_pct) * 2.0
        + out["claims_luxury_modern"] * np.maximum(0.0, 50.0 - lux_pct) * 2.0
        + out["claims_spotless_clean"] * np.maximum(0.0, 50.0 - clean_pct) * 2.0
    ) / 4.0

    out["catfish_gap_score"] = np.round(
        np.clip(0.80 * (hype_pct - vis_pct) + 0.20 * broken_promises, -100.0, 100.0),
        2,
    )

    price_within_room_pct = out.groupby("room_type")["price_usd"].rank(pct=True) * 100.0
    out["hidden_gem_score"] = np.round(
        np.clip(0.65 * (vis_pct - price_within_room_pct) + 0.35 * (vis_pct - hype_pct), -100.0, 100.0),
        2,
    )
    return out
