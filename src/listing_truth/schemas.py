"""Pydantic and Pandera data validation gates for multimodal Airbnb listing records."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa


class MultimodalListingRecord(BaseModel):
    """Single validated real listing combining tabular specs, NLP claims, and CLIP vision scores."""

    listing_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=2)
    neighbourhood: str = Field(..., min_length=2)
    room_type: str = Field(..., min_length=2)
    accommodates: int = Field(..., ge=1, le=20)
    bedrooms: float = Field(..., ge=0.0, le=15.0)
    beds: float = Field(..., ge=0.0, le=25.0)
    bathrooms: float = Field(..., ge=0.0, le=15.0)
    price_usd: float = Field(..., gt=15.0, le=3000.0)
    number_of_reviews: int = Field(..., ge=0)
    review_scores_accuracy: float | None = Field(default=None)
    picture_url: str = Field(..., min_length=8)
    listing_url: str = Field(..., min_length=8)
    clean_description: str
    word_count: int = Field(..., ge=0)
    claims_bright_light: int = Field(..., ge=0, le=1)
    claims_spacious: int = Field(..., ge=0, le=1)
    claims_luxury_modern: int = Field(..., ge=0, le=1)
    claims_spotless_clean: int = Field(..., ge=0, le=1)
    text_hype_score: float = Field(..., ge=0.0, le=100.0)
    vis_brightness: float = Field(..., ge=0.0, le=100.0)
    vis_spaciousness: float = Field(..., ge=0.0, le=100.0)
    vis_luxury_finish: float = Field(..., ge=0.0, le=100.0)
    vis_cleanliness: float = Field(..., ge=0.0, le=100.0)
    vis_composite_quality: float = Field(..., ge=0.0, le=100.0)
    catfish_gap_score: float = Field(..., ge=-100.0, le=100.0)
    hidden_gem_score: float = Field(..., ge=-100.0, le=100.0)


LISTING_DF_SCHEMA = pa.DataFrameSchema(
    columns={
        "listing_id": pa.Column(str, nullable=False),
        "name": pa.Column(str, nullable=False),
        "neighbourhood": pa.Column(str, nullable=False),
        "room_type": pa.Column(str, nullable=False),
        "accommodates": pa.Column(int, [pa.Check.ge(1), pa.Check.le(20)], coerce=True),
        "bedrooms": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(15.0)], coerce=True),
        "beds": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(25.0)], coerce=True),
        "bathrooms": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(15.0)], coerce=True),
        "price_usd": pa.Column(float, [pa.Check.gt(15.0), pa.Check.le(3000.0)], coerce=True),
        "number_of_reviews": pa.Column(int, pa.Check.ge(0), coerce=True),
        "text_hype_score": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(100.0)], coerce=True),
        "vis_brightness": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(100.0)], coerce=True),
        "vis_spaciousness": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(100.0)], coerce=True),
        "vis_luxury_finish": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(100.0)], coerce=True),
        "vis_cleanliness": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(100.0)], coerce=True),
        "vis_composite_quality": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(100.0)], coerce=True),
        "catfish_gap_score": pa.Column(float, [pa.Check.ge(-100.0), pa.Check.le(100.0)], coerce=True),
        "hidden_gem_score": pa.Column(float, [pa.Check.ge(-100.0), pa.Check.le(100.0)], coerce=True),
    },
    strict=False,
    coerce=True,
)


def validate_multimodal_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate multimodal listing DataFrame with Pandera."""
    checked = df.copy()
    checked["listing_id"] = checked["listing_id"].astype(str)
    return LISTING_DF_SCHEMA.validate(checked, lazy=True)
