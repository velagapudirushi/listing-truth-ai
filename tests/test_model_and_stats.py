"""Unit tests for 3-stage Multimodal LightGBM ablation, TreeSHAP, and SciPy audits."""

import pandas as pd

from listing_truth.config import CLIP_INDEX_NPZ_PATH, MULTIMODAL_PARQUET_PATH
from listing_truth.model import (
    explain_listing_fair_price,
    train_multimodal_valuation_engine,
)
from listing_truth.stats_audit import run_multimodal_statistical_audit


def test_multimodal_ablation_and_shap_explanation(tmp_path) -> None:
    df = pd.read_parquet(MULTIMODAL_PARQUET_PATH)
    metrics_out = tmp_path / "ablation.json"
    bundle = train_multimodal_valuation_engine(df, CLIP_INDEX_NPZ_PATH, save_metrics_path=metrics_out)

    assert metrics_out.exists()
    assert bundle.ablation_metrics["stage_3_full_multimodal_clip"]["log_r2"] > 0.35
    assert "fair_price_usd" in bundle.enriched_df.columns

    sample_id = str(bundle.enriched_df.iloc[0]["listing_id"])
    row_s, breakdown = explain_listing_fair_price(bundle, sample_id)
    assert float(row_s["fair_price_usd"]) > 0.0
    assert len(breakdown) == 28


def test_scipy_multimodal_statistical_audit() -> None:
    df = pd.read_parquet(MULTIMODAL_PARQUET_PATH)
    audits = run_multimodal_statistical_audit(df)
    assert set(audits.keys()) == {
        "visual_price_premium",
        "hype_vs_reality_corr",
        "guest_accuracy_validation",
    }
    assert audits["visual_price_premium"].group_high_metric >= audits["visual_price_premium"].group_low_metric
