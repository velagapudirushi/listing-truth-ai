"""Multimodal Fair-Price LightGBM Valuation Engine + 3-Stage Ablation + TreeSHAP."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from listing_truth.config import ARTIFACTS_DIR, CLIP_INDEX_NPZ_PATH

TABULAR_CAT_COLS = ["neighbourhood", "room_type"]
TABULAR_NUM_COLS = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "latitude",
    "longitude",
]
TEXT_CLAIM_COLS = [
    "word_count",
    "claims_bright_light",
    "claims_spacious",
    "claims_luxury_modern",
    "claims_spotless_clean",
    "text_hype_score",
]
VISION_SCORE_COLS = [
    "vis_brightness",
    "vis_spaciousness",
    "vis_luxury_finish",
    "vis_cleanliness",
    "vis_composite_quality",
    "catfish_gap_score",
]
SVD_COLS = [f"clip_svd_{i}" for i in range(1, 9)]

FULL_FEATURE_COLS = TABULAR_CAT_COLS + TABULAR_NUM_COLS + TEXT_CLAIM_COLS + VISION_SCORE_COLS + SVD_COLS

HUMAN_LABELS = {
    "neighbourhood": "Neighborhood",
    "room_type": "Room Type (Entire Home / Private Room)",
    "accommodates": "Max Guests Accommodated",
    "bedrooms": "Bedrooms",
    "beds": "Beds",
    "bathrooms": "Bathrooms",
    "latitude": "Latitude (Micro-Location)",
    "longitude": "Longitude (Micro-Location)",
    "word_count": "Description Word Count",
    "claims_bright_light": "Text Claims 'Bright / Sunlit'",
    "claims_spacious": "Text Claims 'Spacious'",
    "claims_luxury_modern": "Text Claims 'Luxury / Modern'",
    "claims_spotless_clean": "Text Claims 'Spotless'",
    "text_hype_score": "Host Marketing Hype Score (0–100)",
    "vis_brightness": "CLIP Photo Brightness Score",
    "vis_spaciousness": "CLIP Photo Spaciousness Score",
    "vis_luxury_finish": "CLIP Photo Luxury/Modern Finish",
    "vis_cleanliness": "CLIP Photo Cleanliness/Order",
    "vis_composite_quality": "CLIP Overall Visual Quality",
    "catfish_gap_score": "Text-vs-Photo Catfish Gap",
    **{f"clip_svd_{i}": f"CLIP Embedding Visual Component #{i}" for i in range(1, 9)},
}


@dataclass
class MultimodalModelBundle:
    """Trained multimodal fair-price regressor, SVD projector, SHAP explainer, and ablation results."""

    model: lgb.LGBMRegressor
    explainer: shap.TreeExplainer
    svd: TruncatedSVD
    category_levels: dict[str, list[str]]
    ablation_metrics: dict[str, Any]
    enriched_df: pd.DataFrame
    global_shap_df: pd.DataFrame


def _align_raw_embeddings(df: pd.DataFrame, npz_path: Path | None = None) -> np.ndarray:
    """Return raw 512-dim CLIP image embeddings aligned row-by-row with `df`."""
    archive = np.load(npz_path or CLIP_INDEX_NPZ_PATH, allow_pickle=True)
    id_to_idx = {str(lid): i for i, lid in enumerate(archive["listing_ids"].astype(str))}
    emb_matrix = archive["image_embeddings"].astype(np.float32)
    aligned_indices = [id_to_idx.get(str(lid), 0) for lid in df["listing_id"].astype(str)]
    return emb_matrix[aligned_indices]


def _attach_clip_svd_components(
    df: pd.DataFrame,
    npz_path: Path | None = None,
    fitted_svd: TruncatedSVD | None = None,
) -> tuple[pd.DataFrame, TruncatedSVD]:
    """Project 512-dim CLIP image embeddings into 8 dense visual SVD components."""
    X_emb = _align_raw_embeddings(df, npz_path=npz_path)

    if fitted_svd is None:
        svd = TruncatedSVD(n_components=8, random_state=42)
        comps = svd.fit_transform(X_emb)
    else:
        svd = fitted_svd
        comps = svd.transform(X_emb)

    out = df.copy()
    for i, col in enumerate(SVD_COLS):
        out[col] = np.round(comps[:, i], 5)
    return out, svd


def _prepare_matrix(
    df: pd.DataFrame,
    cols: list[str],
    category_levels: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    X = df[cols].copy()
    levels: dict[str, list[str]] = {}
    for c in cols:
        if c in TABULAR_CAT_COLS:
            if category_levels and c in category_levels:
                cats = category_levels[c]
            else:
                cats = sorted(X[c].astype(str).unique().tolist())
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
            levels[c] = cats
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X, levels


def _evaluate_stage_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    y_log: np.ndarray,
    raw_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run leakage-free 5-fold out-of-fold evaluation predicting log1p(price_usd).

    When `raw_embeddings` is passed (Stage 3), `TruncatedSVD` is fit strictly on
    `tr_idx` inside each fold and applied to `va_idx` to prevent projection leakage.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_log_preds = np.zeros(len(df), dtype=np.float64)

    params = {
        "n_estimators": 220,
        "learning_rate": 0.045,
        "num_leaves": 23,
        "min_child_samples": 18,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 1,
    }

    for tr_idx, va_idx in kf.split(df):
        tr_df = df.iloc[tr_idx].copy()
        va_df = df.iloc[va_idx].copy()
        if raw_embeddings is not None:
            fold_svd = TruncatedSVD(n_components=8, random_state=42)
            tr_comps = fold_svd.fit_transform(raw_embeddings[tr_idx])
            va_comps = fold_svd.transform(raw_embeddings[va_idx])
            for i, col in enumerate(SVD_COLS):
                tr_df[col] = np.round(tr_comps[:, i], 5)
                va_df[col] = np.round(va_comps[:, i], 5)

        X_tr, lvls = _prepare_matrix(tr_df, feature_cols)
        X_va, _ = _prepare_matrix(va_df, feature_cols, category_levels=lvls)
        reg = lgb.LGBMRegressor(**params)
        reg.fit(X_tr, y_log[tr_idx])
        oof_log_preds[va_idx] = reg.predict(X_va)

    true_usd = np.expm1(y_log)
    pred_usd = np.clip(np.expm1(oof_log_preds), 20.0, 3000.0)

    return {
        "log_r2": round(float(r2_score(y_log, oof_log_preds)), 4),
        "log_rmse": round(float(root_mean_squared_error(y_log, oof_log_preds)), 4),
        "usd_mae": round(float(mean_absolute_error(true_usd, pred_usd)), 2),
        "oof_pred_usd": pred_usd,
    }


def train_multimodal_valuation_engine(
    df: pd.DataFrame,
    npz_path: Path | None = None,
    save_metrics_path: Path | None = None,
) -> MultimodalModelBundle:
    """Run 3-stage multimodal ablation and fit final TreeSHAP Fair-Price Valuation Engine."""
    raw_emb = _align_raw_embeddings(df, npz_path=npz_path)
    with_svd, svd = _attach_clip_svd_components(df, npz_path=npz_path)
    y_log = np.log1p(df["price_usd"].to_numpy(dtype=np.float64))

    stage1 = _evaluate_stage_cv(df, TABULAR_CAT_COLS + TABULAR_NUM_COLS, y_log)
    stage2 = _evaluate_stage_cv(
        df, TABULAR_CAT_COLS + TABULAR_NUM_COLS + TEXT_CLAIM_COLS, y_log
    )
    stage3 = _evaluate_stage_cv(
        df, FULL_FEATURE_COLS, y_log, raw_embeddings=raw_emb
    )

    # Fit full model for interactive fair-price valuation + TreeSHAP explanations
    X_full, category_levels = _prepare_matrix(with_svd, FULL_FEATURE_COLS)
    final_model = lgb.LGBMRegressor(
        n_estimators=240,
        learning_rate=0.045,
        num_leaves=23,
        min_child_samples=18,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    final_model.fit(X_full, y_log)

    explainer = shap.TreeExplainer(final_model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        shap_matrix = np.asarray(explainer.shap_values(X_full))

    global_shap_df = (
        pd.DataFrame(
            {
                "feature": FULL_FEATURE_COLS,
                "label": [HUMAN_LABELS[c] for c in FULL_FEATURE_COLS],
                "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    enriched = with_svd.copy()
    enriched["fair_price_usd"] = np.round(stage3["oof_pred_usd"], 0)
    enriched["overpricing_usd"] = np.round(enriched["price_usd"] - enriched["fair_price_usd"], 0)
    enriched["overpricing_pct"] = np.round(
        100.0 * (enriched["price_usd"] - enriched["fair_price_usd"]) / np.maximum(enriched["fair_price_usd"], 1.0),
        1,
    )

    ablation_summary = {
        "dataset_listings": int(len(enriched)),
        "stage_1_tabular_only": {
            k: v for k, v in stage1.items() if k != "oof_pred_usd"
        },
        "stage_2_tabular_plus_text": {
            k: v for k, v in stage2.items() if k != "oof_pred_usd"
        },
        "stage_3_full_multimodal_clip": {
            k: v for k, v in stage3.items() if k != "oof_pred_usd"
        },
        "multimodal_r2_gain_over_tabular": round(
            stage3["log_r2"] - stage1["log_r2"], 4
        ),
        "multimodal_mae_dollar_improvement": round(
            stage1["usd_mae"] - stage3["usd_mae"], 2
        ),
        "top_6_shap_drivers": global_shap_df.head(6)[
            ["feature", "label", "mean_abs_shap"]
        ].to_dict(orient="records"),
    }

    metrics_out = save_metrics_path or (ARTIFACTS_DIR / "ablation_metrics.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    return MultimodalModelBundle(
        model=final_model,
        explainer=explainer,
        svd=svd,
        category_levels=category_levels,
        ablation_metrics=ablation_summary,
        enriched_df=enriched,
        global_shap_df=global_shap_df,
    )


def explain_listing_fair_price(
    bundle: MultimodalModelBundle,
    listing_id: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return enriched listing row and per-feature TreeSHAP valuation breakdown."""
    matches = bundle.enriched_df[bundle.enriched_df["listing_id"].astype(str) == str(listing_id)]
    row_df = matches.iloc[[0]] if len(matches) else bundle.enriched_df.iloc[[0]]
    X_row, _ = _prepare_matrix(row_df, FULL_FEATURE_COLS, category_levels=bundle.category_levels)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        vec = np.asarray(bundle.explainer.shap_values(X_row))[0]

    breakdown = (
        pd.DataFrame(
            {
                "feature": FULL_FEATURE_COLS,
                "driver": [HUMAN_LABELS[c] for c in FULL_FEATURE_COLS],
                "value": [str(row_df.iloc[0][c]) for c in FULL_FEATURE_COLS],
                "shap_log_price_impact": vec,
            }
        )
        .assign(abs_impact=lambda d: d["shap_log_price_impact"].abs())
        .sort_values("abs_impact", ascending=False)
        .drop(columns=["abs_impact"])
        .reset_index(drop=True)
    )
    return row_df.iloc[0], breakdown
