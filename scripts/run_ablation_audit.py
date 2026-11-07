"""Execute the 3-stage LightGBM multimodal ablation + SciPy statistical audit on real SF listings."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from listing_truth.config import ARTIFACTS_DIR, CLIP_INDEX_NPZ_PATH, MULTIMODAL_PARQUET_PATH
from listing_truth.model import train_multimodal_valuation_engine
from listing_truth.schemas import validate_multimodal_df
from listing_truth.stats_audit import run_multimodal_statistical_audit
from listing_truth.text_claims import compute_catfish_and_gem_scores


def main() -> None:
    df = validate_multimodal_df(
        compute_catfish_and_gem_scores(pd.read_parquet(MULTIMODAL_PARQUET_PATH))
    )
    df.to_parquet(MULTIMODAL_PARQUET_PATH, index=False)
    bundle = train_multimodal_valuation_engine(df, npz_path=CLIP_INDEX_NPZ_PATH)
    audits = run_multimodal_statistical_audit(bundle.enriched_df)

    combined_report = {
        **bundle.ablation_metrics,
        "statistical_audits": {k: v.to_dict() for k, v in audits.items()},
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "ablation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)

    print(json.dumps(combined_report, indent=2))


if __name__ == "__main__":
    main()
