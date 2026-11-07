"""Download real InsideAirbnb San Francisco listings + photos and score them with OpenAI CLIP.

100% REAL DATA PIPELINE (Zero synthetic rows):
1. Discovers the latest official San Francisco `listings.csv.gz` URL from insideairbnb.com.
2. Cleans real USD prices, room specs, guest `review_scores_accuracy`, and host descriptions.
3. Downloads real listing photographs (`a0.muscache.com`) concurrently in memory.
4. Runs OpenAI CLIP (`openai/clip-vit-base-patch32`) on Apple MPS / CUDA / CPU to compute:
   - Normalized 512-dim visual search embeddings (`clip_search_index.npz`)
   - Contrastive 0–100 visual truth scores (Brightness, Spaciousness, Luxury Finish, Cleanliness)
   - Percentile-calibrated `catfish_gap_score` and `hidden_gem_score`
5. Validates all records via Pandera and writes `data/processed/sf_airbnb_multimodal.parquet`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from listing_truth.clip_scorer import CLIPVisualTruthScorer
from listing_truth.config import (
    CLIP_INDEX_NPZ_PATH,
    MULTIMODAL_PARQUET_PATH,
    load_vision_config,
)
from listing_truth.schemas import validate_multimodal_df
from listing_truth.text_claims import compute_catfish_and_gem_scores, extract_text_claims

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def discover_latest_insideairbnb_url(city_slug: str = "united-states/ca/san-francisco") -> str:
    """Scrape InsideAirbnb's official data catalog page for the latest `listings.csv.gz` URL."""
    resp = requests.get("https://insideairbnb.com/get-the-data/", timeout=25)
    resp.raise_for_status()
    pattern = rf"https?://data\.insideairbnb\.com/{re.escape(city_slug)}/\d{{4}}-\d{{2}}-\d{{2}}/data/listings\.csv\.gz"
    matches = re.findall(pattern, resp.text)
    if not matches:
        raise RuntimeError(f"Could not find {city_slug} listings.csv.gz URL on insideairbnb.com")
    latest_url = sorted(set(matches))[-1]
    logger.info("Discovered latest InsideAirbnb snapshot: %s", latest_url)
    return latest_url


def _parse_price_usd(val: object) -> float | None:
    if pd.isna(val):
        return None
    cleaned = re.sub(r"[^0-9.]", "", str(val))
    if not cleaned:
        return None
    num = float(cleaned)
    return num if 20.0 <= num <= 2500.0 else None


def _parse_bathrooms(row: pd.Series) -> float:
    b_val = row.get("bathrooms")
    if pd.notna(b_val):
        try:
            return max(0.0, min(15.0, float(b_val)))
        except ValueError:
            pass
    txt = str(row.get("bathrooms_text", "") or "").lower()
    if "half" in txt:
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)", txt)
    return max(0.0, min(15.0, float(m.group(1)))) if m else 1.0


def _fetch_single_image(args: tuple[str, str]) -> tuple[str, Image.Image | None]:
    lid, url = args
    try:
        # Append moderate resize query param supported by muscache CDN for fast transfer
        sep = "&" if "?" in url else "?"
        fast_url = f"{url}{sep}im_w=480"
        r = requests.get(
            fast_url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            timeout=10,
        )
        if r.status_code != 200:
            return lid, None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if img.width < 80 or img.height < 80:
            return lid, None
        return lid, img
    except Exception:
        return lid, None


def build_real_multimodal_dataset() -> pd.DataFrame:
    cfg = load_vision_config()
    snapshot_url = discover_latest_insideairbnb_url(cfg["insideairbnb_city_slug"])

    logger.info("Downloading official InsideAirbnb gzipped table...")
    raw = pd.read_csv(snapshot_url, compression="gzip", low_memory=False)
    logger.info("Loaded %d raw InsideAirbnb rows.", len(raw))

    raw["price_usd"] = raw["price"].apply(_parse_price_usd)
    valid = raw.dropna(
        subset=["price_usd", "picture_url", "description", "name", "neighbourhood_cleansed"]
    ).copy()
    valid = valid[valid["description"].astype(str).str.len() >= 60]

    target_n = int(cfg.get("sample_size_listings", 900))
    # Sample slightly more to account for any 404 expired photo URLs
    candidates = valid.sample(n=min(len(valid), int(target_n * 1.15)), random_state=42).copy()

    logger.info("Fetching %d real Airbnb listing photos concurrently...", len(candidates))
    tasks = [
        (str(r["id"]), str(r["picture_url"]))
        for _, r in candidates.iterrows()
    ]
    fetched_map: dict[str, Image.Image] = {}
    with ThreadPoolExecutor(max_workers=24) as pool:
        for lid, img in pool.map(_fetch_single_image, tasks):
            if img is not None:
                fetched_map[lid] = img
                if len(fetched_map) >= target_n:
                    break

    kept = candidates[candidates["id"].astype(str).isin(fetched_map.keys())].head(target_n).copy()
    kept = kept.reset_index(drop=True)
    ordered_images = [fetched_map[str(r["id"])] for _, r in kept.iterrows()]
    logger.info("Successfully downloaded %d verified real listing photographs.", len(ordered_images))

    # Run real OpenAI CLIP on Apple Silicon MPS / GPU
    scorer = CLIPVisualTruthScorer(cfg["clip_model_id"])
    img_embeddings = scorer.encode_images(ordered_images, batch_size=32)
    axis_scores = scorer.score_contrastive_axes(img_embeddings, cfg["contrastive_axes"])
    query_prompts = list(cfg["visual_search_queries"])
    query_embeddings = scorer.encode_texts(query_prompts)

    records: list[dict] = []
    for idx, row in kept.iterrows():
        claims = extract_text_claims(row.get("name"), row.get("description"))
        vb = float(axis_scores["vis_brightness"][idx])
        vs = float(axis_scores["vis_spaciousness"][idx])
        vl = float(axis_scores["vis_luxury_finish"][idx])
        vc = float(axis_scores["vis_cleanliness"][idx])
        v_comp = round(0.30 * vl + 0.25 * vb + 0.25 * vs + 0.20 * vc, 2)

        acc_score = (
            float(row["review_scores_accuracy"])
            if pd.notna(row.get("review_scores_accuracy"))
            else None
        )

        records.append(
            {
                "listing_id": str(row["id"]),
                "name": str(row["name"]).strip()[:140],
                "neighbourhood": str(row["neighbourhood_cleansed"]).strip(),
                "room_type": str(row["room_type"]).strip(),
                "accommodates": int(max(1, min(20, int(row.get("accommodates", 2) or 2)))),
                "bedrooms": float(
                    max(0.0, min(15.0, float(row["bedrooms"])))
                    if pd.notna(row.get("bedrooms"))
                    else 1.0
                ),
                "beds": float(
                    max(0.0, min(25.0, float(row["beds"])))
                    if pd.notna(row.get("beds"))
                    else 1.0
                ),
                "bathrooms": _parse_bathrooms(row),
                "latitude": round(float(row["latitude"]), 5),
                "longitude": round(float(row["longitude"]), 5),
                "price_usd": round(float(row["price_usd"]), 2),
                "number_of_reviews": int(row.get("number_of_reviews", 0) or 0),
                "review_scores_accuracy": acc_score,
                "picture_url": str(row["picture_url"]),
                "listing_url": str(
                    row.get("listing_url") or f"https://www.airbnb.com/rooms/{row['id']}"
                ),
                **claims,
                "vis_brightness": vb,
                "vis_spaciousness": vs,
                "vis_luxury_finish": vl,
                "vis_cleanliness": vc,
                "vis_composite_quality": v_comp,
            }
        )

    out_df = compute_catfish_and_gem_scores(pd.DataFrame(records))

    validated_df = validate_multimodal_df(out_df)

    MULTIMODAL_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_parquet(MULTIMODAL_PARQUET_PATH, index=False)

    np.savez_compressed(
        CLIP_INDEX_NPZ_PATH,
        listing_ids=validated_df["listing_id"].to_numpy(dtype=str),
        image_embeddings=img_embeddings.astype(np.float16),
        query_prompts=np.array(query_prompts, dtype=object),
        query_embeddings=query_embeddings.astype(np.float16),
    )

    logger.info(
        "Saved %d validated real multimodal listings -> %s and CLIP index -> %s",
        len(validated_df),
        MULTIMODAL_PARQUET_PATH,
        CLIP_INDEX_NPZ_PATH,
    )
    return validated_df


if __name__ == "__main__":
    build_real_multimodal_dataset()
