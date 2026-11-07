"""Fast zero-GPU multimodal CLIP cosine search engine for Streamlit Cloud."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from listing_truth.config import CLIP_INDEX_NPZ_PATH, MULTIMODAL_PARQUET_PATH


class MultimodalListingSearch:
    """Searches real Airbnb photos in CLIP vector space without loading PyTorch at runtime."""

    def __init__(
        self,
        parquet_path: Path | None = None,
        npz_path: Path | None = None,
    ) -> None:
        self.df = pd.read_parquet(parquet_path or MULTIMODAL_PARQUET_PATH)
        archive = np.load(npz_path or CLIP_INDEX_NPZ_PATH, allow_pickle=True)
        self.listing_ids: np.ndarray = archive["listing_ids"].astype(str)
        self.image_embeddings: np.ndarray = archive["image_embeddings"].astype(np.float32)
        self.query_prompts: list[str] = [str(q) for q in archive["query_prompts"].tolist()]
        self.query_embeddings: np.ndarray = archive["query_embeddings"].astype(np.float32)

        # Align dataframe order with embedding matrix order
        id_to_idx = {lid: i for i, lid in enumerate(self.listing_ids)}
        self.df = (
            self.df[self.df["listing_id"].astype(str).isin(id_to_idx)]
            .assign(_emb_idx=lambda d: d["listing_id"].astype(str).map(id_to_idx))
            .sort_values("_emb_idx")
            .drop(columns=["_emb_idx"])
            .reset_index(drop=True)
        )
        self.image_embeddings = self.image_embeddings[: len(self.df)]

    def search_by_visual_prompt(
        self,
        prompt: str,
        max_price_usd: float | None = None,
        room_type: str | None = None,
        top_k: int = 12,
    ) -> pd.DataFrame:
        """Rank real listing photos by exact CLIP cosine similarity to a visual prompt."""
        if prompt in self.query_prompts:
            q_idx = self.query_prompts.index(prompt)
            q_vec = self.query_embeddings[q_idx]
        else:
            # Blend closest pre-encoded CLIP prompts via token overlap so free-text also works zero-GPU!
            tokens = set(prompt.lower().split())
            weights = np.array(
                [
                    max(0.05, len(tokens & set(p.lower().split())))
                    for p in self.query_prompts
                ],
                dtype=np.float32,
            )
            q_vec = (weights[:, None] * self.query_embeddings).sum(axis=0)
            q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-8)

        sims = self.image_embeddings @ q_vec
        out = self.df.copy()
        out["clip_similarity"] = np.round(sims, 4)

        if max_price_usd is not None:
            out = out[out["price_usd"] <= max_price_usd]
        if room_type and room_type != "All":
            out = out[out["room_type"] == room_type]

        return out.sort_values("clip_similarity", ascending=False).head(top_k).reset_index(drop=True)
