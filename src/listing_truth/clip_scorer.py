"""Apple-MPS / GPU accelerated OpenAI CLIP zero-shot visual truth scorer & indexer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPVisualTruthScorer:
    """Scores real listing photographs along contrastive visual axes using OpenAI CLIP."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info("Loading %s onto device=%s", model_id, self.device)
        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def _extract_features_tensor(self, out: Any):
        """Handle both raw tensor and BaseModelOutputWithPooling across transformers versions."""
        if hasattr(out, "image_embeds"):
            return out.image_embeds
        if hasattr(out, "text_embeds"):
            return out.text_embeds
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        return out

    def encode_texts(self, prompts: list[str]) -> np.ndarray:
        """Return L2-normalized (N, D) float32 CLIP text embeddings."""
        with self.torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self._extract_features_tensor(self.model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype(np.float32)

    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized (N, D) float32 CLIP image embeddings in batches."""
        all_vecs: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self._extract_features_tensor(self.model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_vecs.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(all_vecs)

    def score_contrastive_axes(
        self,
        image_embeddings: np.ndarray,
        contrastive_axes: dict[str, dict[str, str]],
        logit_scale: float = 28.0,
    ) -> dict[str, np.ndarray]:
        """Compute 0–100 visual truth probabilities for each positive vs negative prompt pair."""
        scores: dict[str, np.ndarray] = {}
        for axis_name, pair in contrastive_axes.items():
            pair_vecs = self.encode_texts([pair["positive"], pair["negative"]])  # (2, D)
            logits = logit_scale * (image_embeddings @ pair_vecs.T)  # (N, 2)
            logits_shifted = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            pos_prob = exp_logits[:, 0] / exp_logits.sum(axis=1)
            scores[axis_name] = np.round(pos_prob * 100.0, 2)
        return scores
