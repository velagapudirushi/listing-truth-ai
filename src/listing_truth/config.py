"""Canonical repository paths and configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "vision_prompts.yaml"
MULTIMODAL_PARQUET_PATH = REPO_ROOT / "data" / "processed" / "sf_airbnb_multimodal.parquet"
CLIP_INDEX_NPZ_PATH = REPO_ROOT / "data" / "processed" / "clip_search_index.npz"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def load_vision_config(path: Path | None = None) -> dict[str, Any]:
    """Load contrastive CLIP prompts and search configurations."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
