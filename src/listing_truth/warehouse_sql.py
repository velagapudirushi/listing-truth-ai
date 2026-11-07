"""DuckDB analytical SQL engine over real San Francisco multimodal Airbnb listings."""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

from listing_truth.config import MULTIMODAL_PARQUET_PATH


class ListingTruthWarehouseSQL:
    """Analytical SQL interface querying the real multimodal Parquet table via DuckDB."""

    def __init__(self, parquet_path: Path | None = None) -> None:
        self.parquet_path = Path(parquet_path or MULTIMODAL_PARQUET_PATH)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        escaped = str(self.parquet_path).replace("'", "''")
        con.execute(f"CREATE VIEW listings AS SELECT * FROM read_parquet('{escaped}')")
        return con

    def run_custom_query(self, sql: str) -> pd.DataFrame:
        """Execute a read-only SELECT / WITH analytical query against `listings`."""
        stripped = sql.strip().lower()
        if not (stripped.startswith("select") or stripped.startswith("with")):
            raise ValueError("Only read-only SELECT / WITH queries are permitted.")
        with self._connect() as con:
            return con.execute(sql).df()

    def get_neighborhood_audit(self) -> pd.DataFrame:
        """Rank neighborhoods by average marketing hype vs CLIP visual reality & price."""
        query = """
        SELECT
            neighbourhood,
            COUNT(*) AS listing_count,
            ROUND(AVG(price_usd), 0) AS avg_price_usd,
            ROUND(AVG(text_hype_score), 1) AS avg_text_hype,
            ROUND(AVG(vis_composite_quality), 1) AS avg_clip_visual_score,
            ROUND(AVG(catfish_gap_score), 1) AS avg_catfish_gap,
            ROUND(AVG(review_scores_accuracy), 2) AS avg_guest_accuracy
        FROM listings
        GROUP BY 1
        HAVING COUNT(*) >= 10
        ORDER BY avg_catfish_gap DESC
        """
        with self._connect() as con:
            return con.execute(query).df()

    def get_top_catfish_listings(self, limit: int = 15) -> pd.DataFrame:
        """Return listings with the largest positive gap between text hype and photo reality."""
        query = f"""
        SELECT
            listing_id,
            name,
            neighbourhood,
            room_type,
            price_usd,
            ROUND(text_hype_score, 1) AS text_hype_score,
            ROUND(vis_composite_quality, 1) AS vis_composite_quality,
            ROUND(catfish_gap_score, 1) AS catfish_gap_score,
            review_scores_accuracy,
            picture_url,
            listing_url,
            clean_description
        FROM listings
        ORDER BY catfish_gap_score DESC
        LIMIT {int(limit)}
        """
        with self._connect() as con:
            return con.execute(query).df()

    def get_top_hidden_gems(self, limit: int = 15) -> pd.DataFrame:
        """Return listings with high CLIP visual quality, modest copy, and strong price value."""
        query = f"""
        SELECT
            listing_id,
            name,
            neighbourhood,
            room_type,
            price_usd,
            ROUND(text_hype_score, 1) AS text_hype_score,
            ROUND(vis_composite_quality, 1) AS vis_composite_quality,
            ROUND(hidden_gem_score, 1) AS hidden_gem_score,
            review_scores_accuracy,
            picture_url,
            listing_url,
            clean_description
        FROM listings
        ORDER BY hidden_gem_score DESC
        LIMIT {int(limit)}
        """
        with self._connect() as con:
            return con.execute(query).df()
