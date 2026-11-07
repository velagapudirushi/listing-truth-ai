"""Interactive 4-Tab Streamlit Dashboard for ListingTruth AI (100% Real InsideAirbnb + CLIP)."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from listing_truth.config import CLIP_INDEX_NPZ_PATH, MULTIMODAL_PARQUET_PATH
from listing_truth.model import (
    explain_listing_fair_price,
    train_multimodal_valuation_engine,
)
from listing_truth.search_engine import MultimodalListingSearch
from listing_truth.stats_audit import run_multimodal_statistical_audit
from listing_truth.warehouse_sql import ListingTruthWarehouseSQL

st.set_page_config(
    page_title="ListingTruth AI | Multimodal Photo-vs-Text Rental Auditor",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading CLIP vector index & fitting Multimodal LightGBM + TreeSHAP...")
def load_full_system():
    raw_df = pd.read_parquet(MULTIMODAL_PARQUET_PATH)
    search_engine = MultimodalListingSearch(MULTIMODAL_PARQUET_PATH, CLIP_INDEX_NPZ_PATH)
    bundle = train_multimodal_valuation_engine(raw_df, CLIP_INDEX_NPZ_PATH)
    stats_results = run_multimodal_statistical_audit(bundle.enriched_df)
    sql_engine = ListingTruthWarehouseSQL(MULTIMODAL_PARQUET_PATH)
    neighborhood_df = sql_engine.get_neighborhood_audit()
    return search_engine, bundle, stats_results, neighborhood_df


def _render_listing_cards(rows_df: pd.DataFrame, score_col: str, score_label: str) -> None:
    """Render 3-column visual listing cards with real Airbnb photographs & CLIP metrics."""
    cols = st.columns(3)
    for idx, row in rows_df.iterrows():
        with cols[idx % 3]:
            st.image(row["picture_url"], width="stretch")
            st.markdown(f"**{row['name'][:68]}**")
            st.caption(
                f"📍 {row['neighbourhood']} · {row['room_type']} · "
                f"**${row['price_usd']:.0f}/night**"
            )
            st.markdown(
                f"- **{score_label}:** `{row[score_col]:.2f}`\n"
                f"- **Text Hype:** `{row['text_hype_score']:.0f}/100` vs "
                f"**CLIP Photo Quality:** `{row['vis_composite_quality']:.0f}/100`\n"
                f"- [Open Real Airbnb Listing ↗]({row['listing_url']})"
            )
            with st.expander("Read Host Marketing Copy"):
                st.write(row["clean_description"][:360] + "...")


def main() -> None:
    st.title("🔍 ListingTruth AI: Multimodal Rental 'BS & Hidden Gem' Detector")
    st.caption(
        "Auditing 100% real San Francisco InsideAirbnb listings by comparing host marketing copy "
        "against real listing photographs using OpenAI CLIP (ViT-B/32), DuckDB SQL, LightGBM & TreeSHAP."
    )

    search_engine, bundle, stats_results, neighborhood_df = load_full_system()
    ablation = bundle.ablation_metrics
    enriched_df = bundle.enriched_df

    with st.sidebar:
        st.header("System Overview")
        st.markdown(
            f"- **Real Listings Indexed:** `{len(enriched_df):,}` (San Francisco)\n"
            "- **Data Source:** Official `InsideAirbnb` Snapshot\n"
            "- **Vision Backbone:** `openai/clip-vit-base-patch32`\n"
            "- **SQL Engine:** `DuckDB` over `Parquet`\n"
            "- **Valuation Core:** `LightGBM` + `TreeSHAP`"
        )
        st.divider()
        max_price_filter = st.slider("Max Nightly Budget ($ USD)", 50, 1000, 450, 25)
        room_types = ["All"] + sorted(enriched_df["room_type"].unique().tolist())
        selected_room_type = st.selectbox("Room Type Filter", room_types)

    # Top KPI Strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Real SF Listings & Photos", f"{len(enriched_df):,}")
    k2.metric(
        "Multimodal Log-Price R²",
        f"{ablation['stage_3_full_multimodal_clip']['log_r2']:.3f}",
        delta=f"+{ablation['multimodal_r2_gain_over_tabular']:.3f} vs Tabular-Only",
    )
    k3.metric(
        "Fair-Price MAE Improvement",
        f"${ablation['stage_3_full_multimodal_clip']['usd_mae']:.0f}",
        delta=f"-${ablation['multimodal_mae_dollar_improvement']:.1f}/night error",
    )
    k4.metric(
        "Visual Luxury Price Premium",
        stats_results["visual_price_premium"].finding_headline,
        delta="Mann-Whitney p < 0.01",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🖼️ CLIP Visual Truth Search & Leaderboards",
            "⚖️ Fair-Price & Catfish Auditor (TreeSHAP)",
            "📈 Multimodal Ablation & SciPy Proof",
            "🧮 Live DuckDB SQL Playground",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: CLIP Visual Search & Catfish vs Hidden Gem Galleries
    # ------------------------------------------------------------------
    with tab1:
        mode = st.radio(
            "Explore Mode",
            options=[
                "🔎 Natural-Language CLIP Photo Search",
                "🚩 Biggest 'Catfish / Overhyped' Listings (High Text Hype, Low Photo Score)",
                "💎 Best 'Hidden Gems' (High Photo Quality, Modest Copy & Underpriced)",
            ],
            horizontal=True,
        )

        if mode.startswith("🔎"):
            c1, c2 = st.columns([2, 1])
            with c1:
                selected_prompt = st.selectbox(
                    "Search Photos by Visual Reality (Prompts Hosts Never Write!)",
                    options=search_engine.query_prompts,
                )
            with c2:
                custom_override = st.text_input(
                    "Or type custom keywords (optional)",
                    placeholder="e.g., cramped dark room or bright modern bedroom",
                )
            active_query = custom_override.strip() if custom_override.strip() else selected_prompt
            results = search_engine.search_by_visual_prompt(
                active_query,
                max_price_usd=float(max_price_filter),
                room_type=selected_room_type,
                top_k=6,
            )
            st.markdown(f"**Top {len(results)} Real Photos Matching:** *\"{active_query}\"*")
            if len(results) > 0:
                _render_listing_cards(results, "clip_similarity", "CLIP Cosine Similarity")
            else:
                st.warning("No listings match current sidebar price/room filters.")

        elif mode.startswith("🚩"):
            filtered = enriched_df[enriched_df["price_usd"] <= max_price_filter]
            if selected_room_type != "All":
                filtered = filtered[filtered["room_type"] == selected_room_type]
            top_catfish = filtered.sort_values("catfish_gap_score", ascending=False).head(6).reset_index(drop=True)
            st.markdown(
                "**Listings where breathless marketing adjectives (`stunning`, `luxury`, `sun-drenched`) "
                "sharply exceed what OpenAI `CLIP` sees in the actual photo:**"
            )
            if len(top_catfish) > 0:
                _render_listing_cards(top_catfish, "catfish_gap_score", "Catfish Gap Score")

        else:
            filtered = enriched_df[enriched_df["price_usd"] <= max_price_filter]
            if selected_room_type != "All":
                filtered = filtered[filtered["room_type"] == selected_room_type]
            top_gems = filtered.sort_values("hidden_gem_score", ascending=False).head(6).reset_index(drop=True)
            st.markdown(
                "**High-visual-quality apartments whose hosts wrote modest descriptions and priced below peers:**"
            )
            if len(top_gems) > 0:
                _render_listing_cards(top_gems, "hidden_gem_score", "Hidden Gem Score")

    # ------------------------------------------------------------------
    # TAB 2: Single-Listing Fair-Price & TreeSHAP Valuation Auditor
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Inspect Any Real Listing: Fair Nightly Value vs. Asked Price")
        sample_choices = (
            enriched_df.assign(abs_gap=lambda d: d["catfish_gap_score"].abs())
            .sort_values("abs_gap", ascending=False)
            .head(80)
        )
        label_map = {
            f"[{r.neighbourhood}] {r.name[:55]} — Asking ${r.price_usd:.0f} (Fair: ${r.fair_price_usd:.0f})": str(
                r.listing_id
            )
            for r in sample_choices.itertuples()
        }
        chosen_label = st.selectbox("Select a Real San Francisco Listing to Audit", list(label_map.keys()))
        row_s, shap_df = explain_listing_fair_price(bundle, label_map[chosen_label])

        left_col, right_col = st.columns([1.05, 1.45])
        with left_col:
            st.image(row_s["picture_url"], width="stretch")
            diff = float(row_s["overpricing_usd"])
            verdict = (
                f"🔴 Overpriced by +${diff:.0f}/night (+{row_s['overpricing_pct']:.0f}%)"
                if diff > 20
                else (
                    f"🟢 Undervalued Gem by -${abs(diff):.0f}/night ({row_s['overpricing_pct']:.0f}%)"
                    if diff < -20
                    else "🟡 Priced Near Multimodal Fair Value"
                )
            )
            st.markdown(f"### {verdict}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Asked Price", f"${row_s['price_usd']:.0f}")
            m2.metric("Multimodal Fair Price", f"${row_s['fair_price_usd']:.0f}")
            m3.metric("Catfish Gap", f"{row_s['catfish_gap_score']:+.1f}")

            st.markdown(
                f"**4-Axis `CLIP` Photo Breakdown:** Brightness `{row_s['vis_brightness']:.0f}` · "
                f"Space `{row_s['vis_spaciousness']:.0f}` · Luxury Finish `{row_s['vis_luxury_finish']:.0f}` · "
                f"Cleanliness `{row_s['vis_cleanliness']:.0f}`"
            )
            st.caption(row_s["clean_description"][:420] + "...")

        with right_col:
            top_s = shap_df.head(10).iloc[::-1]
            bar_colors = ["#f43f5e" if v > 0 else "#10b981" for v in top_s["shap_log_price_impact"]]
            fig_shap = go.Figure(
                go.Bar(
                    x=top_s["shap_log_price_impact"],
                    y=[f"{r.driver} = {r.value}" for r in top_s.itertuples()],
                    orientation="h",
                    marker_color=bar_colors,
                )
            )
            fig_shap.update_layout(
                title="TreeSHAP Fair-Price Drivers (Red = Raises Fair Value, Green = Lowers Value)",
                xaxis_title="SHAP Contribution to log(Price USD)",
                height=430,
                margin=dict(l=10, r=10, t=40, b=20),
            )
            st.plotly_chart(fig_shap, width="stretch")

    # ------------------------------------------------------------------
    # TAB 3: 3-Stage Multimodal Ablation & Statistical Proof
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Does Seeing the Photo Beat Standard Tabular House-Price Models?")
        ab_rows = pd.DataFrame(
            [
                {
                    "Model Stage": "1. Tabular Only (Beds, Baths, Location)",
                    "Out-of-Fold Log-Price R²": ablation["stage_1_tabular_only"]["log_r2"],
                    "Out-of-Fold MAE ($ USD)": ablation["stage_1_tabular_only"]["usd_mae"],
                },
                {
                    "Model Stage": "2. Tabular + NLP Host Claims",
                    "Out-of-Fold Log-Price R²": ablation["stage_2_tabular_plus_text"]["log_r2"],
                    "Out-of-Fold MAE ($ USD)": ablation["stage_2_tabular_plus_text"]["usd_mae"],
                },
                {
                    "Model Stage": "3. Full Multimodal (Tabular + NLP + CLIP Vision)",
                    "Out-of-Fold Log-Price R²": ablation["stage_3_full_multimodal_clip"]["log_r2"],
                    "Out-of-Fold MAE ($ USD)": ablation["stage_3_full_multimodal_clip"]["usd_mae"],
                },
            ]
        )
        st.dataframe(ab_rows, width="stretch", hide_index=True)

        ca, cb = st.columns(2)
        with ca:
            fig_ab = px.bar(
                ab_rows,
                x="Model Stage",
                y="Out-of-Fold Log-Price R²",
                color="Model Stage",
                title="5-Fold Out-of-Fold R² Gain Across Modality Stages",
            )
            st.plotly_chart(fig_ab, width="stretch")
        with cb:
            fig_nb = px.scatter(
                neighborhood_df,
                x="avg_clip_visual_score",
                y="avg_text_hype",
                size="listing_count",
                color="avg_price_usd",
                hover_name="neighbourhood",
                title="SF Neighborhoods: Marketing Hype vs. CLIP Photo Reality",
                labels={
                    "avg_clip_visual_score": "Mean CLIP Visual Quality (0–100)",
                    "avg_text_hype": "Mean Host Text Hype (0–100)",
                },
            )
            st.plotly_chart(fig_nb, width="stretch")

        st.subheader("Rigorous `SciPy` Statistical Hypothesis Audit")
        for res in stats_results.values():
            st.info(f"**{res.test_name}** — {res.plain_english_verdict}")

    # ------------------------------------------------------------------
    # TAB 4: Live DuckDB SQL Playground
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Query Real Multimodal Airbnb Listings Directly With `DuckDB` SQL")
        default_sql = """SELECT
    neighbourhood,
    COUNT(*) AS listings,
    ROUND(AVG(price_usd), 0) AS avg_price_usd,
    ROUND(AVG(text_hype_score), 1) AS avg_hype,
    ROUND(AVG(vis_composite_quality), 1) AS avg_clip_photo_score,
    ROUND(AVG(catfish_gap_score), 1) AS avg_catfish_gap
FROM listings
GROUP BY 1
HAVING COUNT(*) >= 15
ORDER BY avg_catfish_gap DESC;"""
        sql_input = st.text_area("Analytical SQL (`listings` view)", value=default_sql, height=190)
        if st.button("Execute DuckDB Query"):
            try:
                res_df = ListingTruthWarehouseSQL(MULTIMODAL_PARQUET_PATH).run_custom_query(sql_input)
                st.dataframe(res_df, width="stretch")
            except Exception as exc:
                st.error(f"Query rejected: {exc}")


if __name__ == "__main__":
    main()
