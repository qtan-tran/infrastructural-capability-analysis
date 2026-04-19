"""Publication-quality figures: static (matplotlib/seaborn) + interactive (folium/plotly)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import folium
import matplotlib.pyplot as plt
import plotly.express as px
import polars as pl
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="paper")


def _density_vs_capability(df: pl.DataFrame, out: Path) -> None:
    pdf = df.to_pandas()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(
        data=pdf,
        x="linkage_density",
        y="capability_score",
        scatter_kws={"alpha": 0.4, "s": 15},
        line_kws={"color": "crimson"},
        ax=ax,
    )
    ax.set_xlabel("Linkage density (relations / product)")
    ax.set_ylabel("Capability score (z-sum)")
    ax.set_title("Linkage density vs. collective capability")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _density_heatmap(df: pl.DataFrame, out: Path) -> None:
    pdf = df.to_pandas()
    if "region" not in pdf.columns:
        return
    agg = (
        pdf.groupby("region")[["linkage_density", "capability_score"]]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        agg.set_index("region"),
        annot=True,
        cmap="viridis",
        fmt=".3f",
        ax=ax,
    )
    ax.set_title("Region x mean density / capability")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _interactive_scatter(df: pl.DataFrame, out: Path) -> None:
    pdf = df.to_pandas()
    fig = px.scatter(
        pdf,
        x="linkage_density",
        y="capability_score",
        color="region" if "region" in pdf.columns else None,
        hover_data=["community_id", "n_products", "n_countries"]
        if "community_id" in pdf.columns
        else None,
        size="n_products" if "n_products" in pdf.columns else None,
        title="Interactive: linkage density vs. capability",
    )
    fig.write_html(out)


def _geo_map(df: pl.DataFrame, out: Path) -> None:
    """Simple folium world map with region-level aggregates."""
    pdf = df.to_pandas()
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")
    region_centers = {
        "north": (50.0, 10.0),
        "south": (-10.0, 20.0),
        "mixed": (0.0, 40.0),
        "unknown": (0.0, -90.0),
    }
    if "region" in pdf.columns:
        agg = (
            pdf.groupby("region")
            .agg(mean_cap=("capability_score", "mean"), n=("capability_score", "size"))
            .reset_index()
        )
        for _, row in agg.iterrows():
            lat, lon = region_centers.get(row["region"], (0, 0))
            folium.CircleMarker(
                location=[lat, lon],
                radius=5 + float(row["n"]) ** 0.5,
                popup=(
                    f"{row['region']}: mean capability "
                    f"{row['mean_cap']:.2f} (n={int(row['n'])})"
                ),
                color="crimson",
                fill=True,
                fill_opacity=0.6,
            ).add_to(m)
    m.save(str(out))


def _coefficient_plot(causal_results: dict[str, Any], out: Path) -> None:
    ols = causal_results.get("ols", {})
    params = ols.get("params", {})
    pvals = ols.get("pvalues", {})
    cis = ols.get("conf_int", {})
    if not params:
        return
    rows = []
    for k, v in params.items():
        if k == "const":
            continue
        ci = cis.get(k, {})
        rows.append(
            {
                "variable": k,
                "coef": v,
                "ci_low": list(ci.values())[0] if ci else v,
                "ci_high": list(ci.values())[1] if ci else v,
                "pvalue": pvals.get(k, 1.0),
            }
        )
    if not rows:
        return
    pdf = pl.from_dicts(rows).to_pandas().sort_values("coef")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        pdf["coef"],
        pdf["variable"],
        xerr=[pdf["coef"] - pdf["ci_low"], pdf["ci_high"] - pdf["coef"]],
        fmt="o",
        capsize=3,
        color="steelblue",
    )
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_xlabel("OLS coefficient (HC3 SE, 95% CI)")
    ax.set_title("Predictors of capability score")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def generate_all_figures(
    analysis_df: pl.DataFrame,
    causal_results: dict[str, Any],
    results_dir: Path,
) -> None:
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _density_vs_capability(analysis_df, fig_dir / "scatter_density_capability.png")
    _density_heatmap(analysis_df, fig_dir / "heatmap_region.png")
    _interactive_scatter(analysis_df, fig_dir / "scatter_interactive.html")
    _geo_map(analysis_df, fig_dir / "geo_map.html")
    _coefficient_plot(causal_results, fig_dir / "coef_plot.png")
    logger.info("Wrote figures to %s", fig_dir)
