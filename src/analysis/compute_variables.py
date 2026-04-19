"""Compute the independent, dependent, and control variables.

The **capability score** is a composite of:
  1. Verified governance participation (exact match against
     ``governance_bodies.csv``).
  2. Metadata enrichment contributions (relations added *from* the
     community's products).
  3. Reuse metrics (incoming citations / downstream relations).

Crude regex-only governance detection is deliberately *not* used.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from sklearn.preprocessing import StandardScaler

from src.utils.io import load_governance_bodies
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ISO 3166 -> rough Global North / South mapping (UN DESA M49 coarse).
_GLOBAL_NORTH = {
    "AT", "BE", "BG", "CA", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR",
    "GB", "GR", "HR", "HU", "IE", "IS", "IT", "JP", "KR", "LT", "LU", "LV", "MT",
    "NL", "NO", "NZ", "PL", "PT", "RO", "SE", "SI", "SK", "US", "AU",
}


def _classify_region(countries) -> str:
    """Classify a list of ISO country codes as north / south / mixed / unknown.

    Accepts either a Python list/tuple/set or a Polars Series (which is what
    ``map_elements`` hands us when the column dtype is ``List[str]``).
    """
    if countries is None:
        return "unknown"
    # Normalise Polars Series -> list
    if hasattr(countries, "to_list"):
        countries = countries.to_list()
    if not countries:
        return "unknown"
    north = sum(1 for c in countries if c in _GLOBAL_NORTH)
    south = len(countries) - north
    if north and not south:
        return "north"
    if south and not north:
        return "south"
    return "mixed"


def _governance_score(label: str, description: str, gov_bodies: pl.DataFrame) -> int:
    """Count verified governance mentions (case-insensitive substring match)."""
    text = f"{label} {description}".lower()
    hits = 0
    for body in gov_bodies["name"].to_list():
        if body.lower() in text:
            hits += 1
    return hits


def compute_all_variables(
    processed_dir: Path,
    network_stats: pl.DataFrame,
    results_dir: Path,
    mode: str = "eu",
) -> pl.DataFrame:
    """Assemble the analysis DataFrame with all variables.

    Columns returned
    ----------------
    community_id, linkage_density, capability_score, capability_governance,
    capability_enrichment, capability_reuse, n_products, n_countries,
    region, discipline_diversity, oa_share, is_eu_horizon, betweenness,
    eigenvector, modularity_class
    """
    products = pl.read_parquet(processed_dir / "publication.parquet")
    relations_path = processed_dir / "relation.parquet"
    relations = (
        pl.read_parquet(relations_path)
        if relations_path.exists()
        else pl.DataFrame({"source": [], "target": [], "rel_type": []})
    )
    communities_path = processed_dir / "community.parquet"
    communities = (
        pl.read_parquet(communities_path)
        if communities_path.exists()
        else pl.DataFrame({"community_id": [], "label": [], "description": []})
    )
    projects_path = processed_dir / "project.parquet"
    projects = (
        pl.read_parquet(projects_path)
        if projects_path.exists()
        else pl.DataFrame({"project_id": [], "is_horizon": []})
    )

    gov_bodies = load_governance_bodies()

    # --- Community-level product aggregates ----------------------------------
    long = (
        products.select(["product_id", "contexts", "countries", "oa", "subjects"])
        .explode("contexts")
        .drop_nulls("contexts")
        .rename({"contexts": "community_id"})
    )
    product_agg = long.group_by("community_id").agg(
        pl.col("product_id").n_unique().alias("n_products"),
        pl.col("countries").list.explode().unique().alias("all_countries"),
        pl.col("oa").mean().alias("oa_share"),
        pl.col("subjects").list.explode().unique().alias("all_subjects"),
    )

    # --- Governance participation -------------------------------------------
    if communities.is_empty():
        gov_scores = pl.DataFrame(
            {"community_id": product_agg["community_id"], "capability_governance": [0] * product_agg.height}
        )
    else:
        gov_scores = communities.with_columns(
            pl.struct(["label", "description"])
            .map_elements(
                lambda s: _governance_score(
                    str(s["label"] or ""), str(s["description"] or ""), gov_bodies
                ),
                return_dtype=pl.Int64,
            )
            .alias("capability_governance")
        ).select(["community_id", "capability_governance"])

    # --- Enrichment: relations originating from community products ----------
    product_to_comm = long.select(["product_id", "community_id"])
    rel_from = (
        relations.join(product_to_comm, left_on="source", right_on="product_id", how="inner")
        .group_by("community_id")
        .agg(pl.len().alias("capability_enrichment"))
    )

    # --- Reuse: incoming relations (someone cited/related-to community output)
    rel_to = (
        relations.join(product_to_comm, left_on="target", right_on="product_id", how="inner")
        .group_by("community_id")
        .agg(pl.len().alias("capability_reuse"))
    )

    # --- EU Horizon flag via project relations -------------------------------
    horizon_projects = (
        projects.filter(pl.col("is_horizon"))["project_id"].to_list()
        if not projects.is_empty()
        else []
    )
    if horizon_projects:
        horizon_products = (
            relations.filter(
                pl.col("target").is_in(horizon_projects) | pl.col("source").is_in(horizon_projects)
            )
            .select([pl.col("source").alias("product_id")])
            .vstack(
                relations.filter(
                    pl.col("target").is_in(horizon_projects)
                    | pl.col("source").is_in(horizon_projects)
                ).select([pl.col("target").alias("product_id")])
            )
            .unique()
        )
        horizon_comm = (
            product_to_comm.join(horizon_products, on="product_id", how="inner")["community_id"]
            .unique()
            .to_list()
        )
    else:
        horizon_comm = []

    # --- Assemble ------------------------------------------------------------
    df = (
        product_agg
        .join(gov_scores, on="community_id", how="left")
        .join(rel_from, on="community_id", how="left")
        .join(rel_to, on="community_id", how="left")
        .join(network_stats, on="community_id", how="left")
        .with_columns(
            pl.col("capability_governance").fill_null(0),
            pl.col("capability_enrichment").fill_null(0),
            pl.col("capability_reuse").fill_null(0),
            pl.col("linkage_density").fill_null(0.0),
            pl.col("all_countries").list.len().alias("n_countries"),
            pl.col("all_subjects").list.len().alias("discipline_diversity"),
            pl.col("community_id").is_in(horizon_comm).alias("is_eu_horizon"),
        )
        .with_columns(
            pl.col("all_countries")
            .map_elements(_classify_region, return_dtype=pl.Utf8)
            .alias("region")
        )
        .drop(["all_countries", "all_subjects"])
    )

    # Composite capability score (standardised z-score sum)
    scaler = StandardScaler()
    cap_matrix = df.select(
        ["capability_governance", "capability_enrichment", "capability_reuse"]
    ).to_numpy()
    z = scaler.fit_transform(cap_matrix)
    df = df.with_columns(pl.Series("capability_score", z.sum(axis=1)))

    if mode == "eu":
        df_out = df.filter(pl.col("is_eu_horizon"))
        if df_out.height < 10:
            logger.warning("EU-only mode left %d rows; keeping full set.", df_out.height)
            df_out = df
    else:
        df_out = df

    out = results_dir / "capability_scores.parquet"
    df_out.write_parquet(out)
    logger.info("Wrote %s (%d rows)", out, df_out.height)
    return df_out
