"""Build the bipartite community <-> metadata-relations network.

Prefers ``igraph`` for performance. Falls back to ``networkx`` if
igraph is unavailable at import time (CI/Windows edge cases).
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import igraph as ig

    _HAS_IGRAPH = True
except ImportError:  # pragma: no cover
    _HAS_IGRAPH = False
    logger.warning("python-igraph not available; falling back to NetworkX (slower).")


def _products_long(products: pl.DataFrame) -> pl.DataFrame:
    """Explode products to (product_id, community_id) long format."""
    return (
        products.select(["product_id", "contexts"])
        .explode("contexts")
        .drop_nulls("contexts")
        .rename({"contexts": "community_id"})
    )


def build_linkage_network(
    processed_dir: Path,
    results_dir: Path,
    seed: int = 42,
) -> pl.DataFrame:
    """Build the bipartite graph and compute per-community linkage metrics.

    Returns a DataFrame with one row per community.
    """
    products_path = processed_dir / "publication.parquet"
    relations_path = processed_dir / "relation.parquet"
    if not products_path.exists():
        raise FileNotFoundError(f"Missing {products_path}")
    products = pl.read_parquet(products_path)

    long = _products_long(products)
    logger.info(
        "Products-community edges: %d (communities=%d, products=%d)",
        long.height,
        long["community_id"].n_unique(),
        long["product_id"].n_unique(),
    )

    relations = (
        pl.read_parquet(relations_path)
        if relations_path.exists()
        else pl.DataFrame({"source": [], "target": [], "rel_type": []})
    )
    # Only count relations whose source OR target is a product in our sample
    product_ids = set(products["product_id"].to_list())
    rel_filtered = relations.filter(
        pl.col("source").is_in(product_ids) | pl.col("target").is_in(product_ids)
    )

    # Attach community from either endpoint by joining back
    prod_to_comm = (
        long.group_by("product_id")
        .agg(pl.col("community_id").alias("communities"))
        .to_dict(as_series=False)
    )
    p2c: dict[str, list[str]] = dict(
        zip(prod_to_comm["product_id"], prod_to_comm["communities"], strict=False)
    )

    rel_rows: list[dict] = []
    for row in rel_filtered.iter_rows(named=True):
        communities = set(p2c.get(row["source"], [])) | set(p2c.get(row["target"], []))
        for c in communities:
            rel_rows.append({"community_id": c, "rel_type": row["rel_type"]})
    rel_long = pl.from_dicts(rel_rows) if rel_rows else pl.DataFrame(
        {"community_id": [], "rel_type": []}
    )

    # Per-community metrics
    per_community = (
        long.group_by("community_id")
        .agg(
            pl.col("product_id").n_unique().alias("n_products"),
        )
        .join(
            rel_long.group_by("community_id").agg(pl.len().alias("n_relations")),
            on="community_id",
            how="left",
        )
        .with_columns(pl.col("n_relations").fill_null(0))
        .with_columns(
            (pl.col("n_relations") / pl.col("n_products").clip(lower_bound=1)).alias(
                "linkage_density"
            )
        )
    )

    # Centrality via igraph or networkx on community-community projection
    centrality = _centrality_igraph(long) if _HAS_IGRAPH else _centrality_nx(long)

    result = per_community.join(centrality, on="community_id", how="left").with_columns(
        pl.col("betweenness").fill_null(0.0),
        pl.col("eigenvector").fill_null(0.0),
        pl.col("modularity_class").fill_null(-1),
    )

    out = results_dir / "network_stats.parquet"
    result.write_parquet(out)
    logger.info("Wrote %s (%d communities)", out, result.height)
    return result


def _centrality_igraph(long: pl.DataFrame) -> pl.DataFrame:
    """Project products->communities and compute centralities in igraph."""
    pairs = long.group_by("community_id").agg(pl.col("product_id")).to_dict(as_series=False)
    comm_ids = pairs["community_id"]
    comm_products = [set(ps) for ps in pairs["product_id"]]

    # Build community-community edges when they share >=1 product
    edges: list[tuple[int, int]] = []
    for i in range(len(comm_ids)):
        for j in range(i + 1, len(comm_ids)):
            if comm_products[i] & comm_products[j]:
                edges.append((i, j))

    g = ig.Graph(n=len(comm_ids), edges=edges, directed=False)
    g.vs["name"] = comm_ids

    betweenness = g.betweenness()
    # Eigenvector can fail on disconnected graphs; guard it.
    try:
        eigen = g.eigenvector_centrality()
    except Exception:  # pragma: no cover
        eigen = [0.0] * len(comm_ids)
    try:
        clusters = g.community_multilevel().membership
    except Exception:  # pragma: no cover
        clusters = [-1] * len(comm_ids)

    return pl.DataFrame(
        {
            "community_id": comm_ids,
            "betweenness": betweenness,
            "eigenvector": eigen,
            "modularity_class": clusters,
        }
    )


def _centrality_nx(long: pl.DataFrame) -> pl.DataFrame:
    """NetworkX fallback -- slower but portable."""
    pairs = long.group_by("community_id").agg(pl.col("product_id")).to_dict(as_series=False)
    comm_ids = pairs["community_id"]
    comm_products = [set(ps) for ps in pairs["product_id"]]

    graph = nx.Graph()
    graph.add_nodes_from(comm_ids)
    for i, ci in enumerate(comm_ids):
        for j in range(i + 1, len(comm_ids)):
            if comm_products[i] & comm_products[j]:
                graph.add_edge(ci, comm_ids[j])

    betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() < 5000 else {n: 0.0 for n in graph}
    try:
        eigen = nx.eigenvector_centrality_numpy(graph, max_iter=500)
    except Exception:  # pragma: no cover
        eigen = {n: 0.0 for n in graph}
    communities = nx.community.greedy_modularity_communities(graph) if graph.number_of_edges() else []
    classmap: dict[str, int] = {}
    for k, grp in enumerate(communities):
        for node in grp:
            classmap[node] = k

    return pl.DataFrame(
        {
            "community_id": comm_ids,
            "betweenness": [betweenness.get(c, 0.0) for c in comm_ids],
            "eigenvector": [eigen.get(c, 0.0) for c in comm_ids],
            "modularity_class": [classmap.get(c, -1) for c in comm_ids],
        }
    )
