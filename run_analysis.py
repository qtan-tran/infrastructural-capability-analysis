"""Command-line entry point for the infrastructural capability analysis pipeline.

Examples
--------
>>> python run_analysis.py --data-source dump --mode eu --sample 500 --causal ols
>>> python run_analysis.py --data-source dump --mode global --sample 10000 --causal psm
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from src.analysis.build_network import build_linkage_network
from src.analysis.causal_analysis import run_causal_pipeline
from src.analysis.compute_variables import compute_all_variables
from src.analysis.visualize import generate_all_figures
from src.data_fetch.download_dumps import download_openaire_dump
from src.data_fetch.parse_openaire import parse_dump_to_parquet
from src.utils.logging import get_logger

app = typer.Typer(
    name="infrastructural-capability-analysis",
    help="Pipeline for testing the linkage-density -> collective-capability hypothesis.",
    add_completion=False,
)
logger = get_logger(__name__)


@app.command()
def main(
    data_source: Annotated[
        str, typer.Option("--data-source", help="Data source: 'dump' (Zenodo bulk) or 'api' (live).")
    ] = "dump",
    mode: Annotated[
        str, typer.Option("--mode", help="Scope: 'eu' (Horizon-funded) or 'global'.")
    ] = "eu",
    sample: Annotated[
        int, typer.Option("--sample", help="Number of communities to analyse (stratified).")
    ] = 500,
    causal: Annotated[
        str, typer.Option("--causal", help="Causal method: 'ols', 'psm', or 'iv'.")
    ] = "ols",
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 42,
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Directory for cached parquet.")
    ] = Path("data"),
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Directory for output artifacts.")
    ] = Path("results"),
    skip_download: Annotated[
        bool, typer.Option("--skip-download", help="Reuse already-cached parquet.")
    ] = False,
) -> None:
    """Run the full analysis pipeline end-to-end."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(exist_ok=True)

    logger.info("=" * 72)
    logger.info("Infrastructural Capability Analysis -- starting pipeline")
    logger.info(
        "source=%s mode=%s sample=%d causal=%s seed=%d",
        data_source, mode, sample, causal, seed,
    )
    logger.info("=" * 72)

    # 1. Acquire data
    if not skip_download and data_source == "dump":
        dump_paths = download_openaire_dump(target_dir=data_dir / "raw")
        parse_dump_to_parquet(dump_paths, out_dir=data_dir / "processed", sample=sample, mode=mode)
    elif data_source == "api":
        logger.warning(
            "Live API mode is intended for small refreshes only. "
            "Falling back to cached parquet if present."
        )

    processed = data_dir / "processed"
    if not processed.exists():
        raise SystemExit(
            f"No processed parquet found at {processed}. Run without --skip-download first."
        )

    # 2. Build network + compute network-level independent variable
    network_stats = build_linkage_network(
        processed_dir=processed, results_dir=results_dir, seed=seed
    )

    # 3. Compute capability score (dependent) + controls
    analysis_df = compute_all_variables(
        processed_dir=processed,
        network_stats=network_stats,
        results_dir=results_dir,
        mode=mode,
    )

    # 4. Causal estimation
    causal_results = run_causal_pipeline(
        df=analysis_df,
        method=causal,
        results_dir=results_dir,
        seed=seed,
    )

    # 5. Figures
    generate_all_figures(
        analysis_df=analysis_df,
        causal_results=causal_results,
        results_dir=results_dir,
    )

    # 6. Provenance
    provenance = {
        "data_source": data_source,
        "mode": mode,
        "sample": sample,
        "causal": causal,
        "seed": seed,
        "n_communities": int(len(analysis_df)),
    }
    (results_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    logger.info("Pipeline complete. Outputs in %s", results_dir.resolve())


if __name__ == "__main__":
    app()
