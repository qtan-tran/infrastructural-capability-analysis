"""Lightweight IO helpers shared across modules."""
from __future__ import annotations

from pathlib import Path

import polars as pl

_GOV_CSV = Path(__file__).parent / "governance_bodies.csv"


def load_governance_bodies() -> pl.DataFrame:
    """Load the curated list of infrastructural governance bodies."""
    if not _GOV_CSV.exists():
        return pl.DataFrame({"name": [], "category": [], "url": []})
    return pl.read_csv(_GOV_CSV)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
