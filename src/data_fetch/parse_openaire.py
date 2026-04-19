"""Parse OpenAIRE JSONL dumps into Polars DataFrames and cache as Parquet.

The parser is streaming (line-by-line) and handles gzipped files, so it
scales to multi-GB dumps without loading everything into memory.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _open_maybe_gzip(path: Path):
    """Return a text-mode handle whether the file is gzipped or plain."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a (possibly gzipped) JSONL file."""
    with _open_maybe_gzip(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:  # pragma: no cover
                logger.warning("Bad JSON line at %s:%d (%s)", path.name, i, e)


def _extract_contexts(rec: dict[str, Any]) -> list[str]:
    """Return community/context IDs attached to a research product."""
    contexts = rec.get("context") or rec.get("contexts") or []
    out: list[str] = []
    for c in contexts:
        if isinstance(c, dict):
            cid = c.get("id") or c.get("code") or c.get("label")
            if cid:
                out.append(str(cid))
        elif isinstance(c, str):
            out.append(c)
    return out


def _extract_countries(rec: dict[str, Any]) -> list[str]:
    """Pull country codes from affiliations / instances."""
    countries: set[str] = set()
    for inst in rec.get("instance", []) or []:
        for aff in inst.get("affiliation", []) or []:
            code = aff.get("country") or aff.get("countryCode")
            if code:
                countries.add(str(code).upper())
    for aff in rec.get("author", []) or []:
        if isinstance(aff, dict):
            for sub in aff.get("affiliation", []) or []:
                code = (sub.get("country") or {}).get("code") if isinstance(sub, dict) else None
                if code:
                    countries.add(str(code).upper())
    return sorted(countries)


def parse_products(path: Path) -> pl.DataFrame:
    """Parse a publications/datasets/software dump to a Polars DataFrame."""
    rows: list[dict[str, Any]] = []
    for rec in tqdm(_iter_jsonl(path), desc=f"parse:{path.name}"):
        pid = rec.get("id") or rec.get("openaireId")
        if not pid:
            continue
        rows.append(
            {
                "product_id": pid,
                "title": (rec.get("title") or [{}])[0].get("value", "")
                if isinstance(rec.get("title"), list)
                else rec.get("title", ""),
                "type": rec.get("resourceType") or rec.get("type") or "unknown",
                "year": rec.get("publicationYear") or rec.get("year"),
                "oa": bool(rec.get("bestAccessRight", {}).get("code") == "OPEN")
                if isinstance(rec.get("bestAccessRight"), dict)
                else bool(rec.get("isOpenAccess", False)),
                "contexts": _extract_contexts(rec),
                "countries": _extract_countries(rec),
                "subjects": [
                    (s.get("value") if isinstance(s, dict) else s)
                    for s in (rec.get("subject") or [])
                ],
                "doi": next(
                    (p.get("value") for p in (rec.get("pid") or []) if p.get("scheme") == "doi"),
                    None,
                ),
                "n_refs": len(rec.get("reference") or []),
            }
        )
    return pl.from_dicts(rows) if rows else pl.DataFrame()


def parse_projects(path: Path) -> pl.DataFrame:
    """Parse a projects dump (needed for EU Horizon filter)."""
    rows: list[dict[str, Any]] = []
    for rec in tqdm(_iter_jsonl(path), desc=f"parse:{path.name}"):
        pid = rec.get("id") or rec.get("openaireId")
        if not pid:
            continue
        funder = (rec.get("funding", [{}]) or [{}])[0]
        rows.append(
            {
                "project_id": pid,
                "funder": (funder.get("funder") or {}).get("name")
                if isinstance(funder, dict)
                else None,
                "programme": (funder.get("funder") or {}).get("jurisdiction")
                if isinstance(funder, dict)
                else None,
                "is_horizon": "horizon" in json.dumps(rec).lower()
                or "h2020" in json.dumps(rec).lower(),
                "acronym": rec.get("acronym"),
                "title": rec.get("title"),
                "start_date": rec.get("startDate"),
                "end_date": rec.get("endDate"),
            }
        )
    return pl.from_dicts(rows) if rows else pl.DataFrame()


def parse_relations(path: Path) -> pl.DataFrame:
    """Parse the relations dump: source, target, type."""
    rows: list[dict[str, Any]] = []
    for rec in tqdm(_iter_jsonl(path), desc=f"parse:{path.name}"):
        src = rec.get("source") or (rec.get("sourceEntity") or {}).get("id")
        tgt = rec.get("target") or (rec.get("targetEntity") or {}).get("id")
        if not src or not tgt:
            continue
        rows.append(
            {
                "source": src,
                "target": tgt,
                "rel_type": rec.get("relType") or rec.get("type") or "unknown",
                "subrel_type": rec.get("subRelType"),
            }
        )
    return pl.from_dicts(rows) if rows else pl.DataFrame()


def parse_communities(path: Path) -> pl.DataFrame:
    """Parse the contexts/communities dump."""
    rows: list[dict[str, Any]] = []
    for rec in tqdm(_iter_jsonl(path), desc=f"parse:{path.name}"):
        cid = rec.get("id") or rec.get("code")
        if not cid:
            continue
        rows.append(
            {
                "community_id": cid,
                "label": rec.get("label") or rec.get("name"),
                "description": rec.get("description") or "",
                "type": rec.get("type") or "context",
            }
        )
    return pl.from_dicts(rows) if rows else pl.DataFrame()


_PARSERS = {
    "publication": parse_products,
    "dataset": parse_products,
    "software": parse_products,
    "project": parse_projects,
    "relation": parse_relations,
    "community": parse_communities,
    "context": parse_communities,
    "organization": parse_products,  # light fallback; not heavily used
}


def _guess_entity(path: Path) -> str | None:
    name = path.name.lower()
    for key in _PARSERS:
        if key in name:
            return key
    return None


def parse_dump_to_parquet(
    dump_paths: list[Path],
    out_dir: Path,
    sample: int | None = None,
    mode: str = "eu",
) -> dict[str, Path]:
    """Parse every dump file and write one Parquet per entity type.

    Parameters
    ----------
    dump_paths : list[Path]
        Paths to raw dumps or already-extracted JSONL files.
    out_dir : Path
        Destination directory for parquet outputs.
    sample : int | None
        If set, randomly subsample *communities* to this size after parsing.
    mode : str
        'eu' restricts to Horizon-linked communities; 'global' keeps all.

    Returns
    -------
    dict[str, Path]
        Mapping from entity name -> parquet path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[pl.DataFrame]] = {k: [] for k in _PARSERS}

    for p in dump_paths:
        # If it's a tar, the caller should have extracted first.
        entity = _guess_entity(p)
        if not entity:
            logger.warning("Could not guess entity for %s, skipping.", p.name)
            continue
        df = _PARSERS[entity](p)
        if df.is_empty():
            continue
        buckets[entity].append(df)

    out_paths: dict[str, Path] = {}
    for entity, dfs in buckets.items():
        if not dfs:
            continue
        combined = pl.concat(dfs, how="diagonal_relaxed")

        if entity in ("publication", "dataset", "software", "community") and sample:
            if entity == "community" and combined.height > sample:
                combined = combined.sample(sample, seed=42)
                logger.info("Subsampled communities to %d", sample)

        dest = out_dir / f"{entity}.parquet"
        combined.write_parquet(dest)
        out_paths[entity] = dest
        logger.info("Wrote %s (%d rows)", dest, combined.height)

    return out_paths
