"""Crossref polite-pool enrichment for DOIs discovered in OpenAIRE.

Uses aiohttp with a bounded semaphore. The mailto parameter is required
for Crossref's polite pool and is read from ``CONTACT_EMAIL``.

References
----------
https://api.crossref.org/swagger-ui/index.html (polite pool section)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiohttp
import polars as pl
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

CROSSREF_BASE = "https://api.crossref.org/works"
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "anonymous@example.com")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))


async def _fetch_doi(session: aiohttp.ClientSession, sem: asyncio.Semaphore, doi: str) -> dict:
    """Fetch a single DOI record from Crossref, respecting the polite pool."""
    url = f"{CROSSREF_BASE}/{doi}"
    params = {"mailto": CONTACT_EMAIL}
    async with sem:
        try:
            async with session.get(url, params=params, timeout=30) as r:
                if r.status != 200:
                    return {"doi": doi, "status": r.status}
                data = await r.json()
                msg = data.get("message", {})
                return {
                    "doi": doi,
                    "status": 200,
                    "is_referenced_by_count": msg.get("is-referenced-by-count", 0),
                    "references_count": msg.get("references-count", 0),
                    "funder": [f.get("name") for f in msg.get("funder", []) or []],
                    "license": [lic.get("URL") for lic in msg.get("license", []) or []],
                }
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.debug("Crossref error for %s: %s", doi, e)
            return {"doi": doi, "status": -1, "error": str(e)}


async def enrich_dois(dois: list[str]) -> pl.DataFrame:
    """Asynchronously enrich a list of DOIs."""
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    headers = {"User-Agent": f"infrastructural-capability-analysis (mailto:{CONTACT_EMAIL})"}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [_fetch_doi(session, sem, d) for d in dois]
        results = await tqdm_asyncio.gather(*tasks, desc="crossref")
    return pl.from_dicts(results)


def enrich_from_parquet(
    products_parquet: Path,
    out_parquet: Path,
    limit: int | None = None,
) -> Path:
    """Synchronous wrapper -- enrich DOIs present in a products parquet file."""
    df = pl.read_parquet(products_parquet)
    dois = df.filter(pl.col("doi").is_not_null())["doi"].unique().to_list()
    if limit:
        dois = dois[:limit]
    logger.info("Enriching %d DOIs from %s", len(dois), products_parquet)
    enriched = asyncio.run(enrich_dois(dois))
    enriched.write_parquet(out_parquet)
    return out_parquet
