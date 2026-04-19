"""DataCite enrichment for datasets/software discovered in OpenAIRE.

DataCite's REST API does not require a mailto but we include a
descriptive User-Agent in keeping with polite-pool norms.
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

DATACITE_BASE = "https://api.datacite.org/dois"
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "anonymous@example.com")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))


async def _fetch_doi(session: aiohttp.ClientSession, sem: asyncio.Semaphore, doi: str) -> dict:
    async with sem:
        try:
            async with session.get(f"{DATACITE_BASE}/{doi}", timeout=30) as r:
                if r.status != 200:
                    return {"doi": doi, "status": r.status}
                data = await r.json()
                attrs = data.get("data", {}).get("attributes", {})
                return {
                    "doi": doi,
                    "status": 200,
                    "view_count": attrs.get("viewCount", 0),
                    "download_count": attrs.get("downloadCount", 0),
                    "citation_count": attrs.get("citationCount", 0),
                    "related_identifiers": [
                        r.get("relatedIdentifier")
                        for r in attrs.get("relatedIdentifiers", []) or []
                    ],
                    "schema_version": attrs.get("schemaVersion"),
                }
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug("DataCite error for %s: %s", doi, e)
            return {"doi": doi, "status": -1, "error": str(e)}


async def enrich_dois(dois: list[str]) -> pl.DataFrame:
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    headers = {"User-Agent": f"infrastructural-capability-analysis (mailto:{CONTACT_EMAIL})"}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [_fetch_doi(session, sem, d) for d in dois]
        results = await tqdm_asyncio.gather(*tasks, desc="datacite")
    return pl.from_dicts(results)


def enrich_from_parquet(
    products_parquet: Path,
    out_parquet: Path,
    limit: int | None = None,
) -> Path:
    df = pl.read_parquet(products_parquet)
    dois = (
        df.filter(pl.col("doi").is_not_null() & pl.col("type").is_in(["dataset", "software"]))[
            "doi"
        ]
        .unique()
        .to_list()
    )
    if limit:
        dois = dois[:limit]
    logger.info("Enriching %d DataCite DOIs from %s", len(dois), products_parquet)
    enriched = asyncio.run(enrich_dois(dois))
    enriched.write_parquet(out_parquet)
    return out_parquet
