"""Download OpenAIRE Research Graph bulk dumps from Zenodo.

The module prefers the versioned DOI pinned in the environment
(``OPENAIRE_DUMP_DOI``) and caches tarballs under ``data/raw/``.
JSONL extraction is streamed to avoid memory blowups.

References
----------
OpenAIRE Research Graph Zenodo community:
    https://zenodo.org/communities/openaire-research-graph
"""
from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path
from collections.abc import Iterable

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

ZENODO_API = "https://zenodo.org/api/records"
DEFAULT_DOI = os.getenv("OPENAIRE_DUMP_DOI", "10.5281/zenodo.13430566")

DEFAULT_ENTITIES = (
    "publications",
    "projects",
    "datasets",
    "software",
    "organizations",
    "relations",
    "communities",
)


def _record_id_from_doi(doi: str) -> str:
    """Extract the Zenodo record id from a DOI like ``10.5281/zenodo.13430566``."""
    return doi.rsplit(".", 1)[-1]


def _fetch_record_metadata(doi: str) -> dict:
    """Query Zenodo for record metadata given a DOI."""
    record_id = _record_id_from_doi(doi)
    url = f"{ZENODO_API}/{record_id}"
    logger.info("Fetching Zenodo record metadata: %s", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream a file to disk with a progress bar and SHA-256 verification hook."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Skipping existing file: %s", dest)
        return dest

    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        hasher = hashlib.sha256()
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for block in r.iter_content(chunk):
                if not block:
                    continue
                f.write(block)
                hasher.update(block)
                pbar.update(len(block))
        logger.info("SHA256 %s = %s", dest.name, hasher.hexdigest()[:16])
    return dest


def _matches_entity(filename: str, entities: Iterable[str]) -> bool:
    name = filename.lower()
    return any(ent in name for ent in entities)


def download_openaire_dump(
    doi: str = DEFAULT_DOI,
    target_dir: Path = Path("data/raw"),
    entities: Iterable[str] = DEFAULT_ENTITIES,
) -> list[Path]:
    """Download the subset of tarballs that match the requested entities.

    Parameters
    ----------
    doi : str
        Zenodo DOI to pin. Override via env var ``OPENAIRE_DUMP_DOI``.
    target_dir : Path
        Directory to cache downloads.
    entities : Iterable[str]
        Substrings to filter filenames (e.g. ``publications``).

    Returns
    -------
    list[Path]
        Local paths of downloaded tar archives (or extracted directories).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    meta = _fetch_record_metadata(doi)
    files = meta.get("files", [])
    if not files:
        raise RuntimeError(f"No files listed on Zenodo record {doi}. Check DOI.")

    selected = [f for f in files if _matches_entity(f["key"], entities)]
    if not selected:
        logger.warning(
            "No files matched entities=%s; falling back to full record (first 3 files).",
            entities,
        )
        selected = files[:3]

    downloaded: list[Path] = []
    for f in selected:
        url = f["links"]["self"]
        dest = target_dir / f["key"]
        downloaded.append(_download_file(url, dest))
    logger.info("Downloaded %d dump files.", len(downloaded))
    return downloaded


def extract_tar(archive: Path, out_dir: Path) -> list[Path]:
    """Extract a tar archive and return the list of JSONL files inside."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not tarfile.is_tarfile(archive):
        logger.info("%s is not a tar archive; returning as-is.", archive)
        return [archive]

    extracted: list[Path] = []
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith((".json", ".jsonl", ".gz")):
                tf.extract(member, out_dir, filter="data")
                extracted.append(out_dir / member.name)
    logger.info("Extracted %d files from %s", len(extracted), archive.name)
    return extracted


if __name__ == "__main__":
    paths = download_openaire_dump()
    for p in paths:
        extract_tar(p, Path("data/raw/extracted"))
