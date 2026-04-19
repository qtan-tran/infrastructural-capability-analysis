"""Tests with mocked HTTP calls and a synthetic OpenAIRE mini-dump.

Run with: poetry run pytest
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest
import responses
from aioresponses import aioresponses

from src.analysis.build_network import build_linkage_network
from src.analysis.causal_analysis import run_causal_pipeline
from src.analysis.compute_variables import compute_all_variables
from src.data_fetch.crossref_enrich import enrich_dois as crossref_enrich
from src.data_fetch.datacite_enrich import enrich_dois as datacite_enrich
from src.data_fetch.download_dumps import _fetch_record_metadata
from src.data_fetch.parse_openaire import (
    parse_communities,
    parse_products,
    parse_projects,
    parse_relations,
)

# ---------------------------------------------------------------------------
# Synthetic dump fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dump(tmp_path: Path) -> dict[str, Path]:
    """Create a tiny JSONL corpus mimicking OpenAIRE structure."""
    pubs = [
        {
            "id": "pub1",
            "title": [{"value": "Open Science metadata paper"}],
            "resourceType": "publication",
            "publicationYear": 2023,
            "bestAccessRight": {"code": "OPEN"},
            "context": [{"id": "commA"}, {"id": "commB"}],
            "instance": [{"affiliation": [{"countryCode": "DE"}]}],
            "subject": [{"value": "information science"}],
            "pid": [{"scheme": "doi", "value": "10.1234/abc"}],
            "reference": [],
        },
        {
            "id": "pub2",
            "title": [{"value": "Global South repositories"}],
            "resourceType": "publication",
            "publicationYear": 2024,
            "bestAccessRight": {"code": "CLOSED"},
            "context": [{"id": "commB"}],
            "instance": [{"affiliation": [{"countryCode": "BR"}]}],
            "subject": [{"value": "library science"}],
            "pid": [{"scheme": "doi", "value": "10.1234/def"}],
            "reference": [],
        },
    ]
    projects = [
        {"id": "proj1", "title": "Horizon test", "funding": [{"funder": {"name": "EC", "jurisdiction": "EU"}}],
         "acronym": "HTEST", "startDate": "2020-01-01", "endDate": "2024-12-31"}
    ]
    rels = [
        {"source": "pub1", "target": "pub2", "relType": "cites"},
        {"source": "pub1", "target": "proj1", "relType": "isProducedBy"},
    ]
    comms = [
        {"id": "commA", "label": "OpenAIRE DataCite community", "description": "Uses ORCID and DataCite services"},
        {"id": "commB", "label": "Generic community", "description": "No governance mentions."},
    ]

    def _write(name: str, rows: list[dict]) -> Path:
        p = tmp_path / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    def _write_gz(name: str, rows: list[dict]) -> Path:
        p = tmp_path / f"{name}.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    return {
        "publication": _write("publication", pubs),
        "project": _write("project", projects),
        "relation": _write_gz("relation", rels),  # exercise the gzip path
        "community": _write("community", comms),
    }


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

def test_parse_products(synthetic_dump):
    df = parse_products(synthetic_dump["publication"])
    assert df.height == 2
    assert set(df.columns) >= {"product_id", "contexts", "countries", "oa", "doi"}
    assert df.filter(pl.col("product_id") == "pub1")["oa"].item() is True


def test_parse_projects(synthetic_dump):
    df = parse_projects(synthetic_dump["project"])
    assert df.height == 1
    assert df["is_horizon"].item() is True or df["funder"].item() == "EC"


def test_parse_relations_gzip(synthetic_dump):
    df = parse_relations(synthetic_dump["relation"])
    assert df.height == 2
    assert set(df["rel_type"].to_list()) == {"cites", "isProducedBy"}


def test_parse_communities(synthetic_dump):
    df = parse_communities(synthetic_dump["community"])
    assert df.height == 2
    assert "commA" in df["community_id"].to_list()


# ---------------------------------------------------------------------------
# Network & variables
# ---------------------------------------------------------------------------

@pytest.fixture
def processed_dir(tmp_path, synthetic_dump):
    out = tmp_path / "processed"
    out.mkdir()
    parse_products(synthetic_dump["publication"]).write_parquet(out / "publication.parquet")
    parse_projects(synthetic_dump["project"]).write_parquet(out / "project.parquet")
    parse_relations(synthetic_dump["relation"]).write_parquet(out / "relation.parquet")
    parse_communities(synthetic_dump["community"]).write_parquet(out / "community.parquet")
    return out


def test_build_network(tmp_path, processed_dir):
    results = tmp_path / "results"
    results.mkdir()
    stats = build_linkage_network(processed_dir, results, seed=0)
    assert stats.height == 2
    assert "linkage_density" in stats.columns
    assert (stats["linkage_density"] >= 0).all()


def test_compute_variables_governance_detected(tmp_path, processed_dir):
    results = tmp_path / "results"
    results.mkdir()
    stats = build_linkage_network(processed_dir, results, seed=0)
    df = compute_all_variables(processed_dir, stats, results, mode="global")
    assert df.height >= 1
    # commA explicitly mentions ORCID + DataCite -> governance > 0
    commA = df.filter(pl.col("community_id") == "commA")
    if commA.height:
        assert commA["capability_governance"].item() >= 1


def test_causal_pipeline_smoke(tmp_path, processed_dir):
    results = tmp_path / "results"
    results.mkdir()
    stats = build_linkage_network(processed_dir, results, seed=0)
    df = compute_all_variables(processed_dir, stats, results, mode="global")
    # Need >= 4 rows for OLS; duplicate if needed
    if df.height < 10:
        df = pl.concat([df] * 6, how="vertical")
    out = run_causal_pipeline(df, method="psm", results_dir=results, seed=0)
    assert "ols" in out
    assert "params" in out["ols"]


# ---------------------------------------------------------------------------
# HTTP mocks
# ---------------------------------------------------------------------------

@responses.activate
def test_zenodo_metadata_mock():
    responses.add(
        responses.GET,
        "https://zenodo.org/api/records/13430566",
        json={"files": [{"key": "publications.tar", "links": {"self": "https://example.com/p.tar"}}]},
        status=200,
    )
    meta = _fetch_record_metadata("10.5281/zenodo.13430566")
    assert meta["files"][0]["key"] == "publications.tar"


@pytest.mark.asyncio
async def test_crossref_enrich_mock():
    with aioresponses() as m:
        m.get(
            "https://api.crossref.org/works/10.1234/abc?mailto=anonymous@example.com",
            payload={"message": {"is-referenced-by-count": 5, "references-count": 10}},
        )
        df = await crossref_enrich(["10.1234/abc"])
        assert df["is_referenced_by_count"].to_list()[0] == 5


@pytest.mark.asyncio
async def test_datacite_enrich_mock():
    with aioresponses() as m:
        m.get(
            "https://api.datacite.org/dois/10.1234/xyz",
            payload={
                "data": {
                    "attributes": {
                        "viewCount": 3,
                        "downloadCount": 7,
                        "citationCount": 2,
                        "relatedIdentifiers": [],
                    }
                }
            },
        )
        df = await datacite_enrich(["10.1234/xyz"])
        assert df["download_count"].to_list()[0] == 7
