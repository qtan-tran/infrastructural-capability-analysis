# SCALABILITY.md — From a Colab-ready prototype to a petabyte-class pipeline

This document explains *why* the current pipeline works at sample sizes of
~10,000 communities on a laptop, and *how* to extend it for full-corpus
analysis (tens of millions of records, panel data across multiple dump
years, or federated cross-infrastructure analysis).

The guidance is layered. **Phase 1** is what the repository ships with;
phases 2–5 are an expansion roadmap.

---

## Phase 1 — What the current pipeline already does (shipped)

| Concern | Current choice | Why it scales |
|---|---|---|
| Data acquisition | Zenodo bulk dumps, pinned by DOI | No live-API rate limits; deterministic; cacheable |
| Data format on disk | Parquet (columnar, compressed) | 5–10× smaller than JSON; predicate pushdown |
| JSONL parsing | Streaming (`gzip.open` + line iteration) | Memory stays O(1) regardless of file size |
| DataFrames | **Polars** (Rust, multi-threaded) | 3–10× faster than pandas; lazy mode available |
| Graphs | **igraph** (C backend), NetworkX fallback | Handles millions of edges without blowing up |
| Concurrency | `asyncio.Semaphore` bounded aiohttp | Polite but saturates CPU on I/O |
| Reproducibility | Poetry + Docker + seed + `provenance.json` | Byte-for-byte reruns |
| CI | GitHub Actions, Python 3.11 + 3.12 | Offline-safe (all HTTP mocked) |

Realistic ceilings on a 16 GB laptop:
- **~15 M products, ~50 M relations, ~20 k communities** before swap thrashing.
- Beyond that, move to Phase 2.

---

## Phase 2 — Lazy & out-of-core execution (same machine, 10× bigger data)

**Goal:** let a single workstation process a full OpenAIRE dump (~100 GB
decompressed) without OOM.

### 2.1 Switch Polars to `LazyFrame`

Replace eager reads with lazy scans in `parse_openaire.py` and
`compute_variables.py`:

```python
# Before
df = pl.read_parquet("publication.parquet")
agg = df.group_by("community_id").agg(...)

# After
agg = (
    pl.scan_parquet("publication.parquet")
    .group_by("community_id")
    .agg(...)
    .collect(streaming=True)  # spills to disk as needed
)
```

Polars' streaming engine executes group-bys and joins in chunks of
configurable size (`POLARS_STREAMING_CHUNK_SIZE`) and spills to disk.

### 2.2 Partition Parquet by year and entity

Rewrite the processed layout as a Hive-style directory:

```
data/processed/
├── publication/year=2020/part-0.parquet
├── publication/year=2021/part-0.parquet
├── relation/year=2020/part-0.parquet
└── ...
```

`pl.scan_parquet("data/processed/publication/**/*.parquet")` then enables
partition pruning — a "2023-only" query reads only the 2023 partition.

### 2.3 Enable Dask as an optional backend

The `heavy` extra in `pyproject.toml` already includes Dask. Add a
`--backend {polars,dask}` CLI flag and, when `dask`, use
`dask.dataframe.read_parquet` with `blocksize="256MB"`. Dask handles
larger-than-RAM computations natively and supports distributed schedulers
(LocalCluster, SLURM, K8s) with a one-line swap.

### 2.4 Graph-out-of-core

For graphs beyond ~10 M edges:
- Use igraph's `read_graph_edgelist` with memory-mapped edge lists,
  OR
- Switch the community-projection step to a GPU library: **cuGraph**
  (NVIDIA RAPIDS) computes betweenness on 100 M-edge graphs in seconds.

Add an optional extra:

```toml
[tool.poetry.extras]
gpu = ["cugraph", "cudf"]
```

---

## Phase 3 — Cloud & distributed execution (10 × bigger again)

### 3.1 Object storage

Drop the local `data/` directory in favour of S3/GCS/Azure Blob.
Polars, Dask, and DuckDB all read Parquet directly from object storage:

```python
pl.scan_parquet("s3://openaire-dumps/publication/**/*.parquet", storage_options={...})
```

Add a thin `src/utils/storage.py` abstraction so `Path` vs
`s3://`/`gs://` is transparent.

### 3.2 DuckDB as the SQL engine

For aggregate queries that are painful to express in Polars
(multi-join, window functions across communities × years × subjects),
run DuckDB directly over the Parquet lake:

```python
import duckdb
duckdb.sql("""
    SELECT community_id, year, COUNT(*) as n_rel
    FROM read_parquet('s3://.../relation/**/*.parquet')
    GROUP BY community_id, year
""").pl()
```

DuckDB runs out-of-core on a single box and has
join-order/predicate-pushdown smarter than either pandas or Polars.

### 3.3 Airflow / Prefect orchestration

Wrap each CLI step (download → parse → build_network → compute_variables
→ causal) as a task in `prefect` or `airflow`:

```
dag = Flow("ica-pipeline")
download >> parse >> network >> variables >> causal >> figures
```

Benefits: incremental runs, per-task retries, lineage, scheduled
re-runs when a new Zenodo dump appears.

### 3.4 Cluster execution

- **Dask on Coiled / AWS Fargate** for burst compute.
- **Ray** if you want to parallelise the bootstrap step of PSM (500
  resamples → trivially parallelisable across workers).

---

## Phase 4 — Incremental / streaming updates

Today, each run reprocesses the full dump. For a production observatory:

### 4.1 Delta Lake or Apache Iceberg

Store `data/processed/` as a Delta or Iceberg table. New Zenodo
releases become appended snapshots with time-travel queries:

```python
# Query the state of the graph as of dump vYYYY-MM-01
pl.scan_delta("s3://.../publication", version="2025-03-01")
```

### 4.2 Change Data Capture via OpenAIRE Graph API

The [OpenAIRE Graph API](https://graph.openaire.eu/docs/apis/graph-api/)
supports incremental pulls (`dateOfCollection`). Add a nightly
`incremental_refresh.py` that:

1. Queries records modified since the last run.
2. Upserts into the Delta/Iceberg table.
3. Invalidates only the affected partitions downstream.

### 4.3 Streaming relations

For infrastructures that expose Kafka or webhooks (Crossref event data,
DataCite Event Data), pipe new relations into a streaming job
(Flink / Kafka Streams / Materialize) that maintains rolling linkage
densities in near-real-time.

---

## Phase 5 — Federation & cross-infrastructure analysis

The hypothesis is *inherently* about multiple infrastructures. Scaling
beyond OpenAIRE:

### 5.1 Additional sources to ingest

| Source | Value add | Integration note |
|---|---|---|
| OpenAlex | Broader coverage (200 M works) | Monthly snapshots on S3 Requester Pays |
| Crossref full corpus | Citation graph | Polite-pool sampled dumps + Event Data |
| DataCite full dump | Dataset/software relations | Monthly JSON on S3 |
| ROR | Organisation identifiers | Small (<100 MB) JSON dumps |
| OpenCitations COCI | Open citation index | Parquet on FigShare |
| SciNoBo / SciSciNet | Enriched topic labels | Zenodo |
| ORCID public data file | Author-level governance roles | Annual JSON |

### 5.2 Entity resolution

A single publication may appear with slightly different IDs in OpenAIRE
vs OpenAlex vs Crossref. Add a dedicated module
`src/data_fetch/entity_resolve.py` that:
- Normalises DOIs (lowercase, strip `https://doi.org/`)
- Uses ROR for organisation matching
- Uses ORCID for author matching
- Falls back to `dedupe` / `splink` for fuzzy matching

### 5.3 Federated DAG

With multiple sources, the causal DAG grows an explicit
**infrastructure node** per source, allowing analyses like
"communities rich on OpenAIRE but poor on Crossref" — useful for
identifying who is *captured* by one infrastructure vs. *polycentric*.

### 5.4 Governance detection beyond a CSV

`governance_bodies.csv` is a hand-curated seed list. At scale, replace
substring matching with:

- A **fine-tuned NER model** (spaCy or a small BERT) trained on a
  labelled set of community descriptions, committee minutes, GitHub
  org descriptions.
- **Knowledge-graph linking** to Wikidata: resolve detected mentions
  to QIDs and query Wikidata's organisation hierarchy for governance
  membership.
- **GitHub API traversal** (via `pygithub`, already in deps) to verify
  that community members actually commit to infrastructure repos
  (a much harder behavioural proxy than "mentions DataCite").

---

## Phase 6 — Performance ceiling & benchmarks (optional aspiration)

Aspirational targets for a mature observatory:

| Metric | Prototype (now) | Phase 2 | Phase 3 | Phase 5 |
|---|---|---|---|---|
| Products | 10 M | 100 M | 1 B | 5 B |
| Relations | 50 M | 500 M | 5 B | 20 B |
| End-to-end runtime | 1 h | 2 h (lazy) | 20 min (cluster) | 1 h (federated) |
| Incremental update | full rerun | per-partition | per-record | per-event stream |
| Hardware | laptop | 64 GB workstation | 8× c6i.8xlarge | Spark/Dask cluster |

Each phase is additive and backward-compatible: the Phase 1 CLI
(`python run_analysis.py --sample 500 --causal ols`) continues to work
against any of the bigger backends.

---

## Recommended order for adoption

1. **First** (low effort, high payoff): switch to Polars lazy scans
   (Phase 2.1) and partitioned Parquet (Phase 2.2).
2. **Second**: add DuckDB for ad-hoc SQL and object storage (Phase 3.1–3.2).
3. **Third**: Prefect + incremental CDC (Phase 3.3, 4.2).
4. **Fourth**: cross-infrastructure federation (Phase 5) — this is where
   the *research* gets most interesting, because it lets the hypothesis
   be tested not just *within* OpenAIRE but *across competing
   infrastructures*, which is the setting the theory was originally
   articulated for.

---

## Known bottlenecks to watch

- **Community-community projection** in `build_network.py` is O(n²) in
  communities. Above 20 k communities, replace the nested loop with a
  join-based edge construction (group products, then join on
  `product_id` to get all co-occurring community pairs).
- **PSM bootstrap** (500 resamples) is single-threaded. Wrap in
  `joblib.Parallel` for trivial 8–16× speedup.
- **Governance substring match** is O(n_bodies × n_communities). For
  >100 k communities move to a trie / Aho-Corasick
  (`pyahocorasick`).

Each of these is a ~50-line refactor and should be done only when
profiling shows it's the limiting step.
