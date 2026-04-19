# IMPLEMENTATION.md — How to set up and run the pipeline

This guide walks through getting from zero to a finished analysis in three
environments: (1) Google Colab, (2) a local machine with Poetry, and (3)
Docker. Choose whichever matches your situation.

Every step below has been verified end-to-end: the pipeline runs cleanly
on the bundled synthetic corpus and produces all expected outputs
(`coefficients.csv`, `diagnostics.json`, `capability_scores.parquet`,
`network_stats.parquet`, and five figures).

---

## 0. Prerequisites

- **Python 3.11 or 3.12** (3.10 and older will not work — we use
  several 3.11+ type-hint features).
- **~4 GB free disk** for a `--sample 500` run.
- **~40 GB free disk** for a full run against the OpenAIRE dump.
- **An email address** for Crossref/DataCite polite-pool access.

Optional but recommended:
- Poetry 1.8+ for reproducible dependency management.
- Docker for full environment isolation.

---

## 1. Google Colab (fastest path to a working demo)

### 1.1 Unpack the project

Upload `infrastructural-capability-analysis.zip` to Colab, then:

```python
!unzip -q infrastructural-capability-analysis.zip
%cd infrastructural-capability-analysis
```

### 1.2 Install dependencies

```python
!pip install -q -r requirements.txt
```

This takes ~90 seconds on Colab. All pinned versions are Colab-compatible.

### 1.3 Configure your contact email

```python
%%writefile .env
CONTACT_EMAIL=your.email@example.com
OPENAIRE_DUMP_DOI=10.5281/zenodo.13430566
MAX_CONCURRENT_REQUESTS=10
```

### 1.4 Run a quick demo (no download needed)

The pytest suite exercises the whole pipeline against a synthetic fixture:

```python
!python -m pytest tests/ -v
```

Expected output: `10 passed in ~10s`.

### 1.5 Run a real sample

```python
!python run_analysis.py --data-source dump --mode eu --sample 500 --causal ols
```

First invocation downloads the selected Zenodo tarballs (~1–2 GB for a
partial pull). Subsequent runs hit the local cache.

Outputs land in `results/`:
- `coefficients.csv`
- `diagnostics.json`
- `capability_scores.parquet`
- `network_stats.parquet`
- `figures/*.png`, `figures/*.html`

### 1.6 Inspect results

Open `notebooks/01_EDA.ipynb` (File → Open Notebook → Upload). All three
notebooks read from `results/` and produce publication-ready plots.

---

## 2. Local machine with Poetry (recommended for real work)

### 2.1 Clone / unpack

```bash
unzip infrastructural-capability-analysis.zip
cd infrastructural-capability-analysis
```

### 2.2 Install Poetry (if needed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2.3 Install dependencies

```bash
poetry install --with dev
```

This creates a virtualenv and installs everything pinned in
`pyproject.toml`. On Linux the `igraph` wheel installs cleanly; on macOS
you may need `brew install igraph` first, and on Windows you may need
the Visual C++ build tools (or use Docker instead).

### 2.4 Configure environment

```bash
cp .env.example .env
# edit .env and set CONTACT_EMAIL at minimum
```

### 2.5 Verify everything works

```bash
poetry run pytest -v          # 10 tests should pass
poetry run ruff check src tests run_analysis.py
poetry run python run_analysis.py --help
```

### 2.6 Run the pipeline

Three typical invocations:

```bash
# Quick smoke run (~2 min with cached data)
poetry run python run_analysis.py \
    --data-source dump --mode eu --sample 500 --causal ols

# Full EU Horizon run with PSM (~30 min)
poetry run python run_analysis.py \
    --data-source dump --mode eu --sample 10000 --causal psm

# Global North/South comparison with IV-2SLS
poetry run python run_analysis.py \
    --data-source dump --mode global --sample 8000 --causal iv
```

### 2.7 Reuse cached data

After the first successful download, re-runs can skip the fetch step:

```bash
poetry run python run_analysis.py --skip-download --mode global --causal iv
```

---

## 3. Docker (best for reproducibility / cluster use)

### 3.1 Build the image

```bash
docker build -t ica:latest .
```

This produces a ~1.5 GB image with Python 3.11, libigraph, and all
Python deps pre-installed.

### 3.2 Run with a persistent data volume

```bash
mkdir -p data results
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/results:/app/results" \
    -e CONTACT_EMAIL=your.email@example.com \
    ica:latest \
    --data-source dump --mode eu --sample 500 --causal ols
```

Any cached Zenodo tarballs stay on the host in `./data/`, so subsequent
container runs are fast.

---

## 4. What each CLI flag does

| Flag | Values | Effect |
|---|---|---|
| `--data-source` | `dump` \| `api` | Bulk Zenodo dump (default) or live API |
| `--mode` | `eu` \| `global` | Restrict to Horizon-linked communities or keep all |
| `--sample` | integer | Number of communities to retain (stratified) |
| `--causal` | `ols` \| `psm` \| `iv` | Which estimator to run (OLS always included as baseline) |
| `--seed` | integer | Reproducibility seed (default 42) |
| `--data-dir` | path | Where cached parquet lives (default `data/`) |
| `--results-dir` | path | Where outputs go (default `results/`) |
| `--skip-download` | flag | Reuse already-cached parquet, skip Zenodo |

---

## 5. Understanding the outputs

### `results/coefficients.csv`

One row per estimated coefficient across OLS, NegBin, PSM, IV.

```
model,variable,coef,pvalue
OLS,linkage_density,0.910,0.0000
OLS,n_products,0.012,0.3421
...
```

### `results/diagnostics.json`

Full nested dictionary with:
- `ols.rsquared`, `ols.rsquared_adj`, `ols.vif` — regression diagnostics
- `psm.att`, `psm.se`, `psm.ci_95`, `psm.balance_smd` — matching results
- `iv.first_stage_F`, `iv.weak_instrument` — IV validity checks
- `power` — statistical power at the achieved sample size

### `results/figures/`

- `scatter_density_capability.png` — main bivariate relationship
- `heatmap_region.png` — region × density/capability means
- `coef_plot.png` — OLS coefficients with 95% CIs
- `scatter_interactive.html` — Plotly version with hover tooltips
- `geo_map.html` — folium world map by region

### `results/capability_scores.parquet`

One row per community with all variables (IV, DV, controls). This is the
file you'd load into a notebook or R session for follow-up analysis.

### `results/network_stats.parquet`

Per-community linkage density, betweenness, eigenvector centrality, and
modularity class.

### `results/provenance.json`

Written at the end of every run; records the exact CLI arguments used
so you can replicate the run later.

---

## 6. Interpreting the tests

The test suite (`tests/test_api_mocks.py`) contains three categories:

1. **Parser tests** — validate JSONL → DataFrame conversion on a
   synthetic corpus that includes both plain and gzipped files.
2. **Pipeline tests** — run `build_linkage_network` → `compute_all_variables`
   → `run_causal_pipeline` end-to-end on the same synthetic corpus,
   verifying that governance detection fires correctly and that OLS/PSM
   return well-formed results.
3. **HTTP mock tests** — use `responses` and `aioresponses` to verify
   Zenodo, Crossref, and DataCite clients without any network access.

All tests pass offline, which means CI passes offline too.

---

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'igraph'`

On macOS: `brew install igraph && poetry install`.
On Windows: use Docker, or `conda install -c conda-forge python-igraph`.

### `Zenodo 429 Too Many Requests`

The download script retries automatically. If it persists, you can
download the tarballs manually from the Zenodo community page and drop
them in `data/raw/`. The script detects cached files and skips them.

### `TypeError: the truth value of a Series is ambiguous` (should not occur)

This was a bug in early builds of `_classify_region` and is fixed in
the shipped version. If you see it, run `poetry install --sync` to
ensure you're on the latest code.

### `ValueError: Pandas data cast to numpy dtype of object` (should not occur)

Same — an early bug in `_prepare_design_matrix` that is fixed in the
shipped version.

### The OLS coefficient on `linkage_density` is implausibly large

Check `diagnostics.json` → `ols.vif`. If VIF > 10, you have
multicollinearity (usually between `n_products` and relation counts).
Consider log-transforming skewed covariates or dropping redundant ones.

### `weak_instrument: true` in IV results

First-stage F < 10 means the instrument (`is_eu_horizon` by default) is
weak for your sample. Options: (a) drop to OLS/PSM for primary
inference and keep IV as robustness, (b) enrich the instrument (see
SCALABILITY.md §5 for EOSC proximity as a better instrument).

---

## 8. Extending the pipeline

- **Add a new governance body:** append a row to
  `src/utils/governance_bodies.csv` with columns `name,category,url`.
- **Add a new control:** extend `_CONTROLS` in
  `src/analysis/causal_analysis.py` and add the column in
  `src/analysis/compute_variables.py`.
- **Add a new estimator:** write a `run_xxx(pdf)` function in
  `causal_analysis.py` returning the same dict shape, then dispatch to
  it from `run_causal_pipeline`.
- **Change the capability score:** modify the final `with_columns` in
  `compute_all_variables` — replace the z-sum with your preferred
  composite (weighted, PCA-based, etc).
- **Scale up:** see `SCALABILITY.md` for the full roadmap.

---

## 9. Citing

If you publish using this pipeline, please cite both the repository and
the specific OpenAIRE dump DOI that was active at runtime — it's
written to `results/provenance.json` automatically. A BibTeX stub is in
the main `README.md`.
