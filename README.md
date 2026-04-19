# Infrastructural Capability Analysis

[![CI](https://github.com/yourusername/infrastructural-capability-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/infrastructural-capability-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/infrastructural-capability-analysis/blob/main/notebooks/01_EDA.ipynb)

## Hypothesis

> **Research communities with higher relational metadata linkage density in open infrastructures show greater collective capability to shape those infrastructures.**

This repository operationalises, measures, and causally tests the hypothesis using the **OpenAIRE Research Graph** (bulk Zenodo dumps), with Crossref and DataCite enrichment, and applies OLS, Negative Binomial, Propensity Score Matching (PSM), and (optionally) Instrumental Variable (IV) estimation.

## Theoretical Framing

The project draws on:
- Edwards et al. (2013) on infrastructural inversion
- Plantin et al. (2018) on infrastructure studies
- Bowker & Star (1999) on classification/metadata politics
- Fecher & Friesike (2014) and more recent Open Science governance literature

We operationalise *relational metadata linkage density* as the normalised degree of a community node in a bipartite research-products ↔ metadata-relations graph, and *collective capability* as a composite index of (i) verified governance-body participation, (ii) metadata enrichment contributions, and (iii) downstream reuse.

## Repository Structure

```
infrastructural-capability-analysis/
├── src/data_fetch/       # OpenAIRE dump download + parsing + enrichment
├── src/analysis/         # network, variables, causal, visualize
├── notebooks/            # EDA, Regression, Causal DAG
├── tests/                # pytest suite with mocks
├── data/                 # gitignored; cached parquet
└── run_analysis.py       # CLI entry point
```

## Setup

### Option 1: Poetry (recommended)

```bash
git clone https://github.com/yourusername/infrastructural-capability-analysis.git
cd infrastructural-capability-analysis
poetry install
poetry shell
```

### Option 2: pip + requirements.txt (Colab-friendly)

```bash
pip install -r requirements.txt
```

### Option 3: Docker

```bash
docker build -t ica:latest .
docker run --rm -v $(pwd)/data:/app/data ica:latest --help
```

## Downloading OpenAIRE Data

The main pipeline uses **bulk dumps** from the [OpenAIRE Research Graph Zenodo community](https://zenodo.org/communities/openaire-research-graph). The `download_dumps.py` script pins a version and caches everything as Parquet.

```bash
python -m src.data_fetch.download_dumps --version latest --entities publications,projects,relations,communities,organizations
```

**Manual fallback:** if the Zenodo DOI resolver is rate-limited, download the tarballs manually and place them under `data/raw/`.

The live [OpenAIRE Graph API](https://graph.openaire.eu/docs/apis/graph-api/) is used *only* for small refreshes or metadata enrichment on sampled communities.

## Example Runs

```bash
# Quick Colab-size test run (500 communities)
python run_analysis.py --data-source dump --mode eu --sample 500 --causal ols

# Full EU Horizon run with PSM
python run_analysis.py --data-source dump --mode eu --sample 10000 --causal psm

# Global North/South comparison with full causal stack
python run_analysis.py --data-source dump --mode global --sample 8000 --causal iv
```

## Outputs

All outputs land in `results/`:
- `coefficients.csv` – regression coefficients with p-values, CIs, effect sizes
- `diagnostics.json` – VIF, power analysis, covariate balance (post-PSM)
- `figures/` – PNG + interactive HTML (folium/plotly)
- `network_stats.parquet` – per-community linkage density, centrality, modularity
- `capability_scores.parquet` – composite capability index components

## Reproducibility

- **Dependency pinning:** Poetry lockfile + `requirements.txt` export
- **Data version pinning:** Zenodo DOI of the exact dump used is written to `results/provenance.json`
- **Seeds:** `--seed 42` (default) controls all stochastic steps (PSM matching, sampling)
- **CI:** GitHub Actions runs pytest + ruff on every push
- **Binder / Colab:** badges above

## Tests

```bash
poetry run pytest -v
```

The test suite mocks all external API calls (OpenAIRE, Crossref, DataCite, Zenodo) and validates parsing against a small synthetic dump fixture.

## Citation

If you use this code or the derived datasets, please cite:

```bibtex
@software{ica2026,
  author       = {Your Name},
  title        = {Infrastructural Capability Analysis: measuring the shaping power of research communities in open infrastructures},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/yourusername/infrastructural-capability-analysis}
}
```

Also cite:
- OpenAIRE Research Graph: Manghi et al., *Data Intelligence* (2021), DOI `10.1162/dint_a_00100`
- The specific Zenodo DOI of the OpenAIRE dump (written to `results/provenance.json` by the pipeline)

## Data Statement

- **Coverage bias:** OpenAIRE over-represents European and Open Access outputs. Global South and non-English-language communities are under-covered. We partially mitigate with DataCite enrichment and explicit North/South stratification, but residual bias remains.
- **Metadata completeness bias:** Communities with stronger existing infrastructural participation are also more likely to have *well-linked metadata*, creating a mechanical correlation. The causal identification strategy (PSM + IV with EOSC proximity / Horizon mandate as instrument) is designed to address this; limitations are discussed in the paper.
- **Temporal scope:** Default is the most recent full dump; panel analyses require loading ≥2 dumps and the pipeline supports this via `--dumps v1,v2,v3`.

## Limitations

1. OpenAIRE is not a complete census of world research; coverage biases propagate.
2. "Governance participation" is proxied by a curated list in `src/utils/governance_bodies.csv`; the list is non-exhaustive and maintained as a community resource.
3. PSM assumes unconfoundedness on observed controls; IV assumes instrument validity.
4. Community boundaries in OpenAIRE are partly self-defined (contexts), which introduces measurement noise.

Contributions to `governance_bodies.csv` and instrument refinements are welcome via PR.

## License

MIT — see [LICENSE](./LICENSE).
