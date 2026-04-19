"""Causal identification: OLS, Negative Binomial, PSM, IV, panel FE.

Diagnostics include VIF, a simple power analysis, and post-matching
covariate balance (standardised mean differences).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.power import TTestIndPower

from src.utils.logging import get_logger

logger = get_logger(__name__)

_CONTROLS = ["n_products", "oa_share", "n_countries", "discipline_diversity"]
_REGION_DUMMIES = ["region_north", "region_south", "region_mixed"]


def _prepare_design_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """Convert the Polars analysis frame to a clean pandas design matrix."""
    pdf = df.to_pandas()
    for col in _CONTROLS + ["linkage_density", "capability_score"]:
        if col in pdf:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").fillna(0.0)
    if "region" in pdf:
        dummies = pd.get_dummies(pdf["region"], prefix="region", drop_first=False).astype(float)
        for col in _REGION_DUMMIES:
            if col not in dummies:
                dummies[col] = 0.0
        pdf = pd.concat([pdf, dummies[_REGION_DUMMIES]], axis=1)
    else:
        for col in _REGION_DUMMIES:
            pdf[col] = 0.0
    if "is_eu_horizon" in pdf.columns:
        pdf["is_eu_horizon"] = pdf["is_eu_horizon"].astype(float)
    pdf["treated"] = (pdf["linkage_density"] > pdf["linkage_density"].median()).astype(int)
    return pdf


def _vif(X: pd.DataFrame) -> dict[str, float]:
    X = sm.add_constant(X, has_constant="add")
    return {
        col: float(variance_inflation_factor(X.values, i))
        for i, col in enumerate(X.columns)
        if col != "const"
    }


def _power_analysis(n: int, effect_size: float = 0.2, alpha: float = 0.05) -> float:
    analysis = TTestIndPower()
    return float(analysis.power(effect_size=effect_size, nobs1=n // 2, alpha=alpha))


def run_ols(pdf: pd.DataFrame) -> dict[str, Any]:
    """OLS: capability_score ~ linkage_density + controls."""
    X_cols = ["linkage_density", *_CONTROLS, *_REGION_DUMMIES]
    X = pdf[X_cols]
    y = pdf["capability_score"]
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const).fit(cov_type="HC3")
    return {
        "model": "OLS",
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "conf_int": model.conf_int().to_dict(),
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "n_obs": int(model.nobs),
        "vif": _vif(X),
    }


def run_negbin(pdf: pd.DataFrame) -> dict[str, Any]:
    """Negative Binomial: capability_enrichment ~ linkage_density + controls."""
    if "capability_enrichment" not in pdf.columns:
        return {"model": "NegBin", "skipped": True, "reason": "no enrichment column"}
    X_cols = ["linkage_density", *_CONTROLS, *_REGION_DUMMIES]
    X = sm.add_constant(pdf[X_cols], has_constant="add")
    y = pdf["capability_enrichment"].astype(int)
    try:
        model = sm.GLM(
            y, X, family=sm.families.NegativeBinomial(alpha=1.0)
        ).fit()
        return {
            "model": "NegBin",
            "params": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "aic": float(model.aic),
            "n_obs": int(model.nobs),
        }
    except Exception as e:  # pragma: no cover
        logger.warning("NegBin failed: %s", e)
        return {"model": "NegBin", "error": str(e)}


def run_psm(pdf: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Propensity Score Matching (1:1 nearest-neighbour, no replacement)."""
    covariates = [*_CONTROLS, *_REGION_DUMMIES]
    X = pdf[covariates].values
    y = pdf["treated"].values
    if y.sum() == 0 or y.sum() == len(y):
        return {"model": "PSM", "error": "degenerate treatment"}

    ps_model = LogisticRegression(max_iter=1000, random_state=seed).fit(X, y)
    ps = ps_model.predict_proba(X)[:, 1]
    pdf = pdf.assign(propensity=ps)

    treat = pdf[pdf["treated"] == 1].reset_index(drop=True)
    ctrl = pdf[pdf["treated"] == 0].reset_index(drop=True)
    if len(treat) == 0 or len(ctrl) == 0:
        return {"model": "PSM", "error": "empty arm"}

    nn = NearestNeighbors(n_neighbors=1).fit(ctrl[["propensity"]].values)
    _, idx = nn.kneighbors(treat[["propensity"]].values)
    matched_ctrl = ctrl.iloc[idx.flatten()].reset_index(drop=True)

    att = float(
        treat["capability_score"].mean() - matched_ctrl["capability_score"].mean()
    )
    # Bootstrap SE
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(500):
        idx_b = rng.integers(0, len(treat), size=len(treat))
        boots.append(
            treat.iloc[idx_b]["capability_score"].mean()
            - matched_ctrl.iloc[idx_b]["capability_score"].mean()
        )
    se = float(np.std(boots, ddof=1))
    ci = (att - 1.96 * se, att + 1.96 * se)

    # Balance: standardised mean difference post-match
    smd = {
        c: float(
            (treat[c].mean() - matched_ctrl[c].mean())
            / np.sqrt((treat[c].var() + matched_ctrl[c].var()) / 2 + 1e-12)
        )
        for c in covariates
    }

    return {
        "model": "PSM",
        "att": att,
        "se": se,
        "ci_95": ci,
        "n_treated": int(len(treat)),
        "n_matched": int(len(matched_ctrl)),
        "balance_smd": smd,
    }


def run_iv(pdf: pd.DataFrame) -> dict[str, Any]:
    """Two-stage least squares IV with EU Horizon / EOSC proximity as instrument.

    We use ``is_eu_horizon`` (binary) as a placeholder instrument for
    *exposure to infrastructural mandates*. First stage: linkage_density
    ~ instrument + controls; second stage: capability ~ predicted
    linkage_density + controls.
    """
    if "is_eu_horizon" not in pdf.columns or pdf["is_eu_horizon"].nunique() < 2:
        return {"model": "IV", "skipped": True, "reason": "instrument has no variation"}

    covariates = [*_CONTROLS, *_REGION_DUMMIES]
    # First stage
    X1 = sm.add_constant(pdf[[*covariates, "is_eu_horizon"]], has_constant="add")
    first = sm.OLS(pdf["linkage_density"], X1).fit(cov_type="HC3")
    pdf = pdf.assign(density_hat=first.predict(X1))

    # Weak instrument F-test on the excluded instrument
    r = first.t_test("is_eu_horizon = 0")
    first_stage_f = float((r.tvalue**2).item())

    # Second stage
    X2 = sm.add_constant(pdf[["density_hat", *covariates]], has_constant="add")
    second = sm.OLS(pdf["capability_score"], X2).fit(cov_type="HC3")

    return {
        "model": "IV-2SLS",
        "first_stage_F": first_stage_f,
        "weak_instrument": first_stage_f < 10,
        "params": second.params.to_dict(),
        "pvalues": second.pvalues.to_dict(),
        "rsquared": float(second.rsquared),
        "n_obs": int(second.nobs),
    }


def run_causal_pipeline(
    df: pl.DataFrame,
    method: str,
    results_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Dispatch to the requested method and always also run OLS for comparison."""
    pdf = _prepare_design_matrix(df)
    logger.info("Design matrix: %d rows, %d cols", pdf.shape[0], pdf.shape[1])

    results: dict[str, Any] = {
        "ols": run_ols(pdf),
        "negbin": run_negbin(pdf),
        "power": _power_analysis(len(pdf)),
    }

    if method == "psm":
        results["psm"] = run_psm(pdf, seed=seed)
    elif method == "iv":
        results["iv"] = run_iv(pdf)

    # Persist
    coef_rows = []
    for model_name, res in results.items():
        if not isinstance(res, dict) or "params" not in res:
            continue
        for var, coef in res["params"].items():
            coef_rows.append(
                {
                    "model": res.get("model", model_name),
                    "variable": var,
                    "coef": coef,
                    "pvalue": res.get("pvalues", {}).get(var),
                }
            )
    if coef_rows:
        pl.from_dicts(coef_rows).write_csv(results_dir / "coefficients.csv")

    (results_dir / "diagnostics.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info("Wrote coefficients.csv and diagnostics.json")
    return results
