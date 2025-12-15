#!/usr/bin/env python3
"""Discover what drives SPARC pointwise residuals.

This script is meant to help *discover missing physics variables* by
correlating and modeling the pointwise residuals exported from SPARC.

Key design choices:
- Grouped CV by galaxy (no leakage across radii within a galaxy)
- Interpretable outputs: correlations, permutation importance, optional SHAP

Typical usage:
  python scripts/discover_sparc_residual_drivers.py \
      scripts/regression_results/sparc_pointwise_baseline.csv \
      --subset highbulge --target dSigma --model xgb --shap

Input CSV expected columns (your export already includes these):
  galaxy, R_kpc, V_obs_kms, V_bar_kms, V_pred_kms,
  Sigma_req, Sigma_pred, dSigma,
  f_bulge_global, f_bulge_r, f_disk_r, f_gas_r,
  g_bar_SI, Omega_bar_SI, tau_dyn_Myr,
  dlnVbar_dlnR, dlnGbar_dlnR,
  A_use, C_term, h_term
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


# Optional deps
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import shap
except Exception:
    shap = None


# ----------------------------------------------------------------------------
# Constants (match the regression script defaults)
# ----------------------------------------------------------------------------
H0_SI = 2.27e-18
G_DAGGER = 9.598924108127284e-11  # from your reports (cH0/(4*sqrt(pi)))
L0_KPC = 0.40


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.clip(x, floor, None))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candidate 'missing-variable' proxies.

    These are intentionally simple and *dimensionless* so they can plausibly
    generalize across system types.
    """
    out = df.copy()

    # Basic log-scaled accelerations
    out["log10_gbar"] = safe_log10(out["g_bar_SI"].to_numpy())
    out["x_gbar"] = out["g_bar_SI"] / G_DAGGER
    out["log10_x_gbar"] = safe_log10(out["x_gbar"].to_numpy())

    # Orbital-frequency proxy (dimensionless): Omega/H0
    out["Omega_over_H0"] = out["Omega_bar_SI"] / H0_SI
    out["log10_Omega_over_H0"] = safe_log10(out["Omega_over_H0"].to_numpy())

    # A very direct 'tidal/decoherence' proxy:
    #   |Δg|/g over a coherence length L0  ~ |d ln g / d ln R| * (L0/R)
    # High in compact bulges, low in outer disks.
    out["tidal_L0"] = np.abs(out["dlnGbar_dlnR"]) * (L0_KPC / np.clip(out["R_kpc"], 1e-6, None))

    # Shear proxy: d ln Omega / d ln R = d ln(V/R)/d ln R = d ln V/d ln R - 1
    out["dlnOmega_dlnR"] = out["dlnVbar_dlnR"] - 1.0
    out["abs_dlnOmega_dlnR"] = np.abs(out["dlnOmega_dlnR"])

    # Compactness proxy: compare radius to L0 (how "central" the point is)
    out["R_over_L0"] = out["R_kpc"] / L0_KPC

    return out


def select_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    subset = subset.lower().strip()
    if subset in ("all", "*", "any"):
        return df
    if subset in ("highbulge", "high-bulge"):
        return df[df["f_bulge_r"] > 0.6]
    if subset in ("bulgegal", "bulge-galaxies"):
        return df[df["f_bulge_global"] > 0.3]
    if subset in ("diskgal", "disk-galaxies"):
        return df[df["f_bulge_global"] <= 0.3]
    if subset in ("need_sig_lt1", "need_sigma_lt_1"):
        return df[df["need_sigma_lt_1"].astype(bool)]
    raise ValueError(f"Unknown subset: {subset}")


def build_feature_matrix(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Select features; drop obviously leaky columns."""
    default_features = [
        # geometry + regime
        "R_over_Rd",
        "R_over_L0",
        "log10_x_gbar",
        "log10_Omega_over_H0",
        "tidal_L0",
        "abs_dlnOmega_dlnR",
        # morphology
        "f_bulge_global",
        "f_bulge_r",
        "f_disk_r",
        "f_gas_r",
        # existing model factors (allowed: we want to see if residuals key off them)
        "A_use",
        "C_term",
        "h_term",
        # shape
        "dlnVbar_dlnR",
        "dlnGbar_dlnR",
    ]

    cols = feature_cols or default_features
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    X = df[cols].copy()
    return X, cols


def get_target(df: pd.DataFrame, target: str) -> np.ndarray:
    t = target.lower().strip()
    if t == "dsigma":
        return df["dSigma"].to_numpy()
    if t in ("dv", "dv"):
        return (df["V_obs_kms"] - df["V_pred_kms"]).to_numpy()
    if t in ("log_ratio", "logv", "logv_ratio"):
        return safe_log10((df["V_obs_kms"] / df["V_pred_kms"]).to_numpy())
    if t in ("sigma_req", "sigmareq"):
        return df["Sigma_req"].to_numpy()
    raise ValueError(f"Unknown target: {target}")


@dataclass
class CVResult:
    rmse: float
    fold_rmses: List[float]
    perm_importance_mean: pd.Series
    perm_importance_std: pd.Series


def fit_cv_permutation_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model: str = "xgb",
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> CVResult:
    """Grouped CV + permutation importance on each held-out fold."""

    gkf = GroupKFold(n_splits=n_splits)

    feat_names = list(X.columns)
    importances = []
    rmses = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        if model == "xgb":
            if xgb is None:
                raise RuntimeError("xgboost is not installed; use --model ridge")
            # NOTE: use n_jobs=1 to avoid multiprocessing issues in some CI/containers.
            reg = xgb.XGBRegressor(
                n_estimators=800,
                learning_rate=0.04,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
            )
        elif model == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            reg = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=2.0, random_state=random_state)),
            ])
        else:
            raise ValueError(f"Unknown model: {model}")

        reg.fit(X_tr, y_tr)
        pred = reg.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        rmses.append(rmse)

        # NOTE: n_jobs=1 to avoid loky resource_tracker warnings on some systems.
        perm = permutation_importance(
            reg,
            X_te,
            y_te,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )
        importances.append(perm.importances_mean)

        print(f"Fold {fold}/{n_splits}: RMSE = {rmse:.5g}")

    imp = np.vstack(importances)
    imp_mean = pd.Series(imp.mean(axis=0), index=feat_names).sort_values(ascending=False)
    imp_std = pd.Series(imp.std(axis=0), index=feat_names).reindex(imp_mean.index)

    return CVResult(
        rmse=float(np.mean(rmses)),
        fold_rmses=[float(x) for x in rmses],
        perm_importance_mean=imp_mean,
        perm_importance_std=imp_std,
    )


def maybe_run_shap(model, X: pd.DataFrame, out_png: Path, max_points: int = 2000, seed: int = 0) -> None:
    """Generate SHAP summary plot for tree models."""
    if shap is None:
        print("SHAP not available; skipping.")
        return
    if xgb is None:
        print("xgboost not available; skipping SHAP.")
        return

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        idx = rng.choice(len(X), size=max_points, replace=False)
        Xs = X.iloc[idx]
    else:
        Xs = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    plt.figure()
    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Saved SHAP summary to: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover drivers of SPARC pointwise residuals")
    ap.add_argument("csv", type=str, help="Pointwise CSV (e.g., sparc_pointwise_baseline.csv)")
    ap.add_argument("--subset", type=str, default="all", help="all | highbulge | bulgegal | diskgal | need_sigma_lt_1")
    ap.add_argument("--target", type=str, default="dSigma", help="dSigma | dV | log_ratio | Sigma_req")
    ap.add_argument("--model", type=str, default="xgb", help="xgb | ridge")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--perm-repeats", type=int, default=12)
    ap.add_argument("--shap", action="store_true", help="Also generate SHAP summary plot (tree models only)")
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Basic sanity filter
    df = df[(df["V_obs_kms"] > 0) & (df["V_pred_kms"] > 0) & (df["R_kpc"] > 0)].copy()

    df = add_derived_features(df)
    df = select_subset(df, args.subset)

    if len(df) < 50:
        raise SystemExit(f"Subset '{args.subset}' too small: N={len(df)}")

    y = get_target(df, args.target)
    X, feat_cols = build_feature_matrix(df)
    groups = df["galaxy"].to_numpy()

    # Correlation screen
    print("\nTop correlations with target:")
    corrs = {}
    for c in feat_cols:
        try:
            corrs[c] = float(np.corrcoef(X[c].to_numpy(), y)[0, 1])
        except Exception:
            corrs[c] = np.nan
    corr_s = pd.Series(corrs).sort_values(key=lambda s: np.abs(s), ascending=False)
    print(corr_s.head(12).to_string())

    print("\nRunning grouped CV + permutation importance...")
    cv = fit_cv_permutation_importance(
        X, y, groups,
        model=args.model,
        n_splits=args.splits,
        n_repeats=args.perm_repeats,
    )

    print("\nCV RMSE (mean over folds):", f"{cv.rmse:.6g}")
    print("Fold RMSEs:", ", ".join(f"{r:.6g}" for r in cv.fold_rmses))

    print("\nTop permutation importances (mean ± std over folds):")
    top = cv.perm_importance_mean.head(15)
    for f, v in top.items():
        s = cv.perm_importance_std[f]
        print(f"  {f:<22} {v:+.6g} ± {s:.6g}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save a CSV report
    out_csv = outdir / f"sparc_residual_drivers_{args.subset}_{args.target}_{args.model}.csv"
    rep = pd.DataFrame({
        "feature": cv.perm_importance_mean.index,
        "perm_importance_mean": cv.perm_importance_mean.values,
        "perm_importance_std": cv.perm_importance_std.values,
        "corr_with_target": [corrs.get(f, np.nan) for f in cv.perm_importance_mean.index],
    })
    rep.to_csv(out_csv, index=False)
    print(f"\nSaved feature-importance table to: {out_csv}")

    # Optional SHAP (fit on full set)
    if args.shap and args.model == "xgb":
        if xgb is None:
            print("xgboost missing; cannot run SHAP")
        else:
            reg = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.04,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )
            reg.fit(X, y)
            out_png = outdir / f"sparc_shap_{args.subset}_{args.target}.png"
            maybe_run_shap(reg, X, out_png)


if __name__ == "__main__":
    main()


