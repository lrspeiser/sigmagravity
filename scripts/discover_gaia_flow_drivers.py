#!/usr/bin/env python3
"""Discover which local Gaia flow features predict velocity residuals.

Input
-----
CSV produced by export_gaia_pointwise_features.py.

What it does
------------
- trains an XGBoost regressor to predict residuals (v_phi_obs - v_pred)
- reports grouped CV by radial bin (or just random K-fold)
- outputs permutation importance and a SHAP summary plot (optional)

Use this to test hypotheses like:
  "residuals correlate with vorticity vs shear" or
  "residuals correlate with local inter-star separation"

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import shap
except Exception:
    shap = None


def safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.clip(x, floor, None))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['log10_n_star'] = safe_log10(out['n_star_kpc3'].to_numpy())
    out['log10_d_star'] = safe_log10(out['d_star_kpc'].to_numpy())
    out['log10_omega'] = 0.5 * safe_log10(out['omega2'].to_numpy())
    out['log10_shear'] = 0.5 * safe_log10(out['shear2'].to_numpy())
    out['theta_abs'] = np.abs(out['theta'].to_numpy())
    out['anisotropy_sigma'] = (out['sigma_phi_kms'] - out['sigma_R_kms']) / (out['sigma_1d_kms'] + 1e-6)
    out['sigma_ratio_z_R'] = out['sigma_z_kms'] / (out['sigma_R_kms'] + 1e-6)
    return out


def maybe_shap(model, X: pd.DataFrame, out_png: Path, max_points: int = 2000, seed: int = 0) -> None:
    if shap is None:
        print('SHAP not available; skipping')
        return
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        Xs = X.iloc[rng.choice(len(X), size=max_points, replace=False)]
    else:
        Xs = X
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    plt.figure()
    shap.summary_plot(sv, Xs, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f'Saved SHAP summary to: {out_png}')


def main() -> None:
    ap = argparse.ArgumentParser(description='Gaia flow-feature residual driver discovery')
    ap.add_argument('csv', type=str, help='gaia_pointwise_features.csv')
    ap.add_argument('--target', type=str, default='resid_kms', help='resid_kms | abs_resid_kms')
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--perm-repeats', type=int, default=8)
    ap.add_argument('--shap', action='store_true')
    ap.add_argument('--outdir', type=str, default='.')
    args = ap.parse_args()

    if xgb is None:
        raise SystemExit('xgboost not installed')

    df = pd.read_csv(args.csv)
    df = add_features(df)

    # need V_pred and residuals to be present
    if 'resid_kms' not in df.columns:
        raise SystemExit('Input missing resid_kms')

    y = df['resid_kms'].to_numpy(dtype=float)
    if args.target == 'abs_resid_kms':
        y = np.abs(y)

    # Features to try (physics-motivated)
    feat_cols = [
        'R_kpc', 'z_kpc',
        'log10_n_star', 'log10_d_star',
        'sigma_1d_kms', 'sigma_R_kms', 'sigma_phi_kms', 'sigma_z_kms',
        'anisotropy_sigma', 'sigma_ratio_z_R',
        'log10_omega', 'log10_shear', 'theta_abs',
    ]

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = y[X.index]

    kf = KFold(n_splits=args.splits, shuffle=True, random_state=42)
    rmses = []
    imps = []

    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        reg = xgb.XGBRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1,
        )
        reg.fit(X.iloc[tr], y[tr])
        pred = reg.predict(X.iloc[te])
        rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
        rmses.append(rmse)

        perm = permutation_importance(reg, X.iloc[te], y[te], n_repeats=args.perm_repeats, random_state=42, n_jobs=1)
        imps.append(perm.importances_mean)

        print(f'Fold {fold}/{args.splits}: RMSE={rmse:.4g}')

    imp = np.vstack(imps)
    imp_mean = pd.Series(imp.mean(axis=0), index=feat_cols).sort_values(ascending=False)
    imp_std = pd.Series(imp.std(axis=0), index=feat_cols).reindex(imp_mean.index)

    print('\nMean CV RMSE:', np.mean(rmses))
    print('\nTop permutation importances (mean ± std):')
    for f in imp_mean.index[:12]:
        print(f'  {f:<18} {imp_mean[f]:+.4g} ± {imp_std[f]:.4g}')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f'gaia_flow_drivers_{args.target}.csv'
    pd.DataFrame({
        'feature': imp_mean.index,
        'perm_importance_mean': imp_mean.values,
        'perm_importance_std': imp_std.values,
    }).to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')

    if args.shap:
        reg = xgb.XGBRegressor(
            n_estimators=900,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1,
        )
        reg.fit(X, y)
        maybe_shap(reg, X, outdir / f'gaia_shap_{args.target}.png')


if __name__ == '__main__':
    main()


