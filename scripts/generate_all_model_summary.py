#!/usr/bin/env python3
"""
Generate all-model summary multipanel: single-glance comparison.

Top row: RAR scatter (g_pred vs g_obs) per model
Bottom row: Δ histogram per model

Goal: Readers instantly see Σ is simultaneously tight (RAR) and unbiased (Δ histogram).

Inputs:
- pred_csv: data/gaia/outputs/mw_gaia_full_coverage_predicted.csv
- fit_json: data/gaia/outputs/mw_pipeline_run_vendor/fit_params.json

Outputs:
- data/gaia/outputs/mw_all_model_summary.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendor.maxdepth_gaia.models import v_c_nfw, v_c_mond_from_vbar

KPC_M = 3.0856775814913673e19
KM_TO_M = 1000.0


def accel_m_s2(R_kpc: np.ndarray, V_kms: np.ndarray) -> np.ndarray:
    R_m = np.asarray(R_kpc, float) * KPC_M
    V = np.asarray(V_kms, float) * KM_TO_M
    with np.errstate(invalid='ignore', divide='ignore'):
        g = (V * V) / np.maximum(R_m, 1e-30)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', default=str(Path('data')/'gaia'/'outputs'/'mw_gaia_full_coverage_predicted.csv'))
    ap.add_argument('--fit_json', default=str(Path('data')/'gaia'/'outputs'/'mw_pipeline_run_vendor'/'fit_params.json'))
    ap.add_argument('--out_png', default=str(Path('data')/'gaia'/'outputs'/'mw_all_model_summary.png'))
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    fit = json.loads(Path(args.fit_json).read_text())
    
    V200 = float(fit.get('nfw', {}).get('params', {}).get('V200', 180.0))
    c = float(fit.get('nfw', {}).get('params', {}).get('c', 14.0))
    a0 = float(fit.get('mond', {}).get('a0_m_s2', 1.2e-10))
    mond_kind = str(fit.get('mond', {}).get('kind', 'simple'))

    R = df['R_kpc'].to_numpy()
    v_obs = df['v_obs_kms'].to_numpy()
    v_bar = df['v_baryon_kms'].to_numpy()
    v_sig = df['v_model_kms'].to_numpy()

    v_nfw = v_c_nfw(R, V200=V200, c=c)
    v_mond = v_c_mond_from_vbar(R, v_bar, a0_m_s2=a0, kind=mond_kind)

    g_obs = accel_m_s2(R, v_obs)
    g_bar = accel_m_s2(R, v_bar)
    g_sig = accel_m_s2(R, v_sig)
    g_nfw = accel_m_s2(R, v_nfw)
    g_mond = accel_m_s2(R, v_mond)

    log_obs = np.log10(np.clip(g_obs, 1e-20, None))
    log_bar = np.log10(np.clip(g_bar, 1e-20, None))
    log_sig = np.log10(np.clip(g_sig, 1e-20, None))
    log_nfw = np.log10(np.clip(g_nfw, 1e-20, None))
    log_mond = np.log10(np.clip(g_mond, 1e-20, None))

    delta_bar = log_obs - log_bar
    delta_sig = log_obs - log_sig
    delta_nfw = log_obs - log_nfw
    delta_mond = log_obs - log_mond

    models = [
        ('GR (baryons)', log_bar, delta_bar, '#1f77b4'),
        ('\u03a3-Gravity', log_sig, delta_sig, '#d62728'),
        ('MOND', log_mond, delta_mond, '#2ca02c'),
        ('NFW', log_nfw, delta_nfw, '#9467bd')
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))

    for i, (name, log_pred, delta, color) in enumerate(models):
        # Top row: RAR scatter
        ax_rar = axes[0, i]
        m = np.isfinite(log_pred) & np.isfinite(log_obs)
        ax_rar.hexbin(log_pred[m], log_obs[m], gridsize=80, cmap='Blues', mincnt=1, alpha=0.7)
        
        # 1:1 line
        lo = np.nanpercentile([log_obs[m], log_pred[m]], 1)
        hi = np.nanpercentile([log_obs[m], log_pred[m]], 99)
        ax_rar.plot([lo, hi], [lo, hi], 'k--', lw=2, label='1:1')
        
        # Stats
        mu = np.nanmean(delta[np.isfinite(delta)])
        sigma = np.nanstd(delta[np.isfinite(delta)])
        
        ax_rar.set_xlabel('log\u2081\u2080 g_pred [m/s\u00b2]', fontsize=10)
        ax_rar.set_ylabel('log\u2081\u2080 g_obs [m/s\u00b2]', fontsize=10)
        ax_rar.set_title(f'{name}\n\u03bc={mu:+.3f}, \u03c3={sigma:.3f} dex', fontsize=11, weight='bold')
        ax_rar.grid(True, alpha=0.3)
        ax_rar.legend(loc='lower right', fontsize=8)
        
        # Bottom row: Δ histogram
        ax_hist = axes[1, i]
        delta_finite = delta[np.isfinite(delta)]
        ax_hist.hist(delta_finite, bins=60, density=True, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_hist.axvline(mu, color='black', ls='-', lw=2)
        ax_hist.axvline(0, color='red', ls=':', lw=2, alpha=0.8)
        ax_hist.set_xlabel('\u0394 = log\u2081\u2080(g_obs) \u2212 log\u2081\u2080(g_pred) [dex]', fontsize=10)
        ax_hist.set_ylabel('Normalized density', fontsize=10)
        ax_hist.set_xlim(-0.6, 2.0)
        ax_hist.grid(True, alpha=0.3, axis='y')

    # Add super titles
    fig.text(0.5, 0.98, 'All-Model Summary: \u03a3-Gravity Simultaneously Tight and Unbiased', 
             ha='center', fontsize=15, weight='bold')
    fig.text(0.5, 0.52, 'RAR: Observed vs Predicted', ha='center', fontsize=12, style='italic')
    fig.text(0.5, 0.02, 'Residual Distributions', ha='center', fontsize=12, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
