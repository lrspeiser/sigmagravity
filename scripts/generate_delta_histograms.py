#!/usr/bin/env python3
"""
Generate normalized histograms of residual distributions for all models.

Goal: Highlight bias reduction — show Σ centered at 0 while GR/MOND/NFW are skewed/offset.

Inputs:
- pred_csv: data/gaia/outputs/mw_gaia_full_coverage_predicted.csv
- fit_json: data/gaia/outputs/mw_pipeline_run_vendor/fit_params.json

Outputs:
- data/gaia/outputs/mw_delta_histograms.png
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
    ap.add_argument('--out_png', default=str(Path('data')/'gaia'/'outputs'/'mw_delta_histograms.png'))
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    fit = json.loads(Path(args.fit_json).read_text())
    
    # Load model parameters
    V200 = float(fit.get('nfw', {}).get('params', {}).get('V200', 180.0))
    c = float(fit.get('nfw', {}).get('params', {}).get('c', 14.0))
    a0 = float(fit.get('mond', {}).get('a0_m_s2', 1.2e-10))
    mond_kind = str(fit.get('mond', {}).get('kind', 'simple'))

    R = df['R_kpc'].to_numpy()
    v_obs = df['v_obs_kms'].to_numpy()
    v_bar = df['v_baryon_kms'].to_numpy()
    v_sig = df['v_model_kms'].to_numpy()

    # Compute predictions
    v_nfw = v_c_nfw(R, V200=V200, c=c)
    v_mond = v_c_mond_from_vbar(R, v_bar, a0_m_s2=a0, kind=mond_kind)

    g_obs = accel_m_s2(R, v_obs)
    g_bar = accel_m_s2(R, v_bar)
    g_sig = accel_m_s2(R, v_sig)
    g_nfw = accel_m_s2(R, v_nfw)
    g_mond = accel_m_s2(R, v_mond)

    # Residuals
    delta_bar = np.log10(np.clip(g_obs, 1e-20, None)) - np.log10(np.clip(g_bar, 1e-20, None))
    delta_sig = np.log10(np.clip(g_obs, 1e-20, None)) - np.log10(np.clip(g_sig, 1e-20, None))
    delta_nfw = np.log10(np.clip(g_obs, 1e-20, None)) - np.log10(np.clip(g_nfw, 1e-20, None))
    delta_mond = np.log10(np.clip(g_obs, 1e-20, None)) - np.log10(np.clip(g_mond, 1e-20, None))

    # Filter finite values
    delta_bar = delta_bar[np.isfinite(delta_bar)]
    delta_sig = delta_sig[np.isfinite(delta_sig)]
    delta_nfw = delta_nfw[np.isfinite(delta_nfw)]
    delta_mond = delta_mond[np.isfinite(delta_mond)]

    # Compute stats
    models = [
        ('GR (baryons)', delta_bar, '#1f77b4'),
        ('\u03a3-Gravity', delta_sig, '#d62728'),
        ('MOND', delta_mond, '#2ca02c'),
        ('NFW', delta_nfw, '#9467bd')
    ]

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, (name, delta, color) in enumerate(models):
        ax = axes[i]
        
        mu = np.nanmean(delta)
        sigma = np.nanstd(delta)
        n = len(delta)
        
        # Histogram
        ax.hist(delta, bins=80, density=True, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Vertical lines for mean and ±1σ
        ax.axvline(mu, color='black', ls='-', lw=2, label=f'\u03bc = {mu:+.3f} dex')
        ax.axvline(mu - sigma, color='black', ls='--', lw=1.5, alpha=0.6)
        ax.axvline(mu + sigma, color='black', ls='--', lw=1.5, alpha=0.6, 
                   label=f'\u03c3 = {sigma:.3f} dex')
        ax.axvline(0, color='red', ls=':', lw=2, alpha=0.8, label='Ideal (\u0394=0)')
        
        # Annotations
        ax.set_xlabel('\u0394 = log\u2081\u2080(g_obs) \u2212 log\u2081\u2080(g_pred)  [dex]', fontsize=11)
        ax.set_ylabel('Normalized density', fontsize=11)
        ax.set_title(f'{name}  (n = {n:,})', fontsize=12, weight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set consistent x-range for comparison
        ax.set_xlim(-0.6, 2.0)

    plt.suptitle('Residual Distributions: \u03a3-Gravity Centered at Zero', 
                 fontsize=14, weight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
