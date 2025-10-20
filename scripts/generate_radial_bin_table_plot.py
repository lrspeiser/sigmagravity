#!/usr/bin/env python3
"""
Generate visual radial-bin residual table plot (bar chart version of Table 3).

Goal: Quantify where each model succeeds/fails across radial bins.
      Show Σ dominates in 6-12 kpc (coherent tail regime) while matching GR in 3-6 kpc (gate regime).

Inputs:
- pred_csv: data/gaia/outputs/mw_gaia_full_coverage_predicted.csv
- fit_json: data/gaia/outputs/mw_pipeline_run_vendor/fit_params.json

Outputs:
- data/gaia/outputs/mw_radial_bin_table.png
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


def bin_stats_by_R(R: np.ndarray, delta: np.ndarray, edges: np.ndarray):
    idx = np.digitize(R, edges) - 1
    n = len(edges) - 1
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    
    for i in range(n):
        m = (idx == i) & np.isfinite(delta)
        if np.any(m):
            means[i] = np.nanmean(delta[m])
            stds[i] = np.nanstd(delta[m])
            counts[i] = m.sum()
    
    return means, stds, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', default=str(Path('data')/'gaia'/'outputs'/'mw_gaia_full_coverage_predicted.csv'))
    ap.add_argument('--fit_json', default=str(Path('data')/'gaia'/'outputs'/'mw_pipeline_run_vendor'/'fit_params.json'))
    ap.add_argument('--out_png', default=str(Path('data')/'gaia'/'outputs'/'mw_radial_bin_table.png'))
    ap.add_argument('--r_edges', default='3,6,8,10,12,14,16,20,25')
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

    delta_bar = np.log10(np.clip(g_obs, 1e-20, None)) - np.log10(np.clip(g_bar, 1e-20, None))
    delta_sig = np.log10(np.clip(g_obs, 1e-20, None)) - np.log10(np.clip(g_sig, 1e-20, None))
    delta_nfw = np.log10(np.clip(g_obs, 1e-20, None)) - np.log10(np.clip(g_nfw, 1e-20, None))
    delta_mond = np.log10(np.clip(g_obs, 1e-20, None)) - np.log10(np.clip(g_mond, 1e-20, None))

    edges = np.array([float(x) for x in args.r_edges.split(',') if x.strip()])
    
    mean_bar, std_bar, n_bar = bin_stats_by_R(R, delta_bar, edges)
    mean_sig, std_sig, n_sig = bin_stats_by_R(R, delta_sig, edges)
    mean_nfw, std_nfw, n_nfw = bin_stats_by_R(R, delta_nfw, edges)
    mean_mond, std_mond, n_mond = bin_stats_by_R(R, delta_mond, edges)

    # Bin labels
    bin_labels = [f'{edges[i]:.0f}\u2013{edges[i+1]:.0f}' for i in range(len(edges)-1)]
    x = np.arange(len(bin_labels))
    width = 0.2

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: Mean residuals
    ax = axes[0]
    ax.bar(x - 1.5*width, np.abs(mean_bar), width, label='GR (baryons)', color='#1f77b4', alpha=0.8)
    ax.bar(x - 0.5*width, np.abs(mean_sig), width, label='\u03a3-Gravity', color='#d62728', alpha=0.8)
    ax.bar(x + 0.5*width, np.abs(mean_mond), width, label='MOND', color='#2ca02c', alpha=0.8)
    ax.bar(x + 1.5*width, np.abs(mean_nfw), width, label='NFW', color='#9467bd', alpha=0.8)

    ax.set_ylabel('|Mean Residual| [dex]', fontsize=12)
    ax.set_title('Radial Bin Performance: \u03a3-Gravity Achieves Near-Zero Residuals Across All Radii', 
                 fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.6)

    # Add sample size annotations
    for i in x:
        ax.text(i, 1.52, f'n={n_sig[i]:,}', ha='center', va='bottom', fontsize=8, rotation=0, color='0.3')

    # Bottom panel: Improvement factor (GR → Σ)
    ax = axes[1]
    improvement = np.abs(mean_bar) / np.clip(np.abs(mean_sig), 0.001, None)
    colors_imp = ['#228b22' if imp > 2 else '#ff8c00' if imp > 1 else '#d62728' for imp in improvement]
    
    bars = ax.bar(x, improvement, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1)
    ax.axhline(1, color='k', ls='--', lw=1.5, alpha=0.6, label='No improvement')
    
    # Annotate values
    for i, (bar, val) in enumerate(zip(bars, improvement)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{val:.1f}\u00d7', 
                ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_xlabel('Radial Bin [kpc]', fontsize=12)
    ax.set_ylabel('Improvement Factor\n(GR \u2192 \u03a3)', fontsize=12)
    ax.set_title('Where \u03a3-Gravity Dominates GR', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 14)

    plt.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
