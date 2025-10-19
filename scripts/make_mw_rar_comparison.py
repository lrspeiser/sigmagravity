#!/usr/bin/env python3
"""
Create comparison plots for Milky Way Gaia star-level RAR:
- Panel A: R (kpc) vs log10 g (obs) with binned medians per model (GR/baryon, Σ-Gravity, MOND, NFW)
- Panel B: RAR space: log10 g_pred vs log10 g_obs with best-fit lines for each model + 1:1 line

Inputs:
- pred_csv: data/gaia/outputs/mw_gaia_144k_predicted.csv
- fit_json: data/gaia/outputs/mw_pipeline_run_vendor/fit_params.json
Outputs:
- data/gaia/outputs/mw_rar_comparison.png
- data/gaia/outputs/mw_rar_comparison_metrics.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import model utilities from vendored pipeline
import sys
# Ensure repo root is on sys.path so 'vendor' package is importable
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


def bin_median_by_R(R: np.ndarray, Y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    idx = np.digitize(R, edges) - 1
    out = np.full(len(edges)-1, np.nan)
    for i in range(len(edges)-1):
        m = (idx == i) & np.isfinite(Y)
        if np.any(m):
            out[i] = np.nanmedian(Y[m])
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, out


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float,float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    b, a = np.polyfit(x[m], y[m], 1)  # y = a + b x
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', default=str(Path('data')/'gaia'/'outputs'/'mw_gaia_144k_predicted.csv'))
    ap.add_argument('--fit_json', default=str(Path('data')/'gaia'/'outputs'/'mw_pipeline_run_vendor'/'fit_params.json'))
    ap.add_argument('--out_png', default=str(Path('data')/'gaia'/'outputs'/'mw_rar_comparison.png'))
    ap.add_argument('--out_metrics', default=str(Path('data')/'gaia'/'outputs'/'mw_rar_comparison_metrics.json'))
    ap.add_argument('--r_edges', default='3,4,5,6,7,8,9,10,11,12,13,14,16,18,20,22,24')
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    for c in ['R_kpc','v_obs_kms','v_baryon_kms','v_model_kms']:
        if c not in df.columns:
            raise SystemExit(f'Missing column {c} in {args.pred_csv}')

    # Load fit params for NFW and MOND
    fit = json.loads(Path(args.fit_json).read_text())
    V200 = float(fit.get('nfw',{}).get('params',{}).get('V200', 180.0))
    c = float(fit.get('nfw',{}).get('params',{}).get('c', 14.0))
    mond_kind = str(fit.get('mond',{}).get('kind','simple'))
    a0 = float(fit.get('mond',{}).get('a0_m_s2', 1.2e-10))

    R = df['R_kpc'].to_numpy()
    v_obs = df['v_obs_kms'].to_numpy()
    v_bar = df['v_baryon_kms'].to_numpy()
    v_sig = df['v_model_kms'].to_numpy()

    # Per-star NFW and MOND predictions
    v_nfw = v_c_nfw(R, V200=V200, c=c)
    v_mond = v_c_mond_from_vbar(R, v_bar, a0_m_s2=a0, kind=mond_kind)

    # Accelerations
    g_obs = accel_m_s2(R, v_obs)
    g_bar = accel_m_s2(R, v_bar)
    g_sig = accel_m_s2(R, v_sig)
    g_nfw = accel_m_s2(R, v_nfw)
    g_mond = accel_m_s2(R, v_mond)

    # Logs
    log_obs = np.log10(np.clip(g_obs, 1e-20, None))
    log_bar = np.log10(np.clip(g_bar, 1e-20, None))
    log_sig = np.log10(np.clip(g_sig, 1e-20, None))
    log_nfw = np.log10(np.clip(g_nfw, 1e-20, None))
    log_mond = np.log10(np.clip(g_mond, 1e-20, None))

    # Panel A: R vs log10 g_obs with binned medians of predictions
    edges = np.array([float(x) for x in args.r_edges.split(',') if x.strip()])
    Rc, med_obs = bin_median_by_R(R, log_obs, edges)
    _, med_bar = bin_median_by_R(R, log_bar, edges)
    _, med_sig = bin_median_by_R(R, log_sig, edges)
    _, med_nfw = bin_median_by_R(R, log_nfw, edges)
    _, med_mond = bin_median_by_R(R, log_mond, edges)

    # Panel B: OLS fits in RAR space (per model)
    a_bar, b_bar = linear_fit(log_bar, log_obs)
    a_sig, b_sig = linear_fit(log_sig, log_obs)
    a_nfw, b_nfw = linear_fit(log_nfw, log_obs)
    a_mond, b_mond = linear_fit(log_mond, log_obs)

    # Figure
    fig, axes = plt.subplots(1,2, figsize=(13.5,5.5))

    ax = axes[0]
    ax.scatter(R[::50], log_obs[::50], s=2, c='0.7', alpha=0.35, label='Stars (obs, downsampled)')
    ax.plot(Rc, med_obs, 'k-', lw=1.5, label='Median log g_obs')
    ax.plot(Rc, med_bar, color='#1f77b4', lw=1.8, label='GR (baryons)')
    ax.plot(Rc, med_sig, color='#d62728', lw=1.8, label='Σ‑Gravity')
    ax.plot(Rc, med_mond, color='#2ca02c', lw=1.8, label='MOND')
    ax.plot(Rc, med_nfw, color='#9467bd', lw=1.8, label='GR+NFW')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Median log10 g [m/s²]')
    ax.set_title('Milky Way: R vs accelerations (binned medians)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    # Background hexbin of (log_pred vs log_obs) using the Σ‑Gravity as the x for density
    hb = ax.hexbin(log_sig[np.isfinite(log_sig)&np.isfinite(log_obs)],
                   log_obs[np.isfinite(log_sig)&np.isfinite(log_obs)], gridsize=90, cmap='Greys', mincnt=1, alpha=0.5)
    # 1:1 line
    lo = np.nanpercentile(np.concatenate([log_obs, log_bar, log_sig, log_mond, log_nfw]), 1)
    hi = np.nanpercentile(np.concatenate([log_obs, log_bar, log_sig, log_mond, log_nfw]), 99)
    ax.plot([lo,hi],[lo,hi],'k--',lw=1, label='1:1')
    # Fit lines per model
    x = np.linspace(lo, hi, 100)
    ax.plot(x, a_bar + b_bar*x, color='#1f77b4', lw=2, label=f'GR fit: y={a_bar:.2f}+{b_bar:.2f}x')
    ax.plot(x, a_sig + b_sig*x, color='#d62728', lw=2, label=f'Σ fit: y={a_sig:.2f}+{b_sig:.2f}x')
    ax.plot(x, a_mond + b_mond*x, color='#2ca02c', lw=2, label=f'MOND fit: y={a_mond:.2f}+{b_mond:.2f}x')
    ax.plot(x, a_nfw + b_nfw*x, color='#9467bd', lw=2, label=f'NFW fit: y={a_nfw:.2f}+{b_nfw:.2f}x')
    ax.set_xlabel('log10 g_pred [m/s²] (per model)')
    ax.set_ylabel('log10 g_obs [m/s²]')
    ax.set_title('RAR: Observed vs predicted (best-fit lines)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    out_png = Path(args.out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); fig.savefig(out_png, dpi=150)

    metrics = dict(
        fits=dict(
            baryons=dict(intercept=a_bar, slope=b_bar),
            sigma_gravity=dict(intercept=a_sig, slope=b_sig),
            mond=dict(intercept=a_mond, slope=b_mond),
            nfw=dict(intercept=a_nfw, slope=b_nfw),
        ),
        notes='OLS fits of log10 g_obs vs log10 g_pred per model; Panel A uses radial bins defined in r_edges.'
    )
    Path(args.out_metrics).write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print(f'Wrote {out_png}')
    print(f'Wrote {args.out_metrics}')

if __name__ == '__main__':
    main()
