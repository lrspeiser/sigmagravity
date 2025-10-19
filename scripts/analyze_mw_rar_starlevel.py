#!/usr/bin/env python3
"""
Analyze Milky Way RAR at star level using GPU-predicted velocities.

Inputs:
- data/gaia/outputs/mw_gaia_144k_predicted.csv (from scripts/predict_gaia_star_speeds.py)

Outputs (data/gaia/outputs/):
- mw_rar_starlevel.csv                # per-star table with g_bar, g_obs, g_model, logs, residuals
- mw_rar_starlevel.png                # comparison plot (RAR: observed vs baryon vs model)
- mw_rar_starlevel_metrics.txt        # summary metrics (global and by radius bins)

Notes:
- See data/gaia/README.md for provenance and operational details.
- Units: accelerations reported in m/s^2; plots in log10(m/s^2).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KPC_M = 3.0856775814913673e19  # meters per kpc
KM_TO_M = 1000.0


def compute_accel_m_s2(R_kpc: np.ndarray, V_kms: np.ndarray) -> np.ndarray:
    R_m = np.asarray(R_kpc, dtype=float) * KPC_M
    V_m_s = np.asarray(V_kms, dtype=float) * KM_TO_M
    with np.errstate(invalid='ignore', divide='ignore'):
        g = (V_m_s * V_m_s) / np.maximum(R_m, 1e-30)
    return g


def summarize_residuals(log_obs: np.ndarray, log_pred: np.ndarray, name: str) -> dict:
    d = log_obs - log_pred
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(name=name, n=0)
    q = np.nanpercentile(d, [5, 16, 50, 84, 95])
    return dict(
        name=name,
        n=int(d.size),
        mean=float(np.nanmean(d)),
        std=float(np.nanstd(d, ddof=1)),
        p05=float(q[0]), p16=float(q[1]), p50=float(q[2]), p84=float(q[3]), p95=float(q[4]),
    )


def main():
    ap = argparse.ArgumentParser(description='Star-level RAR analysis for the Milky Way')
    ap.add_argument('--pred_csv', default=str(Path('data')/'gaia'/'outputs'/'mw_gaia_144k_predicted.csv'))
    ap.add_argument('--out_prefix', default=str(Path('data')/'gaia'/'outputs'/'mw_rar_starlevel'))
    ap.add_argument('--r_bins', default='3,6,8,10,12,16,25', help='Comma-separated radial bin edges (kpc)')
    ap.add_argument('--hexbin', action='store_true', help='Use hexbin instead of scatter for the plot')
    args = ap.parse_args()

    pred_path = Path(args.pred_csv)
    if not pred_path.exists():
        raise SystemExit(f'Predictions CSV not found: {pred_path}')

    df = pd.read_csv(pred_path)
    required_cols = ['R_kpc','v_obs_kms','v_baryon_kms','v_model_kms']
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise SystemExit(f'CSV missing required columns: {miss}')

    # Compute accelerations (m/s^2)
    g_bar = compute_accel_m_s2(df['R_kpc'].to_numpy(), df['v_baryon_kms'].to_numpy())
    g_obs = compute_accel_m_s2(df['R_kpc'].to_numpy(), df['v_obs_kms'].to_numpy())
    g_mod = compute_accel_m_s2(df['R_kpc'].to_numpy(), df['v_model_kms'].to_numpy())

    # Logs
    with np.errstate(invalid='ignore'):
        log_gbar = np.log10(np.clip(g_bar, 1e-20, None))
        log_gobs = np.log10(np.clip(g_obs, 1e-20, None))
        log_gmod = np.log10(np.clip(g_mod, 1e-20, None))

    # Residuals (dex)
    resid_obs_minus_bar = log_gobs - log_gbar
    resid_obs_minus_mod = log_gobs - log_gmod

    out_prefix = Path(args.out_prefix)
    out_csv = out_prefix.with_suffix('.csv')
    out_png = out_prefix.with_suffix('.png')
    out_txt = out_prefix.with_name(out_prefix.name + '_metrics.txt')
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write star-by-star table
    out_df = df.copy()
    out_df['g_bar_m_s2'] = g_bar
    out_df['g_obs_m_s2'] = g_obs
    out_df['g_model_m_s2'] = g_mod
    out_df['log10_g_bar'] = log_gbar
    out_df['log10_g_obs'] = log_gobs
    out_df['log10_g_model'] = log_gmod
    out_df['resid_log10_obs_minus_bar'] = resid_obs_minus_bar
    out_df['resid_log10_obs_minus_model'] = resid_obs_minus_mod
    out_df.to_csv(out_csv, index=False)

    # Metrics (global)
    lines = []
    m_glob_bar = summarize_residuals(log_gobs, log_gbar, 'obs-minus-baryon (dex)')
    m_glob_mod = summarize_residuals(log_gobs, log_gmod, 'obs-minus-model (dex)')
    lines.append('Global residuals (log10 g_obs - log10 g_pred):')
    for m in (m_glob_bar, m_glob_mod):
        lines.append(f"- {m['name']}: n={m.get('n',0)}, mean={m.get('mean',np.nan):.4f}, std={m.get('std',np.nan):.4f}, "
                     f"p05={m.get('p05',np.nan):.4f}, p16={m.get('p16',np.nan):.4f}, p50={m.get('p50',np.nan):.4f}, "
                     f"p84={m.get('p84',np.nan):.4f}, p95={m.get('p95',np.nan):.4f}")

    # Metrics (by radius)
    edges = np.array([float(x) for x in args.r_bins.split(',') if x.strip()])
    R = df['R_kpc'].to_numpy()
    lines.append('\nBy radial bins (kpc):')
    for i in range(len(edges)-1):
        i0, i1 = edges[i], edges[i+1]
        mask = (R >= i0) & (R < i1) & np.isfinite(log_gobs)
        if not np.any(mask):
            continue
        mb = summarize_residuals(log_gobs[mask], log_gbar[mask], f'obs-minus-baryon {i0:.1f}–{i1:.1f} kpc')
        mm = summarize_residuals(log_gobs[mask], log_gmod[mask], f'obs-minus-model {i0:.1f}–{i1:.1f} kpc')
        for m in (mb, mm):
            lines.append(f"- {m['name']}: n={m.get('n',0)}, mean={m.get('mean',np.nan):.4f}, std={m.get('std',np.nan):.4f}, "
                         f"p16={m.get('p16',np.nan):.4f}, p50={m.get('p50',np.nan):.4f}, p84={m.get('p84',np.nan):.4f}")

    Path(out_txt).write_text('\n'.join(lines), encoding='utf-8')

    # Plot: RAR scatter/hexbin for Observed vs Baryon and Observed vs Model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, y, title in ((axes[0], log_gbar, 'Observed vs Baryon (GR)'), (axes[1], log_gmod, 'Observed vs Σ‑Gravity Model')):
        if args.hexbin:
            hb = ax.hexbin(y[np.isfinite(y) & np.isfinite(log_gobs)],
                           log_gobs[np.isfinite(y) & np.isfinite(log_gobs)],
                           gridsize=80, cmap='viridis', mincnt=1)
            fig.colorbar(hb, ax=ax, label='count')
        else:
            ax.scatter(y, log_gobs, s=3, c='k', alpha=0.15, linewidths=0)
        # 1:1 line
        lo = np.nanpercentile(np.concatenate([y, log_gobs]), 1)
        hi = np.nanpercentile(np.concatenate([y, log_gobs]), 99)
        ax.plot([lo, hi], [lo, hi], 'c--', lw=1)
        ax.set_title(title)
        ax.set_xlabel('log10 g_pred [m/s^2]')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('log10 g_obs [m/s^2]')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f'Wrote {out_csv}')
    print(f'Wrote {out_png}')
    print(f'Wrote {out_txt}')


if __name__ == '__main__':
    main()
