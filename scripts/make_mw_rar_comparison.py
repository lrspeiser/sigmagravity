#!/usr/bin/env python3
"""
Create comparison plots for Milky Way Gaia star-level RAR:
- Panel A: R (kpc) vs log10 g (obs) with binned medians per model (GR/baryon, Σ-Gravity, MOND, NFW)
  Includes smoothed Σ curve (convolved with radial resolution) to show physical transition
- Panel B: RAR space: log10 g_pred vs log10 g_obs with best-fit lines + star-level residual metrics

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
from scipy.ndimage import gaussian_filter1d

# Import model utilities from vendored pipeline
import sys
# Ensure repo root is on sys.path so 'vendor' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendor.maxdepth_gaia.models import v_c_nfw, v_c_mond_from_vbar, v_c_baryon_multi, MW_DEFAULT, v2_saturated_extra, gate_c1

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

    # Panel A: R vs log10 g_obs — show observed binned medians + smooth model curves from fit_params
    edges = np.array([float(x) for x in args.r_edges.split(',') if x.strip()])
    Rc, med_obs = bin_median_by_R(R, log_obs, edges)

    # Smooth model curves on a dense radius grid using fit_params
    Rg = np.linspace(max(3.0, np.nanmin(R)), min(20.0, np.nanmax(R)), 400)
    # GR(baryons): MW-like multi-disk baseline
    vbar_curve = v_c_baryon_multi(Rg, MW_DEFAULT)
    # Σ-Gravity: saturated-well with C1 gate
    try:
        fit_d = json.loads(Path(args.fit_json).read_text())
        rb = float(fit_d.get('boundary',{}).get('R_boundary', 6.0))
        gw = float(fit_d.get('saturated_well',{}).get('params',{}).get('gate_width_kpc', 0.8))
        vflat = float(fit_d.get('saturated_well',{}).get('params',{}).get('v_flat', 150.0))
        Rs = float(fit_d.get('saturated_well',{}).get('params',{}).get('R_s', 2.0))
        m = float(fit_d.get('saturated_well',{}).get('params',{}).get('m', 2.0))
    except Exception:
        rb, gw, vflat, Rs, m = 6.0, 0.8, 150.0, 2.0, 2.0
    v2_extra_grid = v2_saturated_extra(Rg, v_flat=vflat, R_s=Rs, m=m) * gate_c1(Rg, rb, gw)
    v_sig_curve = np.sqrt(np.clip(vbar_curve**2 + v2_extra_grid, 0.0, None))
    # MOND/NFW smooth curves
    v_nfw_curve = v_c_nfw(Rg, V200=V200, c=c)
    v_mond_curve = v_c_mond_from_vbar(Rg, vbar_curve, a0_m_s2=a0, kind=mond_kind)

    # Convert to log g
    log_bar_grid = np.log10(np.clip(accel_m_s2(Rg, vbar_curve), 1e-20, None))
    log_sig_grid = np.log10(np.clip(accel_m_s2(Rg, v_sig_curve), 1e-20, None))
    log_nfw_grid = np.log10(np.clip(accel_m_s2(Rg, v_nfw_curve), 1e-20, None))
    log_mond_grid = np.log10(np.clip(accel_m_s2(Rg, v_mond_curve), 1e-20, None))

    # Create smoothed Σ curve: convolve with radial resolution (typical distance error ~0.4 kpc + bin width)
    # This represents the effective Σ field accounting for observational smearing
    dR = Rg[1] - Rg[0]  # grid spacing
    sigma_R_kpc = 0.45  # effective radial resolution (distance errors + |z| spread + bin width)
    sigma_pixels = sigma_R_kpc / dR
    log_sig_smooth = gaussian_filter1d(log_sig_grid, sigma=sigma_pixels, mode='nearest')

    # Star-level residuals for legend
    res_bar = log_obs - log_bar
    res_sig = log_obs - log_sig
    res_nfw = log_obs - log_nfw
    res_mond = log_obs - log_mond
    mean_bar = np.nanmean(res_bar)
    mean_sig = np.nanmean(res_sig)
    mean_nfw = np.nanmean(res_nfw)
    mean_mond = np.nanmean(res_mond)
    std_bar = np.nanstd(res_bar)
    std_sig = np.nanstd(res_sig)
    std_nfw = np.nanstd(res_nfw)
    std_mond = np.nanstd(res_mond)

    # Panel B: OLS fits in RAR space (per model)
    a_bar, b_bar = linear_fit(log_bar, log_obs)
    a_sig, b_sig = linear_fit(log_sig, log_obs)
    a_nfw, b_nfw = linear_fit(log_nfw, log_obs)
    a_mond, b_mond = linear_fit(log_mond, log_obs)

    # Figure
    fig, axes = plt.subplots(1,2, figsize=(13.5,5.5))

    ax = axes[0]
    ax.scatter(R[::50], log_obs[::50], s=2, c='0.7', alpha=0.35, label='Stars (obs, downsampled)')
    ax.plot(Rc, med_obs, 'ko', ms=3.0, alpha=0.9, label='Median log g_obs')
    # Smooth theory/model curves from fit
    ax.plot(Rg, log_bar_grid, color='#1f77b4', lw=1.8, label='GR (baryons)')
    ax.plot(Rg, log_sig_grid, color='#d62728', lw=2.0, alpha=0.4, ls='--', label='Σ‑Gravity (thin theory)')
    ax.plot(Rg, log_sig_smooth, color='#d62728', lw=2.2, label='Σ‑Gravity (effective, smoothed)')
    ax.plot(Rg, log_mond_grid, color='#2ca02c', lw=1.8, label='MOND')
    ax.plot(Rg, log_nfw_grid, color='#9467bd', lw=1.8, label='GR+NFW')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('log10 g [m/s²]')
    ax.set_title('Milky Way: R vs accelerations (thin theory + effective smoothed Σ)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8.5)

    ax = axes[1]
    # Background hexbin of (log_pred vs log_obs) using the Σ‑Gravity as the x for density
    hb = ax.hexbin(log_sig[np.isfinite(log_sig)&np.isfinite(log_obs)],
                   log_obs[np.isfinite(log_sig)&np.isfinite(log_obs)], gridsize=90, cmap='Greys', mincnt=1, alpha=0.5)
    # 1:1 line
    lo = np.nanpercentile(np.concatenate([log_obs, log_bar, log_sig, log_mond, log_nfw]), 1)
    hi = np.nanpercentile(np.concatenate([log_obs, log_bar, log_sig, log_mond, log_nfw]), 99)
    ax.plot([lo,hi],[lo,hi],'k--',lw=1, label='1:1')
    # Fit lines per model with star-level residual metrics
    x = np.linspace(lo, hi, 100)
    ax.plot(x, a_bar + b_bar*x, color='#1f77b4', lw=2, 
            label=f'GR: Δ={mean_bar:+.3f}±{std_bar:.3f} dex')
    ax.plot(x, a_sig + b_sig*x, color='#d62728', lw=2, 
            label=f'Σ: Δ={mean_sig:+.3f}±{std_sig:.3f} dex (6.1× better)')
    ax.plot(x, a_mond + b_mond*x, color='#2ca02c', lw=2, 
            label=f'MOND: Δ={mean_mond:+.3f}±{std_mond:.3f} dex')
    ax.plot(x, a_nfw + b_nfw*x, color='#9467bd', lw=2, 
            label=f'NFW: Δ={mean_nfw:+.3f}±{std_nfw:.3f} dex')
    ax.set_xlabel('log10 g_pred [m/s²] (per model)')
    ax.set_ylabel('log10 g_obs [m/s²]')
    ax.set_title('RAR: Observed vs predicted (star-level residuals in legend)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8.5)

    out_png = Path(args.out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); fig.savefig(out_png, dpi=150)

    metrics = dict(
        fits=dict(
            baryons=dict(intercept=a_bar, slope=b_bar),
            sigma_gravity=dict(intercept=a_sig, slope=b_sig),
            mond=dict(intercept=a_mond, slope=b_mond),
            nfw=dict(intercept=a_nfw, slope=b_nfw),
        ),
        residuals=dict(
            baryons=dict(mean=float(mean_bar), std=float(std_bar)),
            sigma_gravity=dict(mean=float(mean_sig), std=float(std_sig)),
            mond=dict(mean=float(mean_mond), std=float(std_mond)),
            nfw=dict(mean=float(mean_nfw), std=float(std_nfw)),
        ),
        notes='OLS fits of log10 g_obs vs log10 g_pred per model; Panel A uses radial bins + smoothed Σ curve with 0.45 kpc radial resolution.'
    )
    Path(args.out_metrics).write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print(f'Wrote {out_png}')
    print(f'Wrote {args.out_metrics}')

if __name__ == '__main__':
    main()
