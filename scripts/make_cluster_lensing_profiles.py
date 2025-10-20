#!/usr/bin/env python3
"""
Cluster lensing visuals (self-contained, repo-native)

Generates two figures per selected cluster using the Σ-Gravity kernel:
1) Convergence profiles κ(R)
2) Deflection α(R) with the α=R line and θ_E markers

Models shown (axisymmetric toy for clarity):
- GR-only (baryons):       κ_b,  α_b  (SIS with θ_E,b)
- GR+Dark Matter (ref.):   κ_tot,α_tot (SIS with θ_E,tot = observed θ_E)
- Σ-Gravity (Eq. II.6):    κ_sig,α_sig with K_Σ(R)=A_c·C↑(R)

Key choices:
- We calibrate A_c s.t. <κ_sig>(<θ_E,tot)=1 (Einstein condition by construction)
- We set θ_E,b = f_b * θ_E,tot with a tunable f_b (default 0.33)
- The coherence scale uses a dimensionless fraction ℓ0_frac of θ_E,tot (default 0.6)

Input catalog: data/clusters/master_catalog.csv (expects 'cluster_name' and 'theta_E_obs_arcsec')
Outputs per cluster: data/clusters/figures/<name>_kappa_profiles.png and ..._alpha_profiles.png

This script is deliberately self-contained (no external core/* deps).
It is meant as a clean visual companion to the more detailed triaxial/baryon pipeline.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12


def C_up(R: np.ndarray, ell0: float, p: float = 2.0, ncoh: float = 2.0) -> np.ndarray:
    """Monotone-increasing coherence field; C_up(0)=0; →1 for R >> ell0."""
    R = np.asarray(R, dtype=float)
    x = np.power(np.maximum(R, 0.0) / max(ell0, EPS), p)
    return 1.0 - np.power(1.0 + x, -ncoh)


def cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    mid = 0.5 * (y[1:] + y[:-1])
    dx = np.diff(x)
    acc = np.cumsum(mid * dx)
    return np.concatenate(([0.0], acc))


def alpha_from_kappa(R: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Axisymmetric deflection: α(R) = (2/R) ∫_0^R κ(r) r dr"""
    I = cumulative_trapz(kappa * R, R)
    return 2.0 * I / np.maximum(R, EPS)


def mean_kappa_inside(R: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """⟨κ⟩(<R) = (2/R^2) ∫_0^R κ(r) r dr"""
    I = cumulative_trapz(kappa * R, R)
    return 2.0 * I / np.maximum(R, EPS) ** 2


def calibrate_Ac_for_mean_kappa(thetaE_b: float, R_target: float, R_grid: np.ndarray, C_vals: np.ndarray) -> float:
    """Solve for A_c enforcing ⟨κ_sig⟩(<R_target)=1 given κ_sig=κ_b*(1+A_c C_up), κ_b=θE_b/(2R) (SIS)."""
    mask = R_grid <= R_target
    if mask.sum() < 2:
        raise ValueError("R_target too small on grid")
    I = np.trapz(C_vals[mask], R_grid[mask])  # ∫_0^{R_E} C_up(r) dr
    # 1 = θE_b/R_E + (θE_b * A_c / R_E^2) * I
    return (1.0 - thetaE_b / R_target) * (R_target ** 2) / (thetaE_b * max(I, EPS))


def find_thetaE(R: np.ndarray, alpha: np.ndarray) -> float:
    d = alpha - R
    s = d >= 0
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    x1, x2, y1, y2 = R[i], R[i + 1], d[i], d[i + 1]
    return x1 - y1 * (x2 - x1) / (y2 - y1)


def make_figures_for_cluster(name: str, thetaE_obs_arcsec: float, outdir: Path,
                             fb: float = 0.33, ell0_frac: float = 0.60, p: float = 2.0, ncoh: float = 2.0,
                             r_max_mult: float = 2.5, n: int = 2000, plot_inner_frac: float = 0.05) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    thetaE_tot = float(thetaE_obs_arcsec)
    thetaE_b   = float(fb * thetaE_tot)

    R_max = r_max_mult * thetaE_tot
    R = np.linspace(max(thetaE_tot*1e-4, 1e-6), R_max, int(n))

    # SIS toys
    kappa_b = thetaE_b / (2.0 * R)
    alpha_b = np.full_like(R, thetaE_b)

    kappa_tot = thetaE_tot / (2.0 * R)
    alpha_tot = np.full_like(R, thetaE_tot)

    # Σ-Gravity
    ell0 = float(ell0_frac * thetaE_tot)
    C = C_up(R, ell0=ell0, p=p, ncoh=ncoh)
    A_c = calibrate_Ac_for_mean_kappa(thetaE_b, thetaE_tot, R, C)
    kappa_sig = kappa_b * (1.0 + A_c * C)

    Ic = cumulative_trapz(C, R)
    alpha_sig = thetaE_b + (thetaE_b * A_c / np.maximum(R, EPS)) * Ic

    # Einstein radii
    thetaE_b_meas   = find_thetaE(R, alpha_b)
    thetaE_tot_meas = find_thetaE(R, alpha_tot)
    thetaE_sig_meas = find_thetaE(R, alpha_sig)

    # FIG 1: kappa(R) — avoid divergent inner core in autoscale by masking R < plot_inner_frac·θ_E
    fig1, ax1 = plt.subplots(figsize=(7.6, 5.0))
    mask_plot = R >= max(plot_inner_frac * thetaE_tot, 3.0 * R[0])
    ax1.plot(R[mask_plot], kappa_b[mask_plot],   label="GR only (baryons)", linewidth=1.8)
    ax1.plot(R[mask_plot], kappa_tot[mask_plot], label="GR + Dark Matter",  linewidth=1.8)
    ax1.plot(R[mask_plot], kappa_sig[mask_plot], label="Σ‑Gravity",         linewidth=1.8)
    ax1.set_xlim(0, R_max)
    # Set a sensible upper y-limit from masked data
    ymax = np.nanmax([np.nanmax(kappa_b[mask_plot]), np.nanmax(kappa_tot[mask_plot]), np.nanmax(kappa_sig[mask_plot])])
    ax1.set_ylim(0, float(ymax * 1.1) if np.isfinite(ymax) else None)
    ax1.set_xlabel("Radius R [arcsec]")
    ax1.set_ylabel("Convergence κ(R)")
    ax1.set_title(f"{name}: Convergence profiles")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    f1 = outdir / f"{name}_kappa_profiles.png"
    fig1.savefig(f1, dpi=160)
    plt.close(fig1)

    # FIG 2: alpha(R)
    fig2, ax2 = plt.subplots(figsize=(7.6, 5.0))
    ax2.plot(R[mask_plot], alpha_b[mask_plot],   label="GR only (baryons)", linewidth=1.8)
    ax2.plot(R[mask_plot], alpha_tot[mask_plot], label="GR + Dark Matter",  linewidth=1.8)
    ax2.plot(R[mask_plot], alpha_sig[mask_plot], label="Σ‑Gravity",         linewidth=1.8)
    ax2.plot(R[mask_plot], R[mask_plot], linestyle=":", linewidth=1.2, label="α = R")

    for val, lab in [
        (thetaE_b_meas,   "θ_E (GR only)"),
        (thetaE_tot_meas, "θ_E (GR + DM)"),
        (thetaE_sig_meas, "θ_E (Σ‑Gravity)")
    ]:
        if np.isfinite(val):
            ax2.axvline(val, linestyle="--", linewidth=1.2)
            y0, y1 = ax2.get_ylim()
            ax2.text(val, y0 + 0.06*(y1 - y0), lab, rotation=90, va="bottom", ha="right")

    ax2.set_xlim(0, R_max)
    ax2.set_xlabel("Radius R [arcsec]")
    ax2.set_ylabel("Deflection α(R) [arcsec]")
    ax2.set_title(f"{name}: Deflection with Einstein-radius crossings")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    f2 = outdir / f"{name}_alpha_profiles.png"
    fig2.savefig(f2, dpi=160)
    plt.close(fig2)

    return dict(cluster=name, thetaE_obs=thetaE_tot, thetaE_bary=thetaE_b, A_c=A_c,
                thetaE_measured=dict(bary=thetaE_b_meas, dm=thetaE_tot_meas, sigma=thetaE_sig_meas),
                files=dict(kappa=str(f1), alpha=str(f2)))


def main():
    ap = argparse.ArgumentParser(description="Make cluster lensing visuals (κ, α) with Σ‑Gravity vs references")
    ap.add_argument('--catalog', default=str(Path('data')/'clusters'/'master_catalog.csv'))
    ap.add_argument('--clusters', default='', help='Comma-separated cluster names; omit or use --all to process all')
    ap.add_argument('--fb', type=float, default=0.33, help='Baryon-only Einstein fraction: θ_E,b = fb * θ_E,tot')
    ap.add_argument('--ell0_frac', type=float, default=0.60, help='ℓ0 as fraction of θ_E,tot (dimensionless)')
    ap.add_argument('--p', type=float, default=2.0)
    ap.add_argument('--ncoh', type=float, default=2.0)
    ap.add_argument('--outdir', default=str(Path('data')/'clusters'/'figures'))
    ap.add_argument('--all', action='store_true', help='Process all clusters in the catalog')
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    cols = {c.lower(): c for c in cat.columns}
    name_col = cols.get('cluster_name', None)
    te_col   = cols.get('theta_e_obs_arcsec', None)
    if name_col is None or te_col is None:
        raise SystemExit("Catalog must include 'cluster_name' and 'theta_E_obs_arcsec' columns")

    names = [s.strip() for s in str(args.clusters).split(',') if s.strip()]
    if args.all or not names:
        names = list(cat[name_col].astype(str).values)

    # Case-insensitive mapping for robust lookup
    lower_to_row = {str(r[name_col]).strip().lower(): r for _, r in cat.iterrows()}

    outdir = Path(args.outdir)

    results = []
    for nm in names:
        key = nm.strip().lower()
        row = lower_to_row.get(key)
        if row is None:
            print(f"[warn] cluster '{nm}' not found in catalog; skipping")
            continue
        thetaE = float(row[te_col])
        meta = make_figures_for_cluster(nm, thetaE, outdir, fb=args.fb, ell0_frac=args.ell0_frac,
                                        p=args.p, ncoh=args.ncoh)
        results.append(meta)
        print(f"Wrote figures for {nm}: {meta['files']}")

    # summary JSON
    import json
    (outdir / 'lensing_profiles_summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
