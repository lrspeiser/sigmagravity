#!/usr/bin/env python3
"""
Generate parameter sensitivity tables for galaxies (RAR) and clusters (Einstein radius)
using only code and configs available in the paper_release bundle (plus core modules).

Outputs:
- many_path_model/paper_release/tables/galaxy_param_sensitivity.md
- many_path_model/paper_release/tables/cluster_param_sensitivity.md
- many_path_model/paper_release/figures/param_sensitivity_galaxy.png (optional)

Galaxy method:
- Load SPARC via ValidationSuite
- Use 80/20 stratified split
- For each parameter, vary ±50% around paper_release/config/hyperparams_track2.json and
  compute hold-out RAR scatter (model) using ValidationSuite.compute_btfr_rar(hp_override=...)
- Compute BIC proxy: BIC ≈ k ln N + N ln(σ^2) with σ = RAR scatter (dex)
- Ablations: zero-out β_bulge, α_shear, γ_bar

Cluster method (MACS0416):
- Build 3D baryon model via test_macs0416_projected_kernel functions
- Project to 2D; convolve with kernel2d_sigma for a grid of A_c and ℓ0 (±50% around baseline)
- Compute θ_E from <κ>=1 condition; compare to 30 arcsec (obs) and report fractional error

Note: The cluster sensitivity uses a reduced grid (256x256) for runtime.
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Repo root
ROOT = Path(__file__).resolve().parents[3]
PR = ROOT / 'many_path_model' / 'paper_release'
FIG_DIR = PR / 'figures'
TAB_DIR = PR / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))
from many_path_model.validation_suite import ValidationSuite
from many_path_model.path_spectrum_kernel_track2 import PathSpectrumHyperparams

# Cluster imports
from scripts.test_macs0416_projected_kernel import build_macs0416_baryon_profile_3d, project_to_surface_density
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from many_path_model.lensing_utilities import LensingCosmology


def load_hp() -> PathSpectrumHyperparams:
    hp_path = PR / 'config' / 'hyperparams_track2.json'
    if hp_path.exists():
        with open(hp_path, 'r') as f:
            return PathSpectrumHyperparams.from_dict(json.load(f))
    return PathSpectrumHyperparams()


def bic_proxy(sigma_dex: float, k_params: int, n_points: int) -> float:
    # BIC ≈ k ln N + N ln(σ^2) (up to a constant)
    sigma2 = max(1e-9, float(sigma_dex)**2)
    return k_params * math.log(max(1, n_points)) + n_points * math.log(sigma2)


def galaxy_param_sensitivity(seed: int = 42) -> pd.DataFrame:
    # Setup VS and split
    vs = ValidationSuite(PR / 'results' / 'sensitivity_tmp', load_sparc=True)
    train_df, test_df = vs.perform_train_test_split()
    base_hp = load_hp()

    # Baseline metrics on holdout
    _, rar_base = vs.compute_btfr_rar(test_df, hp_override=base_hp)
    n_points_est = 0
    # Rough point count from last call (stack count not returned; approximate from test_df)
    for _, gal in test_df.iterrows():
        if gal.get('r_all') is not None and gal.get('v_all') is not None:
            n_points_est += len(gal['r_all'])
    k_full = 7  # L0, beta, alpha, gamma_bar, A0, p, n_coh
    bic_base = bic_proxy(rar_base, k_full, n_points_est)

    rows = []
    def record(name: str, hp: PathSpectrumHyperparams, k_params: int, note: str):
        _, rar = vs.compute_btfr_rar(test_df, hp_override=hp)
        bic = bic_proxy(rar, k_params, n_points_est)
        rows.append({
            'config': name,
            'k_params': k_params,
            'rar_scatter_dex': round(float(rar), 3),
            'delta_BIC': round(float(bic - bic_base), 2),
            'note': note
        })

    # Ablations
    record('Full model (baseline)', base_hp, k_full, 'Paper HPs')
    record('No bulge gate (β=0)', PathSpectrumHyperparams(**{**base_hp.to_dict(), 'beta_bulge': 0.0}), k_full-1, 'Ablation')
    record('No shear gate (α=0)', PathSpectrumHyperparams(**{**base_hp.to_dict(), 'alpha_shear': 0.0}), k_full-1, 'Ablation')
    record('No bar gate (γ_bar=0)', PathSpectrumHyperparams(**{**base_hp.to_dict(), 'gamma_bar': 0.0}), k_full-1, 'Ablation')

    # Fix exponents
    record('Fix n_coh=0.5', PathSpectrumHyperparams(**{**base_hp.to_dict(), 'n_coh': 0.5}), k_full-1, 'Fix n_coh')
    record('Fix p=0.75', PathSpectrumHyperparams(**{**base_hp.to_dict(), 'p': 0.75}), k_full-1, 'Fix p')
    record('Fix p=1.0', PathSpectrumHyperparams(**{**base_hp.to_dict(), 'p': 1.0}), k_full-1, 'Fix p')

    # Single-parameter sweeps (±50%)
    def sweep(label: str, key: str, base_val: float, factors=(0.5, 0.75, 1.0, 1.25, 1.5)):
        for f in factors:
            val = float(base_val) * f
            hp = PathSpectrumHyperparams(**{**base_hp.to_dict(), key: val})
            name = f'{label}×{f:.2f}'
            record(name, hp, k_full, 'Sensitivity')

    sweep('L_0', 'L_0', base_hp.L_0)
    sweep('β_bulge', 'beta_bulge', base_hp.beta_bulge)
    sweep('α_shear', 'alpha_shear', base_hp.alpha_shear)
    sweep('γ_bar', 'gamma_bar', base_hp.gamma_bar)
    sweep('A_0', 'A_0', base_hp.A_0)
    sweep('p', 'p', base_hp.p)
    sweep('n_coh', 'n_coh', base_hp.n_coh)

    df = pd.DataFrame(rows)
    # Write table
    out_md = TAB_DIR / 'galaxy_param_sensitivity.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('| Configuration | k | RAR (dex) | ΔBIC | Note |\n')
        f.write('|---|---:|---:|---:|---|\n')
        for _, r in df.iterrows():
            f.write(f"| {r['config']} | {int(r['k_params'])} | {r['rar_scatter_dex']:.3f} | {r['delta_BIC']:+.2f} | {r['note']} |\n")
    print(f'Wrote {out_md}')
    return df


def cluster_param_sensitivity(use_cupy: bool = False) -> pd.DataFrame:
    # Baseline (from paper): ℓ0≈200 kpc, p=2, ncoh=2, A_c≈4.6 (population level)
    ell0_base = 200.0
    A_c_base = 4.6
    p = 2.0
    ncoh = 2.0

    # Build MACS0416 baryons
    r_3d = np.logspace(-1, 3.5, 800)
    rho_3d, info = build_macs0416_baryon_profile_3d(r_3d, verbose=False)
    nx = ny = 256
    R_max = 2000.0
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X*X + Y*Y)
    Sigma_bar = project_to_surface_density(r_3d, rho_3d, R_grid, 1.0, 1.0, use_cupy=use_cupy)

    cosmo = LensingCosmology()
    z_lens = info['z']
    z_src = 2.0
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)

    def theta_E(ell0, A_c):
        Sigma_eff, _, _ = convolve_sigma_with_kernel(
            Sigma_bar, R_grid, float(ell0), p, ncoh, float(A_c), emphasize_interior=True, use_fft=True
        )
        R_bins = np.linspace(0, R_max*0.9, 180)
        _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, R_grid, R_bins)
        valid = np.isfinite(Sigma_eff_prof)
        if not np.any(valid):
            return np.nan
        R_prof = 0.5*(R_bins[:-1] + R_bins[1:])[valid]
        Sigma_eff_prof = Sigma_eff_prof[valid]
        if len(R_prof) < 10:
            return np.nan
        M_enc = np.cumsum(2*np.pi*R_prof*np.nan_to_num(Sigma_eff_prof)) * (R_prof[1]-R_prof[0])
        mean_kappa = M_enc / (np.pi * R_prof**2 * Sigma_crit)
        idx = np.where(mean_kappa >= 1.0)[0]
        if len(idx) == 0:
            return np.nan
        R_E = R_prof[idx[-1]]
        return cosmo.physical_to_angular(R_E, z_lens)

    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    rows = []
    for f in factors:
        th = theta_E(ell0_base, A_c_base*f)
        rows.append({'param':'A_c','factor':f,'theta_E':float(th) if np.isfinite(th) else np.nan})
    for f in factors:
        th = theta_E(ell0_base*f, A_c_base)
        rows.append({'param':'ell0','factor':f,'theta_E':float(th) if np.isfinite(th) else np.nan})

    df = pd.DataFrame(rows)
    # Reference obs 30 arcsec
    df['theta_obs'] = 30.0
    df['frac_error_%'] = 100.0 * np.abs(df['theta_E'] - df['theta_obs']) / df['theta_obs']

    # Write table
    out_md = TAB_DIR / 'cluster_param_sensitivity.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('| Parameter | Factor | theta_E (arcsec) | |Err| (%) |\n')
        f.write('|---|---:|---:|---:|\n')
        for _, r in df.iterrows():
            f.write(f"| {r['param']} | {r['factor']:.2f} | {r['theta_E']:.2f} | {r['frac_error_%']:.1f} |\n")
    print(f'Wrote {out_md}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-galaxy', action='store_true')
    ap.add_argument('--no-cluster', action='store_true')
    ap.add_argument('--gpu', action='store_true', help='Use CuPy acceleration for cluster projection (spherical)')
    args = ap.parse_args()

    if not args.no_galaxy:
        galaxy_param_sensitivity()
    if not args.no_cluster:
        cluster_param_sensitivity(use_cupy=args.gpu)

if __name__ == '__main__':
    main()
