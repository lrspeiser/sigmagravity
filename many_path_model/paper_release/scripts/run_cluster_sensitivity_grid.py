#!/usr/bin/env python3
"""
Run cluster-level sensitivity grid over N≈10 clusters (Tier 1/2) and write a combined table.

- Varies population amplitude mu_A and coherence length ell0_star (±50% around paper values)
- Supports mass-scaling exponent gamma sweep
- Uses CPU multiprocessing (process pool); geometry kept spherical by default for speed
- Outputs: many_path_model/paper_release/tables/cluster_param_sensitivity_n10.md

Note: For full triaxial sensitivity, use existing validation scripts; this grid targets
parameter load-bearing analysis at moderate runtime.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[3]
PR = ROOT / 'many_path_model' / 'paper_release'
TAB_DIR = PR / 'tables'
TAB_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))
from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from scripts.test_macs0416_projected_kernel import project_to_surface_density
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from many_path_model.lensing_utilities import LensingCosmology


def theta_E_for_cluster(cluster_row: pd.Series, ell0_star: float, gamma: float, mu_A: float,
                        grid_n: int = 256, rmax_kpc: float = 2000.0, use_cupy: bool = False) -> float:
    # Build baryon model (3D)
    r_3d = np.logspace(-1, 3.5, 800)
    params = ClusterBaryonParams(
        M_500=cluster_row['M_500_Msun'], R_500=cluster_row['R_500_kpc'],
        z=cluster_row['z_lens'], fgas_target=cluster_row.get('fgas_R500', 0.11),
        T_keV=cluster_row.get('TX_central_keV', 8.0), C0=1.3, eta=2.0, C_max=2.5
    )
    components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)

    # Project to 2D (spherical q_plane=q_LOS=1 for speed)
    nx = ny = grid_n
    R_max = min(rmax_kpc, cluster_row['R_500_kpc'] * 2.2)
    x = np.linspace(-R_max, R_max, nx); y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X*X + Y*Y)
    Sigma_bar = project_to_surface_density(r_3d, components.rho_total, R_grid, 1.0, 1.0, use_cupy=use_cupy)

    # Apply kernel with mass-scaling ell0(M)
    Sigma_eff, _, _ = convolve_sigma_with_kernel(
        Sigma_bar, R_grid, ell0=None, p=2.0, ncoh=2.0, A_c=float(mu_A),
        emphasize_interior=True, use_fft=True, R500=float(cluster_row['R_500_kpc']),
        ell0_star=float(ell0_star), gamma=float(gamma)
    )

    # Compute Einstein radius from <kappa>=1
    cosmo = LensingCosmology()
    if 'z_source' in cluster_row and np.isfinite(cluster_row['z_source']):
        z_source = float(cluster_row['z_source'])
    else:
        z_source = 2.0  # fallback median if catalog lacks P(z_s)
    Sigma_crit = cosmo.critical_surface_density(float(cluster_row['z_lens']), z_source)

    R_bins = np.linspace(0, R_max*0.9, 180)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, R_grid, R_bins)
    valid = np.isfinite(Sigma_eff_prof)
    if not np.any(valid):
        return np.nan
    R_prof = 0.5*(R_bins[:-1] + R_bins[1:])[valid]
    Sigma_eff_prof = Sigma_eff_prof[valid]
    if len(R_prof) < 10:
        return np.nan
    # cumulative mass (trapezoid)
    dR = R_prof[1] - R_prof[0]
    M_enc = np.cumsum(2*np.pi*R_prof*np.nan_to_num(Sigma_eff_prof)) * dR
    mean_kappa = M_enc / (np.pi * R_prof**2 * Sigma_crit)
    idx = np.where(mean_kappa >= 1.0)[0]
    if len(idx) == 0:
        return np.nan
    R_E = R_prof[idx[-1]]
    return cosmo.physical_to_angular(R_E, float(cluster_row['z_lens']))


def run_grid(catalog_path: Path, out_md: Path,
             mu_A_vals=(3.0, 4.6, 6.0), ell0_star_vals=(100.0, 200.0, 300.0), gamma_vals=(0.0, 0.1, 0.2),
             max_workers: int | None = None, grid_n: int = 256, use_cupy: bool = False) -> pd.DataFrame:
    cat = pd.read_csv(catalog_path)
    # Filter Tier 1/2
    if 'tier' in cat.columns:
        cat = cat[cat['tier'].isin([1,2])].copy()
    # Keep ~10 entries
    if len(cat) > 10:
        cat = cat.sort_values(by=cat.columns[0]).head(10).copy()

    tasks = []
    for _, row in cat.iterrows():
        for mu in mu_A_vals:
            for e0 in ell0_star_vals:
                for g in gamma_vals:
                    tasks.append((row.to_dict(), e0, g, mu))

    results = []
    max_workers = max_workers or os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2key = {
            ex.submit(theta_E_for_cluster, pd.Series(t[0]), t[1], t[2], t[3], grid_n, 2000.0, use_cupy): t
            for t in tasks
        }
        for fut in as_completed(fut2key):
            row_dict, e0, g, mu = fut2key[fut]
            name = row_dict['cluster_name']
            try:
                theta = fut.result()
            except Exception as e:
                theta = np.nan
            results.append({
                'cluster': name, 'mu_A': mu, 'ell0_star': e0, 'gamma': g,
                'theta_E_pred': float(theta) if np.isfinite(theta) else np.nan,
                'theta_E_obs': float(row_dict.get('theta_E_obs_arcsec', np.nan))
            })

    df = pd.DataFrame(results)
    df['frac_error_%'] = 100.0 * np.abs(df['theta_E_pred'] - df['theta_E_obs']) / df['theta_E_obs']
    # Write table
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('| Cluster | mu_A | ell0_star [kpc] | gamma | theta_E_pred [\"] | theta_E_obs [\"] | |Err| (%) |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for _, r in df.iterrows():
            f.write(f"| {r['cluster']} | {r['mu_A']:.2f} | {r['ell0_star']:.1f} | {r['gamma']:.2f} | "
                    f"{r['theta_E_pred']:.2f} | {r['theta_E_obs']:.2f} | {r['frac_error_%']:.1f} |\n")
    print(f'Wrote {out_md}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', default=str(ROOT / 'data' / 'clusters' / 'master_catalog.csv'))
    ap.add_argument('--out', default=str(PR / 'tables' / 'cluster_param_sensitivity_n10.md'))
    ap.add_argument('--grid-n', type=int, default=256)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--gpu', action='store_true', help='Use CuPy acceleration (spherical projection only)')
    args = ap.parse_args()

    # Avoid GPU contention: if GPU requested, run single-process
    workers = 1 if args.gpu else args.workers
    run_grid(Path(args.catalog), Path(args.out), max_workers=workers, grid_n=args.grid_n, use_cupy=args.gpu)

if __name__ == '__main__':
    main()
