#!/usr/bin/env python3
"""
Generate cluster ⟨kappa⟩(<R) panels for specified clusters using the paper's Σ‑kernel (Eq. II.6),
with GR(baryons) baseline, Σ‑Gravity median ±68% bands, and observed Einstein radius markers.

Outputs: many_path_model/paper_release/figures/cluster_kappa_panels.png
Dependencies: core/build_cluster_baryons.py, core/kernel2d_sigma.py, many_path_model/lensing_utilities.py
"""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIGDIR = ROOT / 'many_path_model' / 'paper_release' / 'figures'

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from many_path_model.lensing_utilities import LensingCosmology
from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from core.bcg_profiles import estimate_bcg_mass, hernquist_projected_density

KPC_TO_M = 3.0856776e19

# Paper-fixed kernel shape parameters for clusters (Eq. II.6 window W)
FIXED_P = 2.0
FIXED_NCOH = 2.0


def _load_overrides(name: str) -> dict | None:
    cfg_root = ROOT / 'data' / 'clusters'
    p = cfg_root / f"{name.lower()}_config.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _einstein_radius_from_profiles(R_bins, Sigma_eff_prof, z_lens, z_source, cosmo: LensingCosmology) -> float:
    # Compute mean kappa and find last crossing
    valid = np.isfinite(Sigma_eff_prof)
    if valid.sum() < 10:
        return np.nan
    R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
    Sigma_prof = Sigma_eff_prof[valid]
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_source)
    # Mean convergence inside R
    # M_enc = ∫ 2π R Σ(R) dR (cumulative trapezoid)
    M_enc = np.cumsum(2*np.pi*R_prof*Sigma_prof*np.gradient(R_prof))
    kappa_mean = M_enc / (np.pi * R_prof**2 * Sigma_crit)
    kappa_mean[0] = Sigma_prof[0] / Sigma_crit
    idx = np.where(kappa_mean >= 1.0)[0]
    if len(idx) == 0:
        return np.nan
    return float(R_prof[idx[-1]])


def _kappa_profiles(Sigma_map, R_grid_2d, z_lens, z_source, cosmo: LensingCosmology, n_bins: int = 150):
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_source)
    R_bins = np.linspace(0, R_grid_2d.max()*0.9, n_bins)
    _, Sigma_prof, _ = azimuthal_average(Sigma_map, R_grid_2d, R_bins)
    # Mean kappa inside R from cumulative mass
    valid = np.isfinite(Sigma_prof)
    R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
    Sigma_prof = Sigma_prof[valid]
    M_enc = np.cumsum(2*np.pi*R_prof*Sigma_prof*np.gradient(R_prof))
    kappa_mean = M_enc / (np.pi * R_prof**2 * Sigma_crit)
    kappa_mean[0] = Sigma_prof[0] / Sigma_crit
    return R_prof, Sigma_prof/Sigma_crit, kappa_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clusters', default='A2261,MACS1149', help='Comma-separated cluster names')
    ap.add_argument('--catalog', default=str(ROOT/'projects'/'SigmaGravity'/'data'/'clusters'/'master_catalog_paper.csv'))
    ap.add_argument('--posterior', default=str(ROOT/'projects'/'SigmaGravity'/'output'/'pymc_mass_scaled'/'flat_samples_from_pymc.npz'))
    ap.add_argument('--draws', type=int, default=500)
    ap.add_argument('--out', default=str(FIGDIR/'cluster_kappa_panels.png'))
    args = ap.parse_args()

    import pandas as pd
    catalog = pd.read_csv(args.catalog)
    names = [s.strip() for s in args.clusters.split(',') if s.strip()]
    rows = catalog[catalog['cluster_name'].isin(names)].copy()
    if rows.empty:
        raise SystemExit('No matching clusters in catalog')

    # Load posterior NPZ
    import numpy as np
    npz = np.load(args.posterior, allow_pickle=True)
    samples = npz['samples']
    ell0_star_samples = samples[:,0]
    gamma_samples = samples[:,1]
    mu_A_samples = samples[:,2]
    sigma_A_samples = samples[:,3]

    # Prepare fig
    n = len(rows)
    ncols = n
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5.0*nrows), squeeze=False)

    cosmo = LensingCosmology()

    for j, (_, cluster) in enumerate(rows.iterrows()):
        ax = axes[0][j]
        name = cluster['cluster_name']
        z_lens = float(cluster['z_lens'])
        z_source = float(cluster['z_source'])
        R500 = float(cluster['R_500_kpc'])
        thetaE_obs = float(cluster['theta_E_obs_arcsec'])
        thetaE_err = float(cluster['theta_E_err_arcsec'])

        # Overrides
        cfg = _load_overrides(name)
        if cfg and 'arc_redshifts' in cfg and 'z_eff' in cfg['arc_redshifts']:
            z_source = float(cfg['arc_redshifts']['z_eff'])

        # Build baryon model and geometry cache
        r_3d = np.logspace(-1, 3.5, 800)
        params = ClusterBaryonParams(
            M_500=cluster['M_500_Msun'], R_500=R500,
            z=z_lens, fgas_target=cluster['fgas_R500'], T_keV=cluster['TX_central_keV'],
            C0=1.3, eta=2.0, C_max=2.5
        )
        components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)
        rho_total = components.rho_total

        nx, ny = 128, 128
        R_max = min(2500.0, R500 * 2.2)
        x = np.linspace(-R_max, R_max, nx)
        y = np.linspace(-R_max, R_max, ny)
        X, Y = np.meshgrid(x, y)
        R_grid_2d = np.sqrt(X**2 + Y**2)

        # Spherical BCG component
        M_BCG, r_eff_BCG = estimate_bcg_mass(cluster['M_500_Msun'], z_lens)
        Sigma_BCG_spherical = hernquist_projected_density(R_grid_2d, M_BCG, r_eff_BCG)

        # Precompute triaxial projection cache for speed (nearest-neighbor over priors)
        Q_PLANE_GRID = np.linspace(0.6, 1.4, 9)
        Q_LOS_GRID = np.linspace(0.6, 1.4, 9)
        geom_cache = {}
        for q_p in Q_PLANE_GRID:
            for q_l in Q_LOS_GRID:
                Sigma_baryons_triaxial = project_to_surface_density(r_3d, rho_total, R_grid_2d, q_l, q_p)
                Sigma_bar = Sigma_baryons_triaxial + Sigma_BCG_spherical
                geom_cache[(float(q_p), float(q_l))] = Sigma_bar

        def get_Sigma_interp(q_plane, q_los):
            q_p = np.clip(q_plane, Q_PLANE_GRID[0], Q_PLANE_GRID[-1])
            q_l = np.clip(q_los, Q_LOS_GRID[0], Q_LOS_GRID[-1])
            idx_p = np.argmin(np.abs(Q_PLANE_GRID - q_p))
            idx_l = np.argmin(np.abs(Q_LOS_GRID - q_l))
            return geom_cache[(Q_PLANE_GRID[idx_p], Q_LOS_GRID[idx_l])]

        # Baseline GR (spherical approximation from cache nearest to (1,1))
        Sigma_bar_ref = get_Sigma_interp(1.0, 1.0)
        R_prof, kappa_prof_gr, kappa_mean_gr = _kappa_profiles(Sigma_bar_ref, R_grid_2d, z_lens, z_source, cosmo)

        # PPC draws
        rng = np.random.default_rng(7)
        idx = rng.choice(len(ell0_star_samples), size=min(args.draws, len(ell0_star_samples)), replace=False)
        ell0_star_post = ell0_star_samples[idx]
        gamma_post = gamma_samples[idx]
        mu_A_post = mu_A_samples[idx]
        sigma_A_post = sigma_A_samples[idx]

        # Geometry priors
        if cfg and 'geometry' in cfg:
            qp_mu = float(cfg['geometry']['q_plane_prior']['mean'])
            qp_std = float(cfg['geometry']['q_plane_prior']['std'])
            ql_mu = float(cfg['geometry']['q_los_prior']['mean'])
            ql_std = float(cfg['geometry']['q_los_prior']['std'])
        else:
            qp_mu, qp_std, ql_mu, ql_std = 1.0, 0.2, 1.0, 0.2

        # External convergence prior (affects Sigma_eff normalization additively)
        if cfg and 'environment' in cfg:
            kext_std = float(cfg['environment']['kappa_ext_prior'].get('std', 0.03))
        else:
            kext_std = 0.03

        profiles_mean = []
        profiles_mean_gr = kappa_mean_gr  # for overlay
        for i in range(len(ell0_star_post)):
            ell0_cluster = float(ell0_star_post[i] * (R500 / 1000.0)**gamma_post[i])
            A_c = float(np.clip(rng.normal(mu_A_post[i], max(1e-3, sigma_A_post[i])), 1.0, 15.0))
            q_plane_i = float(np.clip(rng.normal(qp_mu, qp_std), Q_PLANE_GRID[0], Q_PLANE_GRID[-1]))
            q_los_i = float(np.clip(rng.normal(ql_mu, ql_std), Q_LOS_GRID[0], Q_LOS_GRID[-1]))
            kappa_ext_i = float(rng.normal(0.0, kext_std))

            Sigma_bar = get_Sigma_interp(q_plane_i, q_los_i)
            Sigma_eff, _, _ = convolve_sigma_with_kernel(
                Sigma_bar, R_grid_2d, ell0_cluster, FIXED_P, FIXED_NCOH, A_c,
                emphasize_interior=True, use_fft=True
            )
            # Add external convergence
            if abs(kappa_ext_i) > 1e-6:
                Sigma_crit = cosmo.critical_surface_density(z_lens, z_source)
                Sigma_eff = Sigma_eff + kappa_ext_i * Sigma_crit

            R_prof_i, _, kappa_mean_i = _kappa_profiles(Sigma_eff, R_grid_2d, z_lens, z_source, cosmo)
            # Interpolate to common R_prof
            if i == 0:
                R_common = R_prof
            kappa_mean_i = np.interp(R_prof, R_prof_i, kappa_mean_i, left=np.nan, right=np.nan)
            profiles_mean.append(kappa_mean_i)

        D = np.vstack(profiles_mean)
        k_lo = np.nanpercentile(D, 16, axis=0)
        k_med = np.nanmedian(D, axis=0)
        k_hi = np.nanpercentile(D, 84, axis=0)

        # Plot
        ax.plot(R_prof, profiles_mean_gr, 'c--', lw=1.2, label='GR (baryons) ⟨κ⟩')
        ax.fill_between(R_prof, k_lo, k_hi, color='salmon', alpha=0.25, linewidth=0)
        ax.plot(R_prof, k_med, 'r-', lw=1.8, label='Σ‑Gravity ⟨κ⟩ (median ±68%)')

        # Observed Einstein radius marker
        # Convert θ_E_obs arcsec to kpc at z_lens
        # θ[arcsec] = (R/D_A) * (180/π) * 3600  =>  R = θ * D_A * (π/180) / 3600
        D_A_kpc = cosmo.angular_diameter_distance_kpc(z_lens)
        RE_kpc = thetaE_obs * D_A_kpc * (np.pi/180.0) / 3600.0
        ax.axvline(RE_kpc, color='k', ls=':', lw=1.2)
        ax.text(RE_kpc, ax.get_ylim()[1]*0.92, f"θ_E^obs={thetaE_obs:.1f}±{thetaE_err:.1f}″", rotation=90, va='top', ha='right')

        ax.set_title(f"{name}: ⟨κ⟩(<R)")
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('⟨κ⟩(<R)')
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc='best', fontsize=9)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()