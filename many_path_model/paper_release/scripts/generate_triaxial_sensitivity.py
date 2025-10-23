#!/usr/bin/env python3
"""
Triaxial sensitivity: θ_E vs q_LOS (and optional vs q_plane) for a chosen cluster
using the Σ‑kernel with parameters set to posterior medians.

Output: many_path_model/paper_release/figures/triaxial_sensitivity_A2261.png
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

FIXED_P = 2.0
FIXED_NCOH = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', default='A2261')
    ap.add_argument('--catalog', default=str(ROOT/'projects'/'SigmaGravity'/'data'/'clusters'/'master_catalog_paper.csv'))
    ap.add_argument('--posterior', default=str(ROOT/'projects'/'SigmaGravity'/'output'/'pymc_mass_scaled'/'flat_samples_from_pymc.npz'))
    ap.add_argument('--out', default=str(FIGDIR/'triaxial_sensitivity_A2261.png'))
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.catalog)
    row = df[df['cluster_name'] == args.cluster].iloc[0]

    # Posterior medians
    data = np.load(args.posterior, allow_pickle=True)
    samples = data['samples']
    ell0_star = float(np.median(samples[:,0]))
    gamma = float(np.median(samples[:,1]))
    muA = float(np.median(samples[:,2]))

    # Cluster basics
    name = row['cluster_name']; z_l = float(row['z_lens']); z_s = float(row['z_source'])
    R500 = float(row['R_500_kpc']); M500 = float(row['M_500_Msun'])

    # Override z_s if config present
    cfg_p = ROOT/'data'/'clusters'/f"{name.lower()}_config.json"
    if cfg_p.exists():
        try:
            cfg = json.loads(cfg_p.read_text())
            if 'arc_redshifts' in cfg and 'z_eff' in cfg['arc_redshifts']:
                z_s = float(cfg['arc_redshifts']['z_eff'])
        except Exception:
            pass

    # Build baryon model
    r_3d = np.logspace(-1, 3.5, 800)
    params = ClusterBaryonParams(M_500=M500, R_500=R500, z=z_l,
                                 fgas_target=row['fgas_R500'], T_keV=row['TX_central_keV'],
                                 C0=1.3, eta=2.0, C_max=2.5)
    components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)
    rho_total = components.rho_total

    nx, ny = 128, 128
    R_max = min(2500.0, R500 * 2.2)
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)

    M_BCG, r_eff_BCG = estimate_bcg_mass(M500, z_l)
    Sigma_BCG = hernquist_projected_density(R_grid_2d, M_BCG, r_eff_BCG)

    cosmo = LensingCosmology()
    ell0_cluster = ell0_star * (R500/1000.0)**gamma
    A_c = float(np.exp(muA))  # muA is in log-space for validate_holdout_mass_scaled? There it used normal on A_c, not log.
    # In that script mu_A was linear mean for A_c, not log. Use muA directly.
    A_c = float(muA)

    q_LOS_vals = np.linspace(0.7, 1.3, 25)
    qp_default = 1.0

    thetaE = []
    for ql in q_LOS_vals:
        Sigma_baryons = project_to_surface_density(r_3d, rho_total, R_grid_2d, ql, qp_default)
        Sigma_bar = Sigma_baryons + Sigma_BCG
        Sigma_eff, _, _ = convolve_sigma_with_kernel(Sigma_bar, R_grid_2d, ell0_cluster,
                                                     FIXED_P, FIXED_NCOH, A_c,
                                                     emphasize_interior=True, use_fft=True)
        # Compute ⟨κ⟩ and last crossing
        R_bins = np.linspace(0, R_max*0.9, 150)
        _, Sigma_prof, _ = azimuthal_average(Sigma_eff, R_grid_2d, R_bins)
        # Mean kappa
        valid = np.isfinite(Sigma_prof)
        if valid.sum() < 10:
            thetaE.append(np.nan); continue
        R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
        Sigma_prof = Sigma_prof[valid]
        Sigma_crit = cosmo.critical_surface_density(z_l, z_s)
        M_enc = np.cumsum(2*np.pi*R_prof*Sigma_prof*np.gradient(R_prof))
        kappa_mean = M_enc / (np.pi * R_prof**2 * Sigma_crit)
        kappa_mean[0] = Sigma_prof[0] / Sigma_crit
        idx = np.where(kappa_mean >= 1.0)[0]
        if len(idx) == 0:
            thetaE.append(np.nan)
        else:
            thetaE.append(float(R_prof[idx[-1]]))

    fig, ax = plt.subplots(1,1,figsize=(6.4,4.4))
    ax.plot(q_LOS_vals, np.array(thetaE), 'b-o', ms=3, lw=1.2)
    ax.set_xlabel('q_LOS'); ax.set_ylabel('θ_E [kpc]')
    ax.set_title(f'Triaxial sensitivity: {name} (q_plane={qp_default})')
    ax.grid(True, alpha=0.3)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=150); plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()