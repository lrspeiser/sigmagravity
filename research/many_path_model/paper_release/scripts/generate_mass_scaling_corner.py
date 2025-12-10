#!/usr/bin/env python3
"""
Generate mass-scaling corner plot (ell0_star, gamma, mu_A, sigma_A) and ℓ0(M) band vs R500.
Outputs:
- many_path_model/paper_release/figures/mass_scaling_corner.png
- many_path_model/paper_release/figures/l0_of_mass_band.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIGDIR = ROOT / 'many_path_model' / 'paper_release' / 'figures'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--posterior', default=str(ROOT/'projects'/'SigmaGravity'/'output'/'pymc_mass_scaled'/'flat_samples_from_pymc.npz'))
    ap.add_argument('--out_corner', default=str(FIGDIR/'mass_scaling_corner.png'))
    ap.add_argument('--out_band', default=str(FIGDIR/'l0_of_mass_band.png'))
    args = ap.parse_args()

    data = np.load(args.posterior, allow_pickle=True)
    samples = data['samples']
    ell0_star = samples[:,0]
    gamma = samples[:,1]
    muA = samples[:,2]
    sigA = samples[:,3]

    # Corner
    try:
        import corner
        corner_fig = corner.corner(np.vstack([ell0_star, gamma, muA, sigA]).T,
                                   labels=[r'$\ell_{0,\star}$ [kpc]', r'$\gamma$', r'$\mu_A$', r'$\sigma_A$'],
                                   quantiles=[0.16, 0.5, 0.84], show_titles=True)
        FIGDIR.mkdir(parents=True, exist_ok=True)
        corner_fig.savefig(args.out_corner, dpi=150, bbox_inches='tight')
        plt.close(corner_fig)
        print(f"Wrote {args.out_corner}")
    except Exception as e:
        print(f"WARN: corner plot not generated: {e}")

    # ℓ0(M) band vs R500
    R500_grid = np.linspace(500.0, 2000.0, 60)
    fig, ax = plt.subplots(1,1,figsize=(6.4,4.6))
    # Subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ell0_star), size=min(4000, len(ell0_star)), replace=False)
    e_s = ell0_star[idx]; g_s = gamma[idx]
    L_lo = []; L_med = []; L_hi = []
    for R in R500_grid:
        L = e_s * (R/1000.0)**g_s
        L_lo.append(np.percentile(L, 16))
        L_med.append(np.percentile(L, 50))
        L_hi.append(np.percentile(L, 84))
    L_lo = np.array(L_lo); L_med = np.array(L_med); L_hi = np.array(L_hi)
    ax.fill_between(R500_grid, L_lo, L_hi, color='lightcoral', alpha=0.4, linewidth=0)
    ax.plot(R500_grid, L_med, 'r-', lw=1.8)
    ax.set_xlabel('R500 [kpc]'); ax.set_ylabel('ℓ0 [kpc]')
    ax.set_title('Mass-scaled coherence ℓ0(M) = ℓ0⋆ (R500/1Mpc)^γ')
    ax.grid(True, alpha=0.3)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out_band, dpi=150); plt.close(fig)
    print(f"Wrote {args.out_band}")


if __name__ == '__main__':
    main()