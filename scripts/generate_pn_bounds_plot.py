#!/usr/bin/env python3
"""
PN error budget illustration (Eq. II.8) for a representative SPARC galaxy.
We plot C1 (v/c)^3 + C2 (v/c)^2 (H/R) + C3 (v/c)^2 (R/RΣ) upper bounds (unit coefficients)
using observed rotation speeds and a simple thickness model H/R=0.05, and RΣ≈R (conservative).

Output: many_path_model/paper_release/figures/pn_bounds_ngc2403.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIGDIR = ROOT / 'many_path_model' / 'paper_release' / 'figures'

KPC_TO_M = 3.0856776e19
KM_TO_M = 1000.0
C = 299792458.0


def load_rotmod(rotmod_path: Path):
    rows = []
    with open(rotmod_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    if not rows:
        raise ValueError('No numeric rows')
    a = np.array(rows)
    return a[:,0], a[:,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rotmod', default=str(ROOT/'many_path_model'/'paper_release'/'data'/'Rotmod_LTG'/'NGC2403_rotmod.dat'))
    ap.add_argument('--out', default=str(FIGDIR/'pn_bounds_ngc2403.png'))
    args = ap.parse_args()

    R_kpc, V_kms = load_rotmod(Path(args.rotmod))
    R_m = R_kpc * KPC_TO_M
    v = V_kms * KM_TO_M
    x = R_kpc

    v_over_c = v / C
    H_over_R = np.full_like(x, 0.05)
    R_over_RSigma = np.ones_like(x)  # conservative upper bound

    term1 = v_over_c**3
    term2 = (v_over_c**2) * H_over_R
    term3 = (v_over_c**2) * R_over_RSigma
    bound = term1 + term2 + term3

    fig, ax = plt.subplots(1,1,figsize=(6.4,4.4))
    ax.plot(x, term1, 'b--', label='C1 (v/c)^3')
    ax.plot(x, term2, 'g--', label='C2 (v/c)^2 (H/R)')
    ax.plot(x, term3, 'm--', label='C3 (v/c)^2 (R/RΣ)')
    ax.plot(x, bound, 'r-', lw=1.8, label='Total upper bound')
    ax.set_yscale('log')
    ax.set_xlabel('R [kpc]'); ax.set_ylabel('Upper bound on |δg_R/g_N|')
    ax.set_title('PN error budget (unit coefficients) — NGC2403')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=150); plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()