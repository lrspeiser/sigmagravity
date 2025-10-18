#!/usr/bin/env python3
"""
Numerical verification of the azimuthal integrals underlying Eq. (II.3):
Compare analytic elliptic-integral expressions vs high-precision quadrature for
I0(R,R')=∫ dφ/Δ and I1(R,R')=∫ cosφ dφ/Δ, then report relative error.

Output: many_path_model/paper_release/figures/ring_kernel_elliptic_check.png
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import integrate, special

ROOT = Path(__file__).resolve().parents[3]
FIGDIR = ROOT / 'many_path_model' / 'paper_release' / 'figures'


def I0_analytic(R, Rp):
    Rg = max(R, Rp); Rl = min(R, Rp)
    k = 2*R*Rp/((R+Rp)**2)
    # K(m) with parameter m=k (here k acts as m)
    K = special.ellipk(k)
    return 4.0/Rg * K


def I1_analytic(R, Rp):
    Rg = max(R, Rp); Rl = min(R, Rp)
    k = 2*R*Rp/((R+Rp)**2)
    K = special.ellipk(k)
    E = special.ellipe(k)
    return 4.0/Rg * ((Rl/Rg)*K - E)


def I_num(R, Rp, kind=0):
    def integrand(phi):
        Delta = np.sqrt(R*R + Rp*Rp - 2*R*Rp*np.cos(phi))
        if kind == 0:
            return 1.0/Delta
        else:
            return np.cos(phi)/Delta
    val, _ = integrate.quad(integrand, 0.0, 2*np.pi, epsabs=1e-10, epsrel=1e-10, limit=400)
    return val


def main():
    R_vals = np.linspace(5.0, 50.0, 30)
    Rp_vals = np.linspace(5.0, 50.0, 30)

    err0 = np.zeros((len(R_vals), len(Rp_vals)))
    err1 = np.zeros((len(R_vals), len(Rp_vals)))

    for i, R in enumerate(R_vals):
        for j, Rp in enumerate(Rp_vals):
            a0 = I0_analytic(R, Rp)
            a1 = I1_analytic(R, Rp)
            n0 = I_num(R, Rp, kind=0)
            n1 = I_num(R, Rp, kind=1)
            err0[i,j] = abs((n0 - a0)/n0) if n0 != 0 else 0.0
            err1[i,j] = abs((n1 - a1)/n1) if n1 != 0 else 0.0

    fig, axes = plt.subplots(1,2,figsize=(10,4.2))
    im0 = axes[0].imshow(err0, origin='lower', extent=[Rp_vals[0], Rp_vals[-1], R_vals[0], R_vals[-1]],
                         aspect='auto', cmap='viridis', vmin=0, vmax=np.nanmax(err0))
    axes[0].set_title('Rel. error |I0_num - I0_an|/I0_num')
    axes[0].set_xlabel("R'"); axes[0].set_ylabel('R')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(err1, origin='lower', extent=[Rp_vals[0], Rp_vals[-1], R_vals[0], R_vals[-1]],
                         aspect='auto', cmap='viridis', vmin=0, vmax=np.nanmax(err1))
    axes[1].set_title('Rel. error |I1_num - I1_an|/I1_num')
    axes[1].set_xlabel("R'"); axes[1].set_ylabel('R')
    fig.colorbar(im1, ax=axes[1])

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / 'ring_kernel_elliptic_check.png'
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")

if __name__ == '__main__':
    main()