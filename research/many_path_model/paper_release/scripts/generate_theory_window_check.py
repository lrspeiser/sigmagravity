#!/usr/bin/env python3
"""
Illustrate the coherence window W(Δ; ℓ0, p, ncoh) shapes and the thin‑ring dominance in
KΣ(R) ∝ ∫ Σ(R') G(R,R') W(|R-R'|) R' dR' using a toy Σ(R').

Output: many_path_model/paper_release/figures/window_and_integrand.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import special

ROOT = Path(__file__).resolve().parents[3]
FIGDIR = ROOT / 'many_path_model' / 'paper_release' / 'figures'


def W(delta, l0, p, ncoh):
    return 1.0 / (1.0 + (np.maximum(delta, 0.0)/l0)**p)**ncoh


def G_ring(R, Rp):
    # Eq. II.3 form via elliptics
    Rg = np.maximum(R, Rp); Rl = np.minimum(R, Rp)
    k = 2*R*Rp/((R+Rp)**2)
    K = special.ellipk(k)
    E = special.ellipe(k)
    return (2*np.pi/Rg) * (K - (Rl/Rg)*E)


def main():
    # Panel 1: W shapes
    d = np.linspace(0.0, 800.0, 400)
    l0 = 200.0
    p1 = W(d, l0, p=1.0, ncoh=2.0)
    p2 = W(d, l0, p=2.0, ncoh=2.0)

    # Panel 2: Thin-ring dominance for a toy Σ(R)
    R = 300.0
    Rp = np.linspace(1.0, 1200.0, 800)
    Sigma_toy = np.exp(-Rp/400.0)
    integrand = Sigma_toy * G_ring(R, Rp) * W(np.abs(R-Rp), l0=200.0, p=2.0, ncoh=2.0) * Rp

    fig, axes = plt.subplots(1,2,figsize=(10.5,4.2))
    axes[0].plot(d, p1, 'b-', lw=1.5, label='p=1, ncoh=2')
    axes[0].plot(d, p2, 'r-', lw=1.5, label='p=2, ncoh=2')
    axes[0].set_xlabel('Δ [kpc]'); axes[0].set_ylabel('W(Δ; ℓ0)')
    axes[0].set_title('Coherence windows W(Δ; ℓ0=200 kpc)')
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(Rp, integrand/np.nanmax(integrand), 'k-', lw=1.5)
    axes[1].axvline(R, color='c', ls='--', lw=1.2)
    axes[1].set_xlabel("R' [kpc]"); axes[1].set_ylabel('Normalized integrand')
    axes[1].set_title("Thin-ring dominance at R=300 kpc")
    axes[1].grid(True, alpha=0.3)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / 'window_and_integrand.png'
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")

if __name__ == '__main__':
    main()