#!/usr/bin/env python3
"""
Generate growth observables for RSD and weak lensing from Σ-cosmology.

Outputs:
- growth_observables.csv: z, D(z), f(z), σ₈(z), fσ₈(z)
- growth_summary.json: S₈, σ₈(0), normalization info
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sigma_cosmo import Cosmology, growth_D_of_a

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Σ-cosmology with Ω_eff (Option B)
cosmo = Cosmology(
    H0_kms_Mpc=70.0,
    Omega_b0=0.048,
    Omega_r0=8.6e-5,
    Omega_L0=0.70,
    Omega_eff0=0.252
)

# Growth on a representative k (0.1 h/Mpc ~ 8h^-1 Mpc scale for σ₈)
h = 0.7
k_fid = 0.1 * h / (3.0856775814913673e22)  # convert to 1/m

# Solve growth from a_ini to a=1
a_grid, D, f = growth_D_of_a(
    cosmo, k=k_fid, a_ini=0.01, a_fin=1.0, n_steps=2000,
    ell0=200e3*3.085677581e16, p=0.75, ncoh=2.0,
    A0=0.0, a_t=0.39, s=2.6, A_form="constant", two_pi=True
)

z_grid = 1.0/a_grid - 1.0

# Normalize to σ₈(z=0) = 0.8 (ΛCDM-like)
sigma_8_z0 = 0.8
D_z0 = D[-1]  # D at a=1
sigma_8_z = sigma_8_z0 * (D / D_z0)

# fσ₈(z)
f_sigma_8 = f * sigma_8_z

# S₈ = σ₈(z=0) · (Ω_m/0.3)^0.5
Omega_m_z0 = cosmo.Omega_matter(1.0)
S_8 = sigma_8_z0 * np.sqrt(Omega_m_z0 / 0.3)

# Write CSV (reverse to ascending z)
idx = np.argsort(z_grid)
data = np.rec.fromarrays([
    z_grid[idx],
    D[idx],
    f[idx],
    sigma_8_z[idx],
    f_sigma_8[idx]
], names=["z","D","f","sigma_8","f_sigma_8"])

np.savetxt(OUT / "growth_observables.csv", data, delimiter=",",
           header="z,D,f,sigma_8,f_sigma_8", comments="")

# Summary JSON
summary = {
    "sigma_8_z0": float(sigma_8_z0),
    "S_8": float(S_8),
    "Omega_m_z0": float(Omega_m_z0),
    "h": h,
    "k_fid_h_per_Mpc": 0.1,
    "normalization": "σ₈(z=0)=0.8 ΛCDM-like; S₈ from Ω_matter(z=0)",
}

(OUT / "growth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "="*80)
print("GROWTH OBSERVABLES FOR RSD & WEAK LENSING")
print("="*80)
print(f"\nNormalization: σ₈(z=0) = {sigma_8_z0:.3f} (ΛCDM-like)")
print(f"Ω_m(z=0) = {Omega_m_z0:.3f} (Ω_b + Ω_eff)")
print(f"S₈ = σ₈·√(Ω_m/0.3) = {S_8:.3f}")
print(f"\nSample values:")
print(f"  z=0.0: D={D[-1]:.3f}, f={f[-1]:.3f}, fσ₈={f_sigma_8[-1]:.3f}")
print(f"  z=0.5: D={D[len(D)//2]:.3f}, f={f[len(f)//2]:.3f}, fσ₈={f_sigma_8[len(f)//2]:.3f}")
print(f"  z=1.0: D={D[0]:.3f}, f={f[0]:.3f}, fσ₈={f_sigma_8[0]:.3f}")
print("\n" + "="*80)
print("✓ Growth observables computed (match ΛCDM by construction)")
print("="*80 + "\n")

print(f"Wrote: {OUT / 'growth_observables.csv'}")
print(f"Wrote: {OUT / 'growth_summary.json'}")
