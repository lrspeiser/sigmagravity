#!/usr/bin/env python3
"""
Generate a k-sweep growth/lensing summary for Σ at linear scales.
Outputs:
- growth_mu_kgrid.csv  (rows: k_hMpc, k_1_per_m, a, mu, D, f)
- kgrid_meta.json      (parameters and grid definition)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sigma_cosmo import Cosmology, AofA, mu_of_k_a, growth_D_of_a

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Background & Σ parameters (Option B: Ω_eff in FRW, μ=1 on linear scales)
cosmo = Cosmology(
    H0_kms_Mpc=70.0,
    Omega_b0=0.048,
    Omega_r0=8.6e-5,
    Omega_L0=0.70,
    Omega_eff0=0.252  # Σ-driven geometric background; 0.30 - 0.048 ≈ ΛCDM Ω_m - Ω_b
)
h = cosmo.H0_kms_Mpc / 100.0
Mpc = 3.0856775814913673e22  # m

# Σ kernel on linear scales: μ≈1 (FRW provides matter background)
ell0 = 200e3 * 3.085677581e16  # m
p, ncoh = 0.75, 2.0  # keep C(R)->1 flat vs k
A0 = 0.0  # μ = 1 + A*C ≈ 1 for linear regime
at = 0.39
s = 2.6
A_form = "constant"  # no time variation; A0=0 → μ=1 everywhere

# k grid in h/Mpc (linear band)
k_h_list = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2], dtype=float)
# Convert to SI 1/m: k_SI = (k_h * h) / Mpc
k_si_list = (k_h_list * h) / Mpc

# Solve growth per k on [a0, a1]
a0, a1 = 0.1, 1.0
n_steps = 1200  # fine enough for smooth f(a)
rows = []

for k_h, k_si in zip(k_h_list, k_si_list):
    # μ(k,a)
    def A_of_a(a):
        return AofA(a, A0=A0, a_t=at, s=s, form=A_form)
    # Growth (returns a-grid in exp(linspace(ln a0, ln a1)))
    a_grid, D, f = growth_D_of_a(
        cosmo, k=k_si, a_ini=a0, a_fin=a1, n_steps=n_steps,
        ell0=ell0, p=p, ncoh=ncoh, A0=A0, a_t=at, s=s, A_form=A_form, two_pi=True
    )
    # μ along the same a_grid
    mu = mu_of_k_a(k_si, a_grid, ell0, p, ncoh, A0=A0, a_t=at, s=s, A_form=A_form, two_pi=True)
    rows.append(np.rec.fromarrays([np.full_like(a_grid, k_h), np.full_like(a_grid, k_si), a_grid, mu, D, f],
                                  names=["k_hMpc","k_1_per_m","a","mu","D","f"]))

tbl = np.hstack(rows)
np.savetxt(OUT / "growth_mu_kgrid.csv", tbl, delimiter=",",
           header="k_hMpc,k_1_per_m,a,mu,D,f", comments="")

meta = dict(
    H0_kms_Mpc=cosmo.H0_kms_Mpc,
    Omega_b0=cosmo.Omega_b0,
    Omega_r0=cosmo.Omega_r0,
    Omega_L0=cosmo.Omega_L0,
    h=h,
    ell0_m=ell0,
    p=p,
    ncoh=ncoh,
    A0=A0,
    A_form=A_form,
    a_t=at,
    s=s,
    a_range=[a0, a1],
    n_steps=n_steps,
    k_hMpc_list=k_h_list.tolist(),
    files=["growth_mu_kgrid.csv"],
)
(OUT / "kgrid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("Wrote:")
print("  ", OUT / "growth_mu_kgrid.csv")
print("  ", OUT / "kgrid_meta.json")
