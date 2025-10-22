#!/usr/bin/env python3
"""
Generate numeric outputs for the Σ-cosmo scaffold:
- mu_k_a1.csv:    μ(k, a=1) over a small k grid
- growth_k.csv:   a,D,f for one representative k
- los_isw.txt:    toy LOS ISW-like redshift summary
- meta.json:      run parameters and references
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sigma_cosmo import Cosmology, AofA, mu_of_k_a, growth_D_of_a, los_isw_redshift

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# --- Parameters (match examples/run_demo.py) ---
cosmo = Cosmology(H0_kms_Mpc=70.0, Omega_b0=0.048, Omega_r0=8.6e-5, Omega_L0=0.70)
ell0 = 200e3 * 3.085677581e16  # m
p, ncoh = 0.75, 0.5
A0 = 4.6
a_t = 0.01
s = 2.0

# --- μ(k,a=1) table ---
# k grid in 1/m; using 2π/k mapping (two_pi=True)
k_vals = np.logspace(-5, -1, 9) / (3.085677581e22)
mu_vals = np.array([mu_of_k_a(k, 1.0, ell0, p, ncoh, A0=A0, a_t=a_t, s=s, A_form="logistic", two_pi=True)
                    for k in k_vals])
mu_tbl = np.rec.fromarrays([k_vals, mu_vals], names=["k_1_per_m","mu_a1"])
np.savetxt(OUT / "mu_k_a1.csv", mu_tbl, delimiter=",", header="k_1_per_m,mu_a1", comments="")

# --- Growth curve for a representative k ---
k_rep = k_vals[len(k_vals)//2]
a_grid, D, f = growth_D_of_a(cosmo, k=k_rep, ell0=ell0, p=p, ncoh=ncoh, A0=A0, a_t=a_t, s=s, A_form="logistic")
growth_tbl = np.rec.fromarrays([a_grid, D, f], names=["a","D","f"])
np.savetxt(OUT / "growth_k_rep.csv", growth_tbl, delimiter=",", header="a,D,f", comments="")

# --- Toy LOS ISW-like redshift ---
a_path = np.linspace(0.1, 1.0, 1001)
A_callable = lambda a: AofA(a, A0=A0, a_t=a_t, s=s, form='logistic')
cz_los = los_isw_redshift(a_path, A_callable, phi_bar_of_a=lambda a: 5e-5*(299792458.0**2)) * 3.0e5  # km/s

with open(OUT / "los_isw.txt", "w", encoding="utf-8") as f:
    f.write(f"Toy LOS ISW-like redshift (c*z) ≈ {cz_los:.6f} km/s\n")

# --- Meta ---
meta = dict(
    H0_kms_Mpc=cosmo.H0_kms_Mpc,
    Omega_b0=cosmo.Omega_b0,
    Omega_r0=cosmo.Omega_r0,
    Omega_L0=cosmo.Omega_L0,
    ell0_m=ell0,
    p=p,
    ncoh=ncoh,
    A0=A0,
    A_form="logistic",
    a_t=a_t,
    s=s,
    k_grid=len(k_vals),
    k_rep=float(k_rep),
    files=["mu_k_a1.csv","growth_k_rep.csv","los_isw.txt"],
)
(OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("Wrote:")
for name in meta["files"]:
    print("  ", OUT / name)
print("  ", OUT / "meta.json")
