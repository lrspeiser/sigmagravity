#!/usr/bin/env python3
"""
Score Σ growth outputs vs ΛCDM reference.
Acceptance bands:
- μ(k,a=1) within ±5% of μ_needed=6.25
- D(a=1) ratio in [0.95, 1.05]
- f(a=1) relative error ≤0.05
- RMSE_D, RMSE_f each ≲0.1 over a∈[0.1,1]
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# --- Load Σ outputs ---
data = np.genfromtxt(OUT / "growth_mu_kgrid.csv", delimiter=",", names=True, encoding="utf-8")
meta = json.loads((OUT / "kgrid_meta.json").read_text(encoding="utf-8"))

# ΛCDM reference params (same H0, Omega_Lambda)
Omega_m_ref = 0.3
Omega_L_ref = 0.7
H0 = meta["H0_kms_Mpc"]

def E_lcdm(a):
    return np.sqrt(Omega_m_ref * a**(-3) + Omega_L_ref)

def Omega_m_a(a):
    return (Omega_m_ref * a**(-3)) / (E_lcdm(a)**2)

def dlnH_dlnA_lcdm(a, eps=1e-6):
    a1 = a*(1-eps); a2 = a*(1+eps)
    H1 = E_lcdm(a1); H2 = E_lcdm(a2)
    return (np.log(H2)-np.log(H1))/(np.log(a2)-np.log(a1))

# ΛCDM growth ODE (simple RK4)
def rk4(f, x0, y0, x1, n):
    x = np.linspace(x0, x1, n+1)
    h = (x1-x0)/n
    y = np.zeros((n+1, len(np.atleast_1d(y0))), dtype=float)
    y[0] = y0
    for i in range(n):
        xi = x[i]; yi = y[i]
        k1 = f(xi, yi)
        k2 = f(xi + 0.5*h, yi + 0.5*h*k1)
        k3 = f(xi + 0.5*h, yi + 0.5*h*k2)
        k4 = f(xi + h, yi + h*k3)
        y[i+1] = yi + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x, y

def growth_lcdm(a0=0.1, a1=1.0, n_steps=1200):
    def rhs(ln_a, y):
        a = np.exp(ln_a)
        Om = Omega_m_a(a)
        dlnH = dlnH_dlnA_lcdm(a)
        D, dD = y
        d2D = -(2.0 + dlnH)*dD + 1.5*Om*D
        return np.array([dD, d2D])
    lna, y = rk4(rhs, np.log(a0), np.array([a0, a0]), np.log(a1), n_steps)
    a_grid = np.exp(lna)
    D = y[:,0]
    f = y[:,1]/np.maximum(D, 1e-30)
    return a_grid, D, f

# Compute ΛCDM reference for same a-grid
a_ref, D_ref, f_ref = growth_lcdm(a0=meta["a_range"][0], a1=meta["a_range"][1], n_steps=meta["n_steps"])

# Option B: Σ FRW includes Ω_eff → Ω_total matches ΛCDM Ω_m
Omega_b0 = meta["Omega_b0"]
Omega_r0 = meta["Omega_r0"]
Omega_L0 = meta["Omega_L0"]
Omega_eff0 = 0.252  # effective Σ background (no DM particle)

def E_sigma(a):
    return np.sqrt(Omega_r0*a**(-4) + (Omega_b0+Omega_eff0)*a**(-3) + Omega_L0)

def Omega_total_sigma(a):
    return ((Omega_b0+Omega_eff0)*a**(-3))/(E_sigma(a)**2)

# μ target: ≈1 (±5%) since FRW now matches ΛCDM matter
mu_needed = np.ones_like(a_ref)  # target μ=1 everywhere

# Extract unique k's from data
k_list = np.unique(data["k_hMpc"])
results = []

for k_h in k_list:
    mask = (data["k_hMpc"] == k_h)
    a_sig = data["a"][mask]
    mu_sig = data["mu"][mask]
    D_sig = data["D"][mask]
    f_sig = data["f"][mask]
    
    # Interpolate ΛCDM to Σ's a-grid
    D_ref_interp = np.interp(a_sig, a_ref, D_ref)
    f_ref_interp = np.interp(a_sig, a_ref, f_ref)
    mu_need_interp = np.interp(a_sig, a_ref, mu_needed)
    
    # At a=1
    idx1 = np.argmin(np.abs(a_sig - 1.0))
    mu1 = mu_sig[idx1]
    D1_sig = D_sig[idx1]
    D1_ref = D_ref_interp[idx1]
    f1_sig = f_sig[idx1]
    f1_ref = f_ref_interp[idx1]
    mu_need_1 = mu_need_interp[idx1]
    
    # Errors at a=1
    mu_err_pct = 100.0 * (mu1 - mu_need_1) / mu_need_1
    D_ratio = D1_sig / D1_ref
    f_rel_err = (f1_sig / f1_ref) - 1.0
    
    # RMSE over full a-range
    rmse_D = float(np.sqrt(np.mean(((D_sig - D_ref_interp)/D_ref_interp)**2)))
    rmse_f = float(np.sqrt(np.mean(((f_sig - f_ref_interp)/f_ref_interp)**2)))
    
    # Pass/fail checks
    mu_pass = abs(mu_err_pct) <= 5.0
    D_pass = 0.95 <= D_ratio <= 1.05
    f_pass = abs(f_rel_err) <= 0.05
    rmse_pass = (rmse_D <= 0.1) and (rmse_f <= 0.1)
    all_pass = mu_pass and D_pass and f_pass and rmse_pass
    
    results.append({
        "k_hMpc": float(k_h),
        "mu_a1": float(mu1),
        "mu_needed_a1": float(mu_need_1),
        "mu_err_pct": float(mu_err_pct),
        "mu_pass": bool(mu_pass),
        "D_a1_ratio": float(D_ratio),
        "D_pass": bool(D_pass),
        "f_a1_rel_err": float(f_rel_err),
        "f_pass": bool(f_pass),
        "rmse_D": float(rmse_D),
        "rmse_f": float(rmse_f),
        "rmse_pass": bool(rmse_pass),
        "ALL_PASS": bool(all_pass),
    })

# Summary
print("\n" + "="*80)
print("Σ-GRAVITY vs ΛCDM ACCEPTANCE SCORECARD")
print("="*80)
print(f"\nTarget at a=1: μ_needed ≈ {mu_needed[-1]:.3f}  (Σ FRW has Ω_eff; expect μ≈1)")
print(f"Acceptance bands:")
print(f"  μ(k,a=1): ±5% of 1.0 (linear regime with Ω_eff in background)")
print(f"  D(a=1):   0.95 ≤ D_Σ/D_ΛCDM ≤ 1.05")
print(f"  f(a=1):   |f_Σ/f_ΛCDM - 1| ≤ 0.05")
print(f"  RMSE:     RMSE_D, RMSE_f each ≤ 0.1 over a∈[0.1,1]\n")

passing = sum(r["ALL_PASS"] for r in results)
total = len(results)

for r in results:
    status = "✓ PASS" if r["ALL_PASS"] else "✗ FAIL"
    print(f"k={r['k_hMpc']:.1e} h/Mpc  {status}")
    print(f"  μ(a=1)={r['mu_a1']:.3f} (need {r['mu_needed_a1']:.3f}; err {r['mu_err_pct']:+.1f}%) {'✓' if r['mu_pass'] else '✗'}")
    print(f"  D ratio={r['D_a1_ratio']:.3f} {'✓' if r['D_pass'] else '✗'}  |  f rel err={r['f_a1_rel_err']:+.3f} {'✓' if r['f_pass'] else '✗'}")
    print(f"  RMSE: D={r['rmse_D']:.3f}, f={r['rmse_f']:.3f} {'✓' if r['rmse_pass'] else '✗'}\n")

print("="*80)
print(f"SUMMARY: {passing}/{total} k-values pass all checks")
if passing == total:
    print("✓ SUCCESS: Σ matches ΛCDM linear growth across the linear band.")
else:
    print("⚠ NEEDS TUNING: adjust A0, a_t, s, or ncoh and re-run.")
print("="*80 + "\n")

# Write JSON summary
summary = {
    "passing": int(passing),
    "total": int(total),
    "mu_needed_a1": float(mu_needed[-1]),
    "results_per_k": results,
}
(OUT / "score_vs_lcdm.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Wrote: {OUT / 'score_vs_lcdm.json'}")
