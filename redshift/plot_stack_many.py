#!/usr/bin/env python3
"""
Plot the stacked gravitational redshift profile with BCG-relative Σ prediction.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'redshift' / 'outputs'

# Load data
data = np.genfromtxt(OUTDIR / 'stack_vs_sigma_many.csv', delimiter=',', names=True, encoding='utf-8')

r = data['r_over_R200']
dv_obs = data['delta_v_obs_kms']
dv_err = data['delta_v_err_kms']
dv_pred = data['delta_v_pred_kms']

fig, ax = plt.subplots(figsize=(8, 6))

# Observed stack with error bars
ax.errorbar(r, dv_obs, yerr=dv_err, fmt='o', color='navy', markersize=8, 
            linewidth=2, capsize=5, label='Observed stack (32 clusters)')

# Σ prediction (BCG-relative)
ax.plot(r, dv_pred, 's-', color='crimson', markersize=7, linewidth=2, 
        label='Σ-Gravity prediction (Hernquist toy)')

ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('r / R200', fontsize=12)
ax.set_ylabel('Δv (km/s)', fontsize=12)
ax.set_title('Stacked Gravitational Redshift: Observed vs Σ-Gravity (BCG-relative)', fontsize=13)
ax.legend(fontsize=10, frameon=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_fig = OUTDIR / 'stack_vs_sigma_many.png'
fig.savefig(out_fig, dpi=150)
print(f"Saved plot to {out_fig}")
plt.close()
