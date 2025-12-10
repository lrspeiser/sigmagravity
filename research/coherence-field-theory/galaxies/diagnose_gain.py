"""
Diagnose why gain function is zero.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from galaxies.resonant_halo_solver import (
    ResonantParams, gain_function, toomre_Q, kappa_from_Omega
)


def diagnose_galaxy(galaxy_name):
    """Check all gain gates for a galaxy."""
    print("\n" + "="*80)
    print(f"DIAGNOSING: {galaxy_name}")
    print("="*80)
    
    # Load data
    loader = RealDataLoader()
    gal = loader.load_rotmod_galaxy(galaxy_name)
    
    r = gal['r']
    v_obs = gal['v_obs']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    
    # Estimate dispersion (dwarf = cold)
    sigma_v = 15.0  # km/s
    
    # Compute surface density (simple estimate)
    # For velocity v, mass M = v² r / G
    G_kpc = 4.302e-3  # kpc (M☉)^-1 (km/s)²
    v_bar = np.sqrt(v_disk**2 + v_gas**2)
    M_enc = r * v_bar**2 / G_kpc
    dM_dr = np.gradient(M_enc, r)
    Sigma_b = np.maximum(dM_dr / (2 * np.pi * r), 1e-5)
    
    # Rotation curve quantities
    Omega = v_obs / r
    ln_Omega = np.log(Omega + 1e-10)
    ln_r = np.log(r)
    dlnOm_dlnr = np.gradient(ln_Omega, ln_r)
    
    print(f"\nGalaxy properties:")
    print(f"  Radius: {r[0]:.2f} - {r[-1]:.2f} kpc")
    print(f"  Velocity: {v_obs.min():.1f} - {v_obs.max():.1f} km/s")
    print(f"  Σ_b: {Sigma_b.min():.1e} - {Sigma_b.max():.1e} M☉/kpc²")
    print(f"  σ_v: {sigma_v} km/s (assumed)")
    print(f"  dlnΩ/dlnr: {dlnOm_dlnr.min():.2f} - {dlnOm_dlnr.max():.2f}")
    
    # Test parameters
    params = ResonantParams(
        m0=0.02,
        R_coh=5.0,
        alpha=1.5,
        lambda_phi=8.0,
        Q_c=1.5,
        Delta_Q=0.2,
        sigma_c=30.0,
        sigma_m=0.25,
        m_max=2
    )
    
    # Compute each gate separately
    g0 = params.alpha / params.R_coh**2
    print(f"\nGain amplitude g₀ = {g0:.4f} kpc⁻²")
    
    # Gate 1: Toomre Q
    Q = toomre_Q(r, Sigma_b, sigma_v * np.ones_like(r), Omega, dlnOm_dlnr)
    S_Q = 0.5 * (1.0 + np.tanh((params.Q_c - Q) / params.Delta_Q))
    
    print(f"\nGate 1: Toomre Q")
    print(f"  Q range: {np.nanmin(Q):.2f} - {np.nanmax(Q):.2f}")
    print(f"  Q < Q_c points: {np.sum(Q < params.Q_c)} / {len(Q)}")
    print(f"  S_Q range: {S_Q.min():.4f} - {S_Q.max():.4f}")
    print(f"  S_Q > 0.1 points: {np.sum(S_Q > 0.1)} / {len(S_Q)}")
    
    # Gate 2: Dispersion
    S_sigma = np.exp(-(sigma_v / params.sigma_c)**2)
    print(f"\nGate 2: Dispersion")
    print(f"  σ_v / σ_c = {sigma_v / params.sigma_c:.2f}")
    print(f"  S_σ = {S_sigma:.4f} (uniform)")
    
    # Gate 3: Resonance
    x = (2.0 * np.pi * r) / params.lambda_phi
    S_res = np.zeros_like(r)
    for m in range(1, params.m_max + 1):
        S_res += np.exp(-((x - m)**2) / (2.0 * params.sigma_m**2))
    
    print(f"\nGate 3: Resonance")
    print(f"  x = 2πr/λ_φ range: {x.min():.2f} - {x.max():.2f}")
    print(f"  Expected peaks at m=1, 2")
    print(f"  S_res range: {S_res.min():.4f} - {S_res.max():.4f}")
    print(f"  S_res > 0.5 points: {np.sum(S_res > 0.5)} / {len(S_res)}")
    
    # Total gain
    g = g0 * S_Q * S_sigma * S_res
    g = np.nan_to_num(g, nan=0.0)
    
    print(f"\nTotal gain g(r):")
    print(f"  g range: {g.min():.4f} - {g.max():.4f} kpc⁻²")
    print(f"  g > 0 points: {np.sum(g > 0)} / {len(g)}")
    print(f"  Tachyonic (g > m₀²={params.m0**2:.4f}): {np.sum(g > params.m0**2)} / {len(g)}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{galaxy_name} - Gain Diagnostics', fontsize=14, fontweight='bold')
    
    # 1. Toomre Q and gate
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(r, Q, 'b-', linewidth=2, label='Q')
    ax.axhline(params.Q_c, color='r', linestyle='--', alpha=0.5, label=f'Q_c={params.Q_c}')
    ax2.plot(r, S_Q, 'g-', linewidth=2, label='S_Q')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Toomre Q', color='b')
    ax2.set_ylabel('Gate S_Q', color='g')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Gate 1: Coldness (Toomre Q)')
    
    # 2. Resonance gate
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(r, x, 'b-', linewidth=2, label='x = 2πr/λ_φ')
    for m in range(1, params.m_max + 1):
        ax.axhline(m, color='r', linestyle='--', alpha=0.5, label=f'm={m}')
    ax2.plot(r, S_res, 'orange', linewidth=2, label='S_res')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Mode number x', color='b')
    ax2.set_ylabel('Gate S_res', color='orange')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Gate 3: Resonance')
    
    # 3. All gates combined
    ax = axes[1, 0]
    ax.plot(r, S_Q, 'b-', label='S_Q (coldness)', linewidth=2)
    ax.axhline(S_sigma, color='g', linestyle='--', label=f'S_σ = {S_sigma:.2f}', linewidth=2)
    ax.plot(r, S_res, 'orange', label='S_res (resonance)', linewidth=2)
    ax.plot(r, S_Q * S_sigma * S_res, 'r-', label='Product', linewidth=2, alpha=0.7)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Gate value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('All Gates')
    
    # 4. Final gain
    ax = axes[1, 1]
    ax.plot(r, g, 'purple', linewidth=2, label='g(r)')
    ax.axhline(params.m0**2, color='r', linestyle='--', label=f'm₀² = {params.m0**2:.4f}', alpha=0.5)
    ax.fill_between(r, 0, g, where=(g > params.m0**2), alpha=0.2, color='pink', label='Tachyonic')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Gain g(r) (kpc⁻²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Total Gain')
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'gain_diagnostics')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{galaxy_name}_gain_diagnostic.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")
    
    plt.show()


if __name__ == '__main__':
    # Test on DDO154 (known dwarf)
    diagnose_galaxy('DDO154')
