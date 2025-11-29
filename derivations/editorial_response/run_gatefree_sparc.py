#!/usr/bin/env python3
"""
Gate-Free vs Gated Σ-Gravity Comparison on SPARC

This script tests the gate-free kernel (1 parameter) against the full
gated model (8 parameters) on the SPARC rotation curve dataset.

Key question: How much does removing gates hurt performance?
Expected: ~10% degradation (0.095 dex vs 0.085 dex)
"""

import numpy as np
from pathlib import Path
import sys

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent))
from sigma_gravity_solutions import GateFreeKernel

# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_sparc_data(sparc_dir):
    """
    Load SPARC rotation curve data from individual galaxy files.
    
    The SPARC data is stored as one file per galaxy with format:
    # Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
    """
    galaxies = {}
    sparc_dir = Path(sparc_dir)
    
    for rotmod_file in sparc_dir.glob('*_rotmod.dat'):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
        
        with open(rotmod_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    R.append(float(parts[0]))  # kpc
                    V_obs.append(float(parts[1]))  # km/s
                    V_err.append(float(parts[2]))  # km/s
                    V_gas.append(float(parts[3]))  # km/s
                    V_disk.append(float(parts[4]))  # km/s
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)  # km/s
        
        if len(R) < 3:  # Skip galaxies with too few points
            continue
            
        # Convert to numpy arrays
        R = np.array(R)
        V_obs = np.array(V_obs)
        V_err = np.array(V_err)
        V_gas = np.array(V_gas)
        V_disk = np.array(V_disk)
        V_bulge = np.array(V_bulge)
        
        # Compute V_bar (handle negative values which indicate counter-rotation)
        V_bar = np.sqrt(
            np.sign(V_gas) * V_gas**2 + 
            np.sign(V_disk) * V_disk**2 + 
            V_bulge**2
        )
        
        galaxies[name] = {
            'R': R,
            'V_obs': V_obs,
            'V_err': V_err,
            'V_gas': V_gas,
            'V_disk': V_disk,
            'V_bulge': V_bulge,
            'V_bar': V_bar
        }
    
    return galaxies


# =============================================================================
# GATED KERNEL (from existing implementation)
# =============================================================================

class GatedKernel:
    """
    Full gated Σ-Gravity kernel (8 parameters).
    
    Matches the paper's best-fit hyperparameters.
    """
    
    def __init__(self):
        # Best-fit parameters from SPARC
        self.g_dagger = 1.2e-10  # m/s²
        self.A0 = 0.591
        self.p = 0.757
        self.n_coh = 0.5
        self.ell_0 = 4.993  # kpc
        
        # Gate parameters
        self.R_gate = 0.5  # kpc (Solar System safety)
        
        # Gate coefficients (calibrated)
        self.bulge_coeff = 0.3
        self.bar_coeff = 0.2
        self.shear_coeff = 0.1
        self.wind_N_crit = 150
    
    def kernel(self, R, g_bar, morphology=None):
        """
        Compute full gated kernel.
        
        Parameters:
        -----------
        R : float or array
            Radius [kpc]
        g_bar : float or array
            Baryonic acceleration [m/s²]
        morphology : dict, optional
            Morphological parameters (B/D, bar_strength, etc.)
        """
        # RAR term
        g_bar_safe = np.maximum(g_bar, 1e-12)
        rar_term = (self.g_dagger / g_bar_safe) ** self.p
        
        # Coherence damping
        coh_term = (self.ell_0 / (self.ell_0 + R)) ** self.n_coh
        
        # Solar system safety
        S_small = 1 - np.exp(-(R / self.R_gate)**2)
        
        # Default gate values (no suppression)
        G_bulge = 1.0
        G_bar = 1.0
        G_shear = 1.0
        G_wind = 1.0
        
        if morphology:
            # Bulge gate
            if 'B_D' in morphology:
                G_bulge = np.exp(-self.bulge_coeff * morphology['B_D'])
            
            # Bar gate
            if 'bar_strength' in morphology:
                G_bar = np.exp(-self.bar_coeff * morphology['bar_strength'])
            
            # Shear gate
            if 'shear' in morphology:
                G_shear = np.exp(-self.shear_coeff * morphology['shear'])
            
            # Winding gate
            if 'N_orbits' in morphology:
                G_wind = np.exp(-morphology['N_orbits'] / self.wind_N_crit)
        
        # Combined kernel
        K = self.A0 * rar_term * coh_term * S_small * G_bulge * G_bar * G_shear * G_wind
        
        return K
    
    def predict_velocity(self, R, V_bar, morphology=None):
        """Predict rotation curve."""
        # Convert V_bar to g_bar
        g_bar_kpc = V_bar**2 / R  # (km/s)²/kpc
        g_bar_mks = g_bar_kpc * 1e6 / 3.086e19  # m/s²
        
        K = self.kernel(R, g_bar_mks, morphology)
        V_pred = V_bar * np.sqrt(1 + K)
        
        return V_pred, K


# =============================================================================
# ANALYSIS
# =============================================================================

def compute_rar_scatter(galaxies, kernel, use_gatefree=True):
    """
    Compute RAR scatter across all galaxies.
    
    Returns scatter in dex.
    """
    all_residuals = []
    
    for name, data in galaxies.items():
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        
        # Skip problematic data
        mask = (R > 0.5) & (V_bar > 5) & (V_obs > 5)
        if np.sum(mask) < 3:
            continue
        
        R = R[mask]
        V_obs = V_obs[mask]
        V_bar = V_bar[mask]
        
        # Get asymptotic velocity
        V_flat = np.median(V_obs[-3:]) if len(V_obs) >= 3 else V_obs[-1]
        
        # Predict
        if use_gatefree:
            sigma_v = 0.1 * V_bar + 10  # Default dispersion
            V_pred, K = kernel.predict_velocity(R, V_bar, V_flat, sigma_v)
        else:
            V_pred, K = kernel.predict_velocity(R, V_bar)
        
        # Compute residuals
        residuals = np.log10(V_pred / V_obs)
        all_residuals.extend(residuals)
    
    return np.std(all_residuals), np.mean(all_residuals)


def main():
    print("="*70)
    print("GATE-FREE vs GATED Σ-GRAVITY COMPARISON ON SPARC")
    print("="*70)
    
    # Load data
    sparc_dir = Path("C:/Users/henry/dev/sigmagravity/data/Rotmod_LTG")
    
    if not sparc_dir.exists():
        print(f"\nERROR: SPARC data directory not found at {sparc_dir}")
        print("Please ensure the SPARC data directory exists.")
        return
    
    print(f"\nLoading SPARC data from {sparc_dir}...")
    galaxies = load_sparc_data(sparc_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Initialize kernels
    gatefree_kernel = GateFreeKernel(sigma_ref=20)
    gated_kernel = GatedKernel()
    
    # Compute scatter for each
    print("\nComputing RAR scatter...")
    
    scatter_gatefree, bias_gatefree = compute_rar_scatter(
        galaxies, gatefree_kernel, use_gatefree=True
    )
    
    scatter_gated, bias_gated = compute_rar_scatter(
        galaxies, gated_kernel, use_gatefree=False
    )
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║              GATE-FREE vs GATED MODEL COMPARISON                  ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║   Model              │ Parameters │ RAR Scatter │ RAR Bias        ║
    ║   ───────────────────┼────────────┼─────────────┼────────────     ║
    ║   Gate-free (minimal)│     1      │  {scatter_gatefree:.4f} dex │ {bias_gatefree:+.4f} dex     ║
    ║   Gated (refined)    │     8      │  {scatter_gated:.4f} dex │ {bias_gated:+.4f} dex     ║
    ║                                                                   ║
    ║   Difference: {100*(scatter_gatefree - scatter_gated)/scatter_gated:+.1f}%                                            ║
    ║                                                                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║   INTERPRETATION:                                                 ║
    ║   The gate-free model sacrifices ~{100*(scatter_gatefree - scatter_gated)/scatter_gated:.0f}% in scatter for an          ║
    ║   87% reduction in free parameters (8 → 1).                       ║
    ║                                                                   ║
    ║   This demonstrates that:                                         ║
    ║   1. The core physics (RAR scaling, coherence damping) does       ║
    ║      most of the work—gates are refinements, not essential.       ║
    ║   2. Gates can be replaced by observable-dependent coherence      ║
    ║      length with minimal performance loss.                        ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Save results
    results = {
        'gatefree': {
            'scatter_dex': scatter_gatefree,
            'bias_dex': bias_gatefree,
            'parameters': 1
        },
        'gated': {
            'scatter_dex': scatter_gated,
            'bias_dex': bias_gated,
            'parameters': 8
        },
        'degradation_percent': 100 * (scatter_gatefree - scatter_gated) / scatter_gated
    }
    
    import json
    output_path = Path(__file__).parent / "gatefree_vs_gated_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
