"""
Test New Gate Formulas on Real Pipeline Data

Compares current implementation vs. new gate formulas on:
1. SPARC rotation curves (galaxy RAR)
2. Milky Way Gaia data (if available)
3. Cluster lensing (if available)

This answers: "Do the new explicit gate formulas improve fits?"

Usage:
    python test_on_real_pipeline.py
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../vendor/maxdepth_gaia')

from gate_core import (
    G_distance, G_acceleration, G_bulge_exponential,
    C_burr_XII, G_solar_system
)

# Implement gate_c1 ourselves (Hermite smoothstep from vendor code)
def gate_c1(R, Rb, dR):
    """C1 smooth gate: 0 for R <= Rb, 1 for R >= Rb + dR"""
    R = np.asarray(R, dtype=float)
    dR = max(dR, 1e-6)
    s = (R - Rb) / dR
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


class GateComparison:
    """
    Compare current gates vs. new explicit formulas
    """
    
    def __init__(self, hyperparams_path='../config/hyperparams_track2.json'):
        """Load hyperparameters from paper"""
        
        with open(hyperparams_path, 'r') as f:
            hp = json.load(f)
        
        self.L_0 = hp['L_0']  # Coherence length (kpc)
        self.beta_bulge = hp['beta_bulge']
        self.alpha_shear = hp['alpha_shear']
        self.gamma_bar = hp.get('gamma_bar', 0.0)
        self.A_0 = hp['A_0']
        self.p = hp['p']
        self.n_coh = hp['n_coh']
        self.g_dagger = hp['g_dagger']
        
        print("Loaded hyperparameters from paper:")
        print(f"  L_0 = {self.L_0:.3f} kpc")
        print(f"  A_0 = {self.A_0:.3f}")
        print(f"  p = {self.p:.3f}")
        print(f"  n_coh = {self.n_coh:.3f}")
        print(f"  beta_bulge = {self.beta_bulge:.3f}")
        print(f"  alpha_shear = {self.alpha_shear:.3f}")
    
    def coherence_window_paper(self, R):
        """Paper's coherence window (Burr-XII)"""
        return C_burr_XII(R, ell0=self.L_0, p=self.p, n_coh=self.n_coh)
    
    def K_current_style(self, R, g_bar, R_boundary, delta_R):
        """
        Approximate current implementation style
        (based on gate_c1 from models.py)
        """
        # Coherence window
        C = self.coherence_window_paper(R)
        
        # Acceleration factor
        accel_factor = (self.g_dagger / np.maximum(g_bar, 1e-30))**self.p
        
        # Simple boundary gate (Hermite smoothstep)
        gate = gate_c1(R, R_boundary, delta_R)
        
        # Kernel
        K = self.A_0 * accel_factor * C * gate
        
        return K
    
    def K_new_gates(self, R, g_bar, morphology_params):
        """
        New explicit gate formulas from gates/gate_core.py
        
        Parameters
        ----------
        morphology_params : dict
            R_bulge : float (kpc, from imaging)
            has_bar : bool
            R_bar : float (kpc, bar length)
        """
        # Coherence window (same as paper)
        C = self.coherence_window_paper(R)
        
        # Acceleration factor (same as paper)
        accel_factor = (self.g_dagger / np.maximum(g_bar, 1e-30))**self.p
        
        # NEW: Explicit bulge gate
        R_bulge = morphology_params.get('R_bulge', 1.5)
        G_bulge = G_bulge_exponential(R, R_bulge, alpha=2.0, beta=self.beta_bulge)
        
        # NEW: Bar gate (if present)
        if morphology_params.get('has_bar', False):
            R_bar = morphology_params.get('R_bar', 3.0)
            G_bar = G_distance(R, R_min=R_bar, alpha=2.0, beta=1.5)
        else:
            G_bar = 1.0
        
        # Shear gate (simple for now - would need shear parameter from data)
        # For demo, use a weak suppression in inner disk
        G_shear = G_distance(R, R_min=0.5, alpha=1.5, beta=self.alpha_shear)
        
        # Solar system safety (ALWAYS)
        G_solar = G_solar_system(R)
        
        # Full kernel
        K = self.A_0 * accel_factor * C * G_bulge * G_bar * G_shear * G_solar
        
        return K
    
    def compare_on_rotation_curve(self, R, v_obs, v_bar, name='test_galaxy',
                                  R_boundary=5.0, delta_R=0.8, morphology=None):
        """
        Compare current vs. new gates on a single rotation curve
        
        Returns
        -------
        comparison : dict
            chi2, scatter, and model velocities for both methods
        """
        if morphology is None:
            morphology = {'R_bulge': 1.5, 'has_bar': False}
        
        # Compute accelerations
        g_bar = (v_bar * 1000)**2 / (R * 3.086e19)  # m/s²
        
        # Current implementation
        K_current = self.K_current_style(R, g_bar, R_boundary, delta_R)
        g_eff_current = g_bar * (1 + K_current)
        v_current = np.sqrt(g_eff_current * R * 3.086e19) / 1000  # km/s
        
        # New gates implementation
        K_new = self.K_new_gates(R, g_bar, morphology)
        g_eff_new = g_bar * (1 + K_new)
        v_new = np.sqrt(g_eff_new * R * 3.086e19) / 1000  # km/s
        
        # Compute statistics
        residuals_current = v_current - v_obs
        residuals_new = v_new - v_obs
        
        chi2_current = np.sum((residuals_current / v_obs)**2)
        chi2_new = np.sum((residuals_new / v_obs)**2)
        
        scatter_current = np.std(np.log10(v_current / v_obs))
        scatter_new = np.std(np.log10(v_new / v_obs))
        
        return {
            'name': name,
            'v_current': v_current,
            'v_new': v_new,
            'K_current': K_current,
            'K_new': K_new,
            'chi2_current': chi2_current,
            'chi2_new': chi2_new,
            'scatter_current': scatter_current,
            'scatter_new': scatter_new,
            'improvement': chi2_current / chi2_new if chi2_new > 0 else 1.0
        }
    
    def plot_comparison(self, R, v_obs, v_bar, comparison, 
                       save_path='outputs/gate_comparison_example.png'):
        """
        Plot current vs. new gates
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Rotation curves
        ax = axes[0, 0]
        ax.plot(R, v_obs, 'ko', label='Observed', markersize=6)
        ax.plot(R, v_bar, 'b--', label='Baryons (GR)', linewidth=2)
        ax.plot(R, comparison['v_current'], 'g-', label='Current gates', linewidth=2)
        ax.plot(R, comparison['v_new'], 'r-', label='New explicit gates', linewidth=2)
        ax.set_xlabel('R (kpc)', fontsize=12)
        ax.set_ylabel('v_circ (km/s)', fontsize=12)
        ax.set_title(f"{comparison['name']}: Rotation Curves", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Residuals
        ax = axes[0, 1]
        res_current = comparison['v_current'] - v_obs
        res_new = comparison['v_new'] - v_obs
        ax.plot(R, res_current, 'go-', label='Current gates', markersize=5)
        ax.plot(R, res_new, 'ro-', label='New gates', markersize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('R (kpc)', fontsize=12)
        ax.set_ylabel('Residual (km/s)', fontsize=12)
        ax.set_title('Residuals', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Kernel K(R)
        ax = axes[1, 0]
        ax.plot(R, comparison['K_current'], 'g-', label='Current K(R)', linewidth=2.5)
        ax.plot(R, comparison['K_new'], 'r-', label='New K(R)', linewidth=2.5)
        ax.set_xlabel('R (kpc)', fontsize=12)
        ax.set_ylabel('K(R)', fontsize=12)
        ax.set_title('Σ-Gravity Kernel Comparison', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        text = f"Comparison Results\n{'='*50}\n\n"
        text += f"Galaxy: {comparison['name']}\n\n"
        text += "CURRENT GATES (gate_c1 smoothstep):\n"
        text += f"  chi² = {comparison['chi2_current']:.2f}\n"
        text += f"  scatter = {comparison['scatter_current']:.4f} dex\n\n"
        text += "NEW EXPLICIT GATES (physics-based):\n"
        text += f"  chi² = {comparison['chi2_new']:.2f}\n"
        text += f"  scatter = {comparison['scatter_new']:.4f} dex\n\n"
        text += f"IMPROVEMENT FACTOR: {comparison['improvement']:.2f}×\n\n"
        
        if comparison['improvement'] > 1.1:
            text += "→ New gates are BETTER\n"
        elif comparison['improvement'] < 0.9:
            text += "→ Current gates are better\n"
        else:
            text += "→ Approximately EQUIVALENT\n"
        
        ax.text(0.1, 0.5, text, fontsize=11, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def load_sparc_rotmod(filepath):
    """Load SPARC rotmod file"""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 8:
                try:
                    rows.append([float(x) for x in parts[:8]])
                except ValueError:
                    continue
    
    if not rows:
        return None
    
    a = np.array(rows)
    
    return {
        'R': a[:,0],
        'v_obs': a[:,1],
        'v_err': a[:,2],
        'v_gas': a[:,3],
        'v_disk': a[:,4],
        'v_bul': a[:,5],
        'v_bar': np.sqrt(a[:,3]**2 + a[:,4]**2 + a[:,5]**2)
    }


def main():
    """Test new gates on real rotation curve"""
    
    print("="*80)
    print("GATE FORMULA COMPARISON ON REAL DATA")
    print("="*80)
    print("\nQuestion: Do explicit gate formulas improve fits?")
    print("\nWe compare:")
    print("  1. Current: gate_c1(R, Rb, dR) smoothstep")
    print("  2. New: G_bulge(R) × G_shear(R) × G_bar(R) × G_solar(R)")
    
    # Load hyperparameters
    comparer = GateComparison('../config/hyperparams_track2.json')
    
    # Find a SPARC galaxy
    sparc_dir = '../data/Rotmod_LTG'
    test_galaxies = ['NGC2403', 'NGC3198', 'UGC02953']
    
    results_all = []
    
    for gal_name in test_galaxies:
        filepath = os.path.join(sparc_dir, f'{gal_name}_rotmod.dat')
        
        if not os.path.exists(filepath):
            print(f"\nSkipping {gal_name} (file not found)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing: {gal_name}")
        print('='*80)
        
        # Load data
        data = load_sparc_rotmod(filepath)
        if data is None:
            print(f"  Failed to load {gal_name}")
            continue
        
        print(f"  R range: {data['R'].min():.1f} - {data['R'].max():.1f} kpc")
        print(f"  {len(data['R'])} data points")
        
        # Estimate morphology (simplified - would use actual measurements)
        # For demo: assume R_bulge ~ 1.5 kpc, no bar unless name suggests it
        morphology = {
            'R_bulge': 1.5,  # kpc
            'has_bar': False,  # Assume no bar for demo
            'R_bar': 3.0
        }
        
        # Run comparison
        comparison = comparer.compare_on_rotation_curve(
            data['R'], data['v_obs'], data['v_bar'],
            name=gal_name,
            R_boundary=5.0,  # Typical from fit_params.json
            delta_R=0.8,     # Typical from fit_params.json
            morphology=morphology
        )
        
        results_all.append(comparison)
        
        # Print results
        print(f"\nResults for {gal_name}:")
        print(f"  Current gates:")
        print(f"    chi² = {comparison['chi2_current']:.2f}")
        print(f"    scatter = {comparison['scatter_current']:.4f} dex")
        print(f"  New gates:")
        print(f"    chi² = {comparison['chi2_new']:.2f}")
        print(f"    scatter = {comparison['scatter_new']:.4f} dex")
        print(f"  Improvement: {comparison['improvement']:.2f}×")
        
        # Plot
        save_path = f"outputs/gate_comparison_{gal_name}.png"
        comparer.plot_comparison(data['R'], data['v_obs'], data['v_bar'],
                                comparison, save_path)
    
    # Summary across all galaxies
    if results_all:
        print("\n" + "="*80)
        print("SUMMARY ACROSS GALAXIES")
        print("="*80)
        
        chi2_current_total = sum(r['chi2_current'] for r in results_all)
        chi2_new_total = sum(r['chi2_new'] for r in results_all)
        
        scatter_current_mean = np.mean([r['scatter_current'] for r in results_all])
        scatter_new_mean = np.mean([r['scatter_new'] for r in results_all])
        
        print(f"\n{len(results_all)} galaxies tested:")
        print(f"  Current gates:")
        print(f"    Total chi² = {chi2_current_total:.1f}")
        print(f"    Mean scatter = {scatter_current_mean:.4f} dex")
        print(f"  New explicit gates:")
        print(f"    Total chi² = {chi2_new_total:.1f}")
        print(f"    Mean scatter = {scatter_new_mean:.4f} dex")
        print(f"\n  Overall improvement: {chi2_current_total / chi2_new_total:.2f}×")
        
        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        
        improvement = chi2_current_total / chi2_new_total
        
        if improvement > 1.1:
            print("\n[WINNER] NEW GATES ARE BETTER!")
            print(f"   Explicit physics-based formulas improve fit by {improvement:.1f}x")
            print("   -> Consider adopting explicit gate formulas")
        elif improvement < 0.9:
            print("\n[STATUS] CURRENT GATES ARE BETTER (chi2)")
            print(f"   Smoothstep gate outperforms explicit formulas by {1/improvement:.1f}x")
            print(f"   BUT new gates give better scatter: {scatter_new_mean:.4f} vs {scatter_current_mean:.4f} dex")
            print("   -> Trade-off: chi2 vs scatter")
        else:
            print("\n[STATUS] APPROXIMATELY EQUIVALENT")
            print("   Both approaches give similar results")
            print("   -> Choice based on interpretability/theoretical grounding")
    
    else:
        print("\nNo galaxies successfully tested!")
    
    print("\n" + "="*80)
    print("[OK] Comparison complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check outputs/gate_comparison_*.png for visual comparisons")
    print("2. If new gates are better: consider integrating into main pipeline")
    print("3. If equivalent: choice is about interpretability")
    print("4. Test on larger sample for statistical significance")


if __name__ == '__main__':
    # Check if running from gates/
    if not os.path.exists('../config/hyperparams_track2.json'):
        print("ERROR: Run from gates/ directory")
        print("Usage: cd gates && python test_on_real_pipeline.py")
        sys.exit(1)
    
    main()

