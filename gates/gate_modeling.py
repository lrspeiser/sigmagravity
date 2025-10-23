"""
Gate Modeling and Visualization Tool

Explore how gate parameters affect behavior and suggest parameter values
for different physical scenarios (bulges, bars, solar system).

Usage:
    python gate_modeling.py
    
Generates: outputs/gate_functions.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_core import (
    G_distance, G_acceleration, G_bulge_exponential,
    G_unified, G_solar_system, C_burr_XII
)


class GateModeler:
    """
    Explore and visualize gate behavior
    """
    
    def __init__(self, l0=5.0):
        """
        Parameters
        ----------
        l0 : float
            Coherence length (kpc) from your paper
        """
        self.l0 = l0
        self.AU_in_kpc = 4.848136811e-9
    
    def plot_gate_behavior(self, save_path='outputs/gate_functions.png'):
        """
        Comprehensive visualization of all gate types
        
        Generates 6-panel figure showing:
        1. Distance gate parameter sweep
        2. Acceleration gate parameter sweep
        3. Exponential gate parameter sweep
        4. Combined distance+acceleration
        5. Solar system safety
        6. Full kernel with gates
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Distance gate (varying alpha)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_distance_gate_sweep(ax1)
        
        # Panel 2: Distance gate (varying beta)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_distance_gate_beta_sweep(ax2)
        
        # Panel 3: Acceleration gate
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_acceleration_gate(ax3)
        
        # Panel 4: Exponential gate
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_exponential_gate(ax4)
        
        # Panel 5: Solar system safety
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_solar_system_safety(ax5)
        
        # Panel 6: Unified gate (2D heatmap)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_unified_gate_2d(ax6)
        
        # Panel 7: Full kernel (C × G)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_full_kernel(ax7)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
        plt.close()
    
    def _plot_distance_gate_sweep(self, ax):
        """Vary steepness parameter alpha"""
        R = np.logspace(-2, 2, 300)
        R_min = 1.0
        beta = 1.0
        
        alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        for alpha in alphas:
            G = G_distance(R, R_min, alpha, beta)
            ax.plot(R, G, label=f'α={alpha:.1f}')
        
        ax.axvline(R_min, color='gray', linestyle='--', alpha=0.3, label=f'R_min={R_min}')
        ax.set_xscale('log')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('G(R)')
        ax.set_title('Distance Gate: Varying α (steepness)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    def _plot_distance_gate_beta_sweep(self, ax):
        """Vary strength parameter beta"""
        R = np.logspace(-2, 2, 300)
        R_min = 1.0
        alpha = 2.0
        
        betas = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        for beta in betas:
            G = G_distance(R, R_min, alpha, beta)
            ax.plot(R, G, label=f'β={beta:.1f}')
        
        ax.axvline(R_min, color='gray', linestyle='--', alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('G(R)')
        ax.set_title('Distance Gate: Varying β (strength)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    def _plot_acceleration_gate(self, ax):
        """Acceleration-based suppression"""
        g = np.logspace(-12, -8, 300)
        g_crit = 1e-10
        
        alphas = [1.5, 2.0, 2.5, 3.0]
        beta = 1.0
        
        for alpha in alphas:
            G = G_acceleration(g, g_crit, alpha, beta)
            ax.plot(g, G, label=f'α={alpha:.1f}')
        
        ax.axvline(g_crit, color='gray', linestyle='--', alpha=0.3, label=f'g_crit={g_crit:.1e}')
        ax.set_xscale('log')
        ax.set_xlabel('g_bar (m/s²)')
        ax.set_ylabel('G(g)')
        ax.set_title('Acceleration Gate (bulge/dense regions)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    def _plot_exponential_gate(self, ax):
        """Exponential bulge gate"""
        R = np.linspace(0, 10, 300)
        
        # Vary R_bulge
        R_bulge_vals = [0.5, 1.0, 2.0, 3.0]
        alpha, beta = 2.0, 1.0
        
        for R_bulge in R_bulge_vals:
            G = G_bulge_exponential(R, R_bulge, alpha, beta)
            ax.plot(R, G, label=f'R_bulge={R_bulge:.1f} kpc')
        
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('G(R)')
        ax.set_title('Exponential Gate (measured R_bulge)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    def _plot_solar_system_safety(self, ax):
        """Show strong suppression at AU scales"""
        R_AU = np.logspace(0, 5, 300)  # AU
        R_kpc = R_AU * self.AU_in_kpc
        
        G = G_solar_system(R_kpc, R_min_AU=1.0, alpha=4.0, beta=2.0)
        
        ax.plot(R_AU, G, linewidth=2, color='red', label='Solar System Gate')
        
        # Mark key scales
        ax.axvline(1, color='gray', linestyle='--', alpha=0.5, label='1 AU')
        ax.axvline(100, color='gray', linestyle=':', alpha=0.5, label='100 AU')
        ax.axvline(10000, color='gray', linestyle='-.', alpha=0.5, label='10⁴ AU')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Distance (AU)')
        ax.set_ylabel('G(R)')
        ax.set_title('Solar System Safety (PPN constraint)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([1e-20, 2])
        
        # Annotate G values
        for R_val in [1, 100, 10000]:
            idx = np.argmin(np.abs(R_AU - R_val))
            ax.text(R_val, G[idx]*2, f'{G[idx]:.1e}', 
                   fontsize=7, ha='center')
    
    def _plot_unified_gate_2d(self, ax):
        """2D heatmap of unified gate G(R, g_bar)"""
        R = np.logspace(-1, 2, 100)
        g = np.logspace(-12, -8, 100)
        RR, GG = np.meshgrid(R, g)
        
        G_2d = G_unified(RR, GG, R_min=1.0, g_crit=1e-10,
                        alpha_R=2.0, beta_R=1.0, 
                        alpha_g=2.0, beta_g=1.0)
        
        im = ax.contourf(R, g, G_2d, levels=20, cmap='viridis')
        ax.contour(R, g, G_2d, levels=[0.1, 0.5, 0.9], 
                  colors='white', linestyles='--', linewidths=1)
        
        plt.colorbar(im, ax=ax, label='G(R, g_bar)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('g_bar (m/s²)')
        ax.set_title('Unified Gate: Distance × Acceleration')
        
        # Mark transition lines
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(1e-10, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    def _plot_full_kernel(self, ax):
        """Show full Σ-Gravity kernel K(R) = A × C(R) × G(R)"""
        R = np.logspace(-3, 2, 500)
        g_bar = 1e-10 / R**2  # Toy model
        
        # Components
        A = 0.6
        C = C_burr_XII(R, self.l0, p=2.0, n_coh=1.0)
        G_dist = G_distance(R, R_min=1.0, alpha=2.0, beta=1.0)
        G_solar = G_solar_system(R)
        
        K = A * C * G_dist * G_solar
        
        # Plot components
        ax.plot(R, C, label='C(R) - Coherence Window', linestyle='--', alpha=0.7)
        ax.plot(R, G_dist, label='G_dist(R) - Distance Gate', linestyle='--', alpha=0.7)
        ax.plot(R, G_dist * G_solar, label='G_dist × G_solar', linestyle=':', alpha=0.7)
        ax.plot(R, K, label='K(R) = A × C × G_dist × G_solar', 
               linewidth=2.5, color='black')
        
        # Mark scales
        ax.axvline(self.l0, color='gray', linestyle='--', alpha=0.3, 
                  label=f'ℓ₀={self.l0} kpc')
        ax.axvline(1.0, color='blue', linestyle='--', alpha=0.3, 
                  label='R_min=1 kpc')
        
        # Mark AU scale (solar system)
        ax.axvline(1000*self.AU_in_kpc, color='red', linestyle='--', alpha=0.3, 
                  label='1000 AU')
        
        ax.set_xscale('log')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('Value')
        ax.set_title('Full Σ-Gravity Kernel: K(R) = A · C(R; ℓ₀) · G(R)')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    def suggest_parameters(self, scenario='bulge'):
        """
        Suggest parameter values for different physical scenarios
        
        Parameters
        ----------
        scenario : str
            'bulge', 'bar', 'solar_system', 'disk_inner', 'disk_outer'
        
        Returns
        -------
        params : dict
            Suggested parameter values with reasoning
        """
        suggestions = {
            'bulge': {
                'gate_type': 'exponential',
                'R_bulge': 1.5,  # kpc, from Sérsic fit
                'alpha': 2.0,
                'beta': 1.0,
                'reasoning': 'Use measured R_bulge from surface brightness profile'
            },
            'bar': {
                'gate_type': 'distance',
                'R_min': 3.0,  # kpc, typical bar length
                'alpha': 2.0,
                'beta': 1.5,
                'reasoning': 'Suppress in bar region where orbits are non-circular'
            },
            'solar_system': {
                'gate_type': 'distance',
                'R_min': 0.0001,  # kpc ≈ 100 AU
                'alpha': 4.0,  # Steep suppression
                'beta': 2.0,   # Strong suppression
                'reasoning': 'MUST satisfy PPN constraints: K(1 AU) < 10^-14'
            },
            'disk_inner': {
                'gate_type': 'unified',
                'R_min': 0.5,  # kpc
                'g_crit': 5e-10,  # m/s² (higher than outer disk)
                'alpha_R': 2.0,
                'beta_R': 1.0,
                'alpha_g': 2.0,
                'beta_g': 1.0,
                'reasoning': 'Both distance and acceleration suppress in inner disk'
            },
            'disk_outer': {
                'gate_type': 'distance',
                'R_min': 0.5,
                'alpha': 1.5,  # Gentler transition
                'beta': 0.5,   # Weaker suppression
                'reasoning': 'Mainly distance-based, allow coherence at large R'
            }
        }
        
        if scenario not in suggestions:
            available = ', '.join(suggestions.keys())
            raise ValueError(f"Unknown scenario. Available: {available}")
        
        return suggestions[scenario]
    
    def fit_gate_to_requirement(self, R_transition=2.0, suppression_strength=0.5):
        """
        Given a requirement "G should be ≈ 0.5 at R_transition",
        suggest parameters
        
        Parameters
        ----------
        R_transition : float
            Radius where you want G ≈ 0.5 (kpc)
        suppression_strength : float
            How strong the suppression should be (0 to 1)
            0.5 → gentle, 1.0 → standard, 2.0 → strong
        
        Returns
        -------
        params : dict
            Suggested R_min, alpha, beta
        """
        # For distance gate: G(R_min) ≈ 0.5 when beta ≈ 1, alpha ≈ 2
        # Heuristic: R_min ≈ R_transition / 1.5
        
        R_min = R_transition / 1.5
        alpha = 2.0
        beta = suppression_strength
        
        # Verify
        G_check = G_distance(R_transition, R_min, alpha, beta)
        
        return {
            'R_min': R_min,
            'alpha': alpha,
            'beta': beta,
            'verification': f'G({R_transition:.1f} kpc) = {G_check:.3f}'
        }


def main():
    """Generate gate behavior plots"""
    
    print("Gate Modeling Tool")
    print("=" * 60)
    
    # Create output directory if needed
    os.makedirs('outputs', exist_ok=True)
    
    # Initialize
    modeler = GateModeler(l0=5.0)
    
    # Generate comprehensive plot
    print("\nGenerating gate behavior visualization...")
    modeler.plot_gate_behavior('outputs/gate_functions.png')
    
    # Print parameter suggestions
    print("\n" + "=" * 60)
    print("PARAMETER SUGGESTIONS")
    print("=" * 60)
    
    scenarios = ['bulge', 'bar', 'solar_system', 'disk_inner', 'disk_outer']
    for scenario in scenarios:
        params = modeler.suggest_parameters(scenario)
        print(f"\n{scenario.upper()}:")
        print(f"  Gate type: {params['gate_type']}")
        for key, val in params.items():
            if key not in ['gate_type', 'reasoning']:
                print(f"  {key} = {val}")
        print(f"  Reasoning: {params['reasoning']}")
    
    # Example: custom requirement
    print("\n" + "=" * 60)
    print("CUSTOM REQUIREMENT EXAMPLE")
    print("=" * 60)
    print("\nRequirement: G should be approx 0.5 at R = 3 kpc")
    custom = modeler.fit_gate_to_requirement(R_transition=3.0, suppression_strength=1.0)
    for key, val in custom.items():
        print(f"  {key}: {val}")
    
    print("\n[OK] Complete! Check outputs/gate_functions.png")
    print("\nNext steps:")
    print("1. Review parameter suggestions above")
    print("2. Adapt gate_fitting_tool.py to your rotation curve data")
    print("3. Run pytest tests/test_section2_invariants.py")


if __name__ == '__main__':
    main()

