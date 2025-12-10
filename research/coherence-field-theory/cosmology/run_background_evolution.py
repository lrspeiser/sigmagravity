"""
Run background evolution and generate analysis plots.
"""

from background_evolution import CoherenceCosmology
import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz

def main():
    print("=" * 70)
    print("COHERENCE FIELD THEORY - BACKGROUND COSMOLOGY")
    print("=" * 70)
    
    # Test different parameter combinations
    param_sets = [
        {'V0': 1.0e-6, 'lambda_param': 1.0, 'label': 'lambda=1.0'},
        {'V0': 1.5e-6, 'lambda_param': 0.8, 'label': 'lambda=0.8'},
        {'V0': 0.8e-6, 'lambda_param': 1.2, 'label': 'lambda=1.2'},
    ]
    
    results_list = []
    
    for params in param_sets:
        print(f"\n{'=' * 70}")
        print(f"Testing parameters: V0={params['V0']:.2e}, lambda={params['lambda_param']:.2f}")
        print(f"{'=' * 70}")
        
        cosmo = CoherenceCosmology(V0=params['V0'], lambda_param=params['lambda_param'])
        results = cosmo.evolve()
        
        print(f"\nDensity parameters at a=1:")
        print(f"  Omega_m0  = {results['Omega_m0']:.4f}")
        print(f"  Omega_phi0  = {results['Omega_phi0']:.4f}")
        
        results['cosmo'] = cosmo
        results['params'] = params
        results_list.append(results)
    
    # Plot comparison across parameter sets
    print(f"\n{'=' * 70}")
    print("Generating comparison plots...")
    print(f"{'=' * 70}")
    
    # H(z) comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    z_vals = np.linspace(0.0, 2.0, 200)
    
    for results in results_list:
        cosmo = results['cosmo']
        label = results['params']['label']
        
        H_z = cosmo.compute_H_of_z(z_vals)
        dL_z = cosmo.compute_dL(z_vals)
        
        axes[0].plot(z_vals, H_z, label=f"Coherence: {label}", linewidth=2)
        axes[1].plot(z_vals, dL_z, label=f"Coherence: {label}", linewidth=2)
    
    # Add ΛCDM reference
    Omega_m_ref = 0.3
    Omega_L_ref = 0.7
    a_sample = 1.0 / (1.0 + z_vals)
    H_LCDM = np.sqrt(Omega_m_ref * a_sample**-3 + Omega_L_ref)
    axes[0].plot(z_vals, H_LCDM, 'k--', label='ΛCDM (0.3/0.7)', linewidth=2, alpha=0.7)
    
    # ΛCDM dL
    dL_LCDM = []
    for z in z_vals:
        zs = np.linspace(0.0, z, 800)
        a_s = 1.0 / (1.0 + zs)
        H_s = np.sqrt(Omega_m_ref * a_s**-3 + Omega_L_ref)
        chi = trapz(1.0 / H_s, zs)
        dL_LCDM.append((1.0 + z) * chi)
    axes[1].plot(z_vals, dL_LCDM, 'k--', label='ΛCDM (0.3/0.7)', linewidth=2, alpha=0.7)
    
    axes[0].set_xlabel('Redshift z', fontsize=12)
    axes[0].set_ylabel('H(z) / H₀', fontsize=12)
    axes[0].set_title('Hubble Parameter vs Redshift', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Redshift z', fontsize=12)
    axes[1].set_ylabel('$d_L$ (c/H₀)', fontsize=12)
    axes[1].set_title('Luminosity Distance vs Redshift', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.png', dpi=300)
    print("Saved: parameter_comparison.png")
    plt.show()
    
    # Detailed analysis for best-fit parameters
    print(f"\n{'=' * 70}")
    print("Detailed analysis for reference parameters (V0=1e-6, lambda=1.0)...")
    print(f"{'=' * 70}")
    
    cosmo_ref = CoherenceCosmology(V0=1.0e-6, lambda_param=1.0)
    cosmo_ref.evolve()
    cosmo_ref.plot_density_evolution(savefig='density_evolution_ref.png')
    cosmo_ref.compare_with_LCDM(z_max=2.0, savefig_prefix='lcdm_comparison')
    
    print(f"\n{'=' * 70}")
    print("Analysis complete! Check generated PNG files.")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()

