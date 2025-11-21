#!/usr/bin/env python3
"""
Unit Test: Yukawa Kernel Hankel Transform Identity

Verifies that the Hankel transform of the Yukawa kernel recovers the
2D momentum-space propagator:

    ∫₀^∞ R dR J₀(kR) K₀(R/ℓ) = 1 / (k² + ℓ⁻²)

This validates the theoretical derivation in LINEAR_RESPONSE_DERIVATION.md.

Reference: Gradshteyn & Ryzhik, "Table of Integrals", 7th ed., eq. 6.565.3
"""

import numpy as np
from scipy.special import j0, k0  # Bessel J_0 and modified Bessel K_0
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def yukawa_propagator_momentum(k, ell):
    """
    Yukawa propagator in momentum space: G̃(k) = 1 / (k² + ℓ⁻²)
    
    Parameters:
    - k: wavenumber [kpc⁻¹]
    - ell: coherence length [kpc]
    
    Returns:
    - propagator [kpc²]
    """
    return 1.0 / (k**2 + 1.0/ell**2)


def yukawa_kernel_real(R, ell):
    """
    Yukawa kernel in real space: K(R) = (1/2π) K₀(R/ℓ)
    
    Parameters:
    - R: radius [kpc]
    - ell: coherence length [kpc]
    
    Returns:
    - kernel [dimensionless]
    """
    x = R / ell
    if x < 1e-10:
        # Small-argument limit: K₀(x) ≈ -ln(x/2) - γ_E
        # Use regularized form to avoid singularity
        x = 1e-10
    return k0(x) / (2.0 * np.pi)


def hankel_transform_numerical(k, ell, R_max=50.0, n_points=1000):
    """
    Numerically compute Hankel transform of K₀(R/ℓ):
    
    H[K₀](k) = ∫₀^∞ R dR J₀(kR) K₀(R/ℓ)
    
    Gradshteyn & Ryzhik identity (6.565.3):
    ∫₀^∞ x J₀(ax) / (x² + b²) dx = K₀(ab)
    
    Substituting x = kR, we get:
    ∫₀^∞ R J₀(kR) K₀(R/ℓ) dR = 1 / (k² + ℓ⁻²)
    
    Parameters:
    - k: wavenumber [kpc⁻¹]
    - ell: coherence length [kpc]
    - R_max: integration upper limit [kpc]
    - n_points: number of integration points
    
    Returns:
    - Hankel transform [kpc²]
    """
    def integrand(R):
        if R < 1e-10:
            return 0.0
        x = R / ell
        if x < 1e-10:
            x = 1e-10
        # Direct K₀ without the 1/(2π) factor
        return R * j0(k * R) * k0(x)
    
    result, error = quad(integrand, 0, R_max, limit=100)
    return result


def test_hankel_identity(ell=1.0, k_range=None, plot=True):
    """
    Test Hankel transform identity for various k values.
    
    Parameters:
    - ell: coherence length [kpc]
    - k_range: array of wavenumbers [kpc⁻¹] (default: log-spaced)
    - plot: whether to generate comparison plot
    
    Returns:
    - max_rel_error: maximum relative error across k_range
    """
    if k_range is None:
        k_range = np.logspace(-1, 1, 20)  # 0.1 to 10 kpc⁻¹
    
    # Compute both sides of the identity
    analytic = np.array([yukawa_propagator_momentum(k, ell) for k in k_range])
    numerical = np.array([hankel_transform_numerical(k, ell) for k in k_range])
    
    # Relative error
    rel_error = np.abs((numerical - analytic) / analytic)
    max_rel_error = np.max(rel_error)
    
    print("="*80)
    print("Hankel Transform Identity Test")
    print("="*80)
    print(f"Coherence length: ℓ = {ell:.2f} kpc")
    print(f"Number of k points: {len(k_range)}")
    print(f"k range: {k_range[0]:.2e} to {k_range[-1]:.2e} kpc⁻¹")
    print(f"\nMax relative error: {max_rel_error:.4e}")
    print(f"Mean relative error: {np.mean(rel_error):.4e}")
    print("="*80)
    
    # Detailed breakdown
    print("\nDetailed Results:")
    print(f"{'k [kpc⁻¹]':<15} {'Analytic':<15} {'Numerical':<15} {'Rel Error':<15}")
    print("-"*60)
    for i in range(len(k_range)):
        print(f"{k_range[i]:<15.4e} {analytic[i]:<15.6f} {numerical[i]:<15.6f} {rel_error[i]:<15.4e}")
    
    # Acceptance criterion
    success = max_rel_error < 0.01  # 1% error threshold
    if success:
        print(f"\n✓ TEST PASSED: Max error {max_rel_error:.4e} < 1%")
    else:
        print(f"\n✗ TEST FAILED: Max error {max_rel_error:.4e} >= 1%")
    
    # Plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top: Propagator comparison
        ax1.loglog(k_range, analytic, 'b-', lw=2, label='Analytic: 1/(k² + ℓ⁻²)')
        ax1.loglog(k_range, numerical, 'ro', ms=6, label='Numerical: Hankel[K₀(R/ℓ)]')
        ax1.set_xlabel('k [kpc⁻¹]')
        ax1.set_ylabel('G̃(k) [kpc²]')
        ax1.set_title(f'Yukawa Propagator (ℓ = {ell:.2f} kpc)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add k = ℓ⁻¹ marker
        ax1.axvline(1/ell, color='gray', linestyle='--', alpha=0.5, label=f'k = ℓ⁻¹ = {1/ell:.2f}')
        
        # Bottom: Relative error
        ax2.loglog(k_range, rel_error, 'g-', lw=2)
        ax2.axhline(0.01, color='r', linestyle='--', label='1% threshold')
        ax2.set_xlabel('k [kpc⁻¹]')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Numerical Integration Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'gpm_tests')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'hankel_transform_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {plot_path}")
        plt.close()
    
    return max_rel_error


def test_real_space_kernel(ell=1.0, R_range=None, plot=True):
    """
    Visualize the Yukawa kernel in real space: K(R) = (1/2π) K₀(R/ℓ)
    
    Checks asymptotic behavior:
    - Small R: K₀(x) ~ -ln(x) (logarithmic divergence, regularized)
    - Large R: K₀(x) ~ √(π/2x) exp(-x) (exponential decay)
    """
    if R_range is None:
        R_range = np.logspace(-2, 1.5, 100) * ell  # 0.01ℓ to 30ℓ
    
    K_R = np.array([yukawa_kernel_real(R, ell) for R in R_range])
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Real-space kernel
        ax1.loglog(R_range/ell, K_R * (2*np.pi), 'b-', lw=2, label='K₀(R/ℓ)')
        ax1.set_xlabel('R / ℓ')
        ax1.set_ylabel('K₀(R/ℓ)')
        ax1.set_title('Yukawa Kernel (Real Space)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add asymptotic limits
        x_small = R_range[R_range < ell] / ell
        x_large = R_range[R_range > ell] / ell
        
        if len(x_small) > 0:
            # Small-R: -ln(x/2) - γ_E (Euler's constant)
            gamma_E = 0.5772
            asymp_small = -np.log(x_small/2) - gamma_E
            ax1.loglog(x_small, asymp_small, 'r--', alpha=0.5, label='~-ln(R/2ℓ)')
        
        if len(x_large) > 0:
            # Large-R: √(π/2x) exp(-x)
            asymp_large = np.sqrt(np.pi / (2*x_large)) * np.exp(-x_large)
            ax1.loglog(x_large, asymp_large, 'g--', alpha=0.5, label='~√(π/2x)e⁻ˣ')
        
        ax1.legend()
        
        # Right: Screened vs unscreened
        R_lin = np.linspace(0.1, 10, 100) * ell
        K_yukawa = np.array([yukawa_kernel_real(R, ell) for R in R_lin])
        K_newton = 1.0 / (2 * np.pi * R_lin)  # Newtonian 1/R
        
        ax2.plot(R_lin/ell, K_yukawa * (2*np.pi) * ell, 'b-', lw=2, label='GPM (Yukawa)')
        ax2.plot(R_lin/ell, K_newton * (2*np.pi) * ell, 'r--', lw=2, label='Newtonian (1/R)')
        ax2.set_xlabel('R / ℓ')
        ax2.set_ylabel('Kernel × R [normalized]')
        ax2.set_title('Screening Effect')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 10])
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'gpm_tests')
        plot_path = os.path.join(output_dir, 'yukawa_kernel_real_space.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Real-space kernel plot saved to {plot_path}")
        plt.close()


def test_multiple_ell_values():
    """
    Test Hankel identity for multiple coherence lengths.
    """
    ell_values = [0.5, 1.0, 2.0, 5.0]  # kpc
    
    print("\n" + "="*80)
    print("Testing Multiple Coherence Lengths")
    print("="*80)
    
    results = {}
    for ell in ell_values:
        print(f"\nℓ = {ell:.1f} kpc:")
        max_error = test_hankel_identity(ell=ell, plot=False)
        results[ell] = max_error
    
    print("\n" + "="*80)
    print("Summary:")
    print("-"*80)
    for ell, err in results.items():
        status = "✓ PASS" if err < 0.01 else "✗ FAIL"
        print(f"ℓ = {ell:.1f} kpc: max error = {err:.4e}  {status}")
    print("="*80)
    
    return all(err < 0.01 for err in results.values())


def main():
    """
    Run all Yukawa kernel tests.
    """
    print("="*80)
    print("YUKAWA KERNEL HANKEL TRANSFORM VERIFICATION")
    print("="*80)
    print("\nVerifying theoretical identity:")
    print("  ∫₀^∞ R dR J₀(kR) K₀(R/ℓ) = 1 / (k² + ℓ⁻²)")
    print("\nReference: Gradshteyn & Ryzhik (2007), eq. 6.565.3")
    print("="*80 + "\n")
    
    # Test 1: Single ℓ value with plots
    print("Test 1: Hankel transform identity (ℓ = 1.0 kpc)")
    print("-"*80)
    test_hankel_identity(ell=1.0, plot=True)
    
    # Test 2: Real-space kernel visualization
    print("\n\nTest 2: Real-space kernel visualization")
    print("-"*80)
    test_real_space_kernel(ell=1.0, plot=True)
    
    # Test 3: Multiple ℓ values
    print("\n\nTest 3: Multiple coherence lengths")
    print("-"*80)
    all_passed = test_multiple_ell_values()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nConclusion:")
        print("  The Hankel transform identity is verified numerically.")
        print("  The Yukawa kernel K₀(R/ℓ) is the correct 2D Green's function")
        print("  for the dressed gravitational propagator G̃(k) = 1/(k² + ℓ⁻²).")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check numerical integration parameters or kernel implementation.")
    print("="*80)


if __name__ == '__main__':
    main()
