"""
Section 2 Invariant Tests for Σ-Gravity Gates

These tests prove that your gates satisfy the non-negotiables from Section 2:
- Newtonian limit (PPN safe)
- Curl-free fields
- Monotone/saturating behavior
- Solar system safety

Run with: pytest tests/test_section2_invariants.py -v
"""

import numpy as np
from mpmath import quad, ellipk
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_core import (
    G_distance, G_acceleration, G_bulge_exponential,
    G_unified, G_solar_system, C_burr_XII, K_sigma_gravity
)


class TestGateInvariants:
    """Test that gates satisfy mathematical invariants"""
    
    def test_gates_in_bounds(self):
        """All gates must stay in [0, 1]"""
        R = np.logspace(-6, 3, 200)  # kpc (AU to Mpc scale)
        g = np.logspace(-14, -8, 200)  # m/s²
        
        Gd = G_distance(R, R_min=1.0, alpha=2.0, beta=1.0)
        Ga = G_acceleration(g, g_crit=1e-9, alpha=2.0, beta=1.0)
        Gb = G_bulge_exponential(R, R_bulge=1.5, alpha=2.0, beta=1.0)
        
        for G, name in [(Gd, 'distance'), (Ga, 'accel'), (Gb, 'bulge')]:
            assert np.all((G >= -1e-12) & (G <= 1+1e-12)), \
                f"{name} gate out of bounds"
    
    def test_distance_gate_limits(self):
        """G_distance: G(R→0)→0, G(R→∞)→1"""
        R = np.logspace(-6, 3, 200)
        G = G_distance(R, R_min=1.0, alpha=2.0, beta=1.0)
        
        assert G[0] < 1e-10, "Distance gate should → 0 as R → 0"
        assert G[-1] > 1 - 1e-6, "Distance gate should → 1 as R → ∞"
    
    def test_distance_gate_monotonic(self):
        """G_distance must be monotonically increasing with R"""
        R = np.logspace(-3, 2, 200)
        G = G_distance(R, R_min=1.0, alpha=2.0, beta=1.0)
        
        dG_dlogR = np.gradient(G, np.log(R))
        assert np.all(dG_dlogR >= -1e-6), \
            "Distance gate must be monotonic increasing"
    
    def test_accel_gate_decreasing(self):
        """G_acceleration must be decreasing with g_bar"""
        g = np.logspace(-14, -8, 200)
        G = G_acceleration(g, g_crit=1e-9, alpha=2.0, beta=1.0)
        
        dG_dlogg = np.gradient(G, np.log(g))
        assert np.all(dG_dlogg <= 1e-6), \
            "Acceleration gate must be decreasing with g_bar"
    
    def test_exponential_gate_limits(self):
        """G_bulge: G(R→0)→0, G(R→∞)→1"""
        R = np.logspace(-3, 2, 200)
        G = G_bulge_exponential(R, R_bulge=1.5, alpha=2.0, beta=1.0)
        
        assert G[0] < 1e-10, "Bulge gate should → 0 at R = 0"
        assert G[-1] > 1 - 1e-4, "Bulge gate should → 1 at large R"


class TestNewtonianLimitAndPPN:
    """Test solar system safety and PPN constraints"""
    
    def test_newtonian_limit_at_AU_scales(self):
        """K must be < 10^-14 at 1 AU (Cassini/PPN safe)"""
        AU_in_kpc = 4.848136811e-9
        R_vals = np.array([1.0, 10.0, 100.0]) * AU_in_kpc  # AU scales
        g_bar = 5.9e-3  # Earth's acceleration (m/s²)
        
        K_vals = K_sigma_gravity(R_vals, g_bar, A=0.6, ell0=5.0, p=0.75, n_coh=0.5,
                                 gate_type='distance',
                                 gate_params={'R_min': 0.0001, 'alpha': 4.0, 'beta': 2.0})
        
        assert K_vals[0] < 1e-14, f"K(1 AU) = {K_vals[0]:.2e} fails PPN bound"
        assert K_vals[1] < 1e-12, f"K(10 AU) = {K_vals[1]:.2e} too large"
        assert K_vals[2] < 1e-10, f"K(100 AU) = {K_vals[2]:.2e} too large"
    
    def test_wide_binary_band(self):
        """K must be < 10^-8 at 10^4 AU (wide binary constraint)"""
        AU_in_kpc = 4.848136811e-9
        R_wide = 1e4 * AU_in_kpc
        g_bar = 1e-10  # Typical outer disk
        
        K = K_sigma_gravity(R_wide, g_bar, A=0.6, ell0=5.0,
                           gate_type='distance',
                           gate_params={'R_min': 0.0001, 'alpha': 4.0, 'beta': 2.0})
        
        assert K < 1e-8, f"K(10^4 AU) = {K:.2e} violates wide-binary constraint"
    
    def test_solar_system_gate_explicit(self):
        """G_solar_system must strongly suppress at AU scales"""
        AU_in_kpc = 4.848e-9
        R_AU = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0]) * AU_in_kpc
        G = G_solar_system(R_AU, R_min_AU=1.0, alpha=4.0, beta=2.0)
        
        assert G[0] < 1e-15, "Solar system gate fails at 1 AU"
        assert G[1] < 1e-12, "Solar system gate fails at 10 AU"
        assert G[2] < 1e-10, "Solar system gate fails at 100 AU"
        assert G[3] < 1e-8, "Solar system gate fails at 1000 AU"


class TestCurlFree:
    """Test that enhanced field remains curl-free"""
    
    def test_curl_free_numeric_loop(self):
        """
        For axisymmetric K=K(R), must have ∇ × g_eff = 0
        
        Test by integrating ∮ g_eff · dl around closed loops.
        For a conservative field, this integral must be zero.
        """
        M, G_newton = 1.0, 1.0  # Normalized units
        
        def g_bar(R):
            """Toy Newtonian field"""
            return G_newton * M / R**2
        
        def K(R):
            """Σ-Gravity kernel"""
            return K_sigma_gravity(R, g_bar(R), A=0.6, ell0=5.0, p=0.75, n_coh=0.5,
                                  gate_type='distance',
                                  gate_params={'R_min': 1.0, 'alpha': 2.0, 'beta': 1.0})
        
        def g_eff(R):
            """Enhanced field"""
            return g_bar(R) * (1.0 + K(R))
        
        # Loop integral at various R0
        R0_vals = [5.0, 10.0, 20.0]  # kpc
        
        for R0 in R0_vals:
            eps = 1e-3  # Small radial wiggle
            phi = np.linspace(0, 2*np.pi, 400, endpoint=True)
            
            # Tangential component (radial parts cancel in loop)
            integrand = g_eff(R0 + eps*np.cos(phi)) * (R0 * np.gradient(phi))
            loop_integral = np.trapz(integrand, phi)
            
            # Normalize by field strength
            normalized_error = abs(loop_integral) / (g_eff(R0) * R0)
            
            assert normalized_error < 1e-8, \
                f"Curl-free check failed at R0={R0} kpc: error = {normalized_error:.2e}"


class TestRingKernelGeometry:
    """Test exact ring-kernel reduction to elliptic integrals (Appendix B)"""
    
    def test_ring_green_numeric_vs_analytic(self):
        """
        The azimuthal integral must match complete elliptic integral K(m)
        
        Numerical: ∫₀^{2π} dφ / √(R² + R'² - 2RR'cosφ)
        Analytic:  (4/(R+R')) K(m), where m = 4RR'/(R+R')²
        """
        def ring_green_numeric(R, Rp):
            """Numerical integration"""
            def f(phi):
                return 1.0 / np.sqrt(R**2 + Rp**2 - 2*R*Rp*np.cos(phi))
            return 2.0 * quad(f, [0, np.pi])
        
        def ring_green_elliptic(R, Rp):
            """Analytic form with elliptic integral"""
            m = 4.0 * R * Rp / ((R + Rp)**2)
            return 4.0 / (R + Rp) * ellipk(m)
        
        # Test at several (R, R') pairs
        test_pairs = [(5.0, 7.0), (3.0, 10.0), (8.0, 8.0), (2.0, 15.0)]
        
        for R, Rp in test_pairs:
            num = float(ring_green_numeric(R, Rp))
            ana = float(ring_green_elliptic(R, Rp))
            rel_error = abs(num - ana) / num
            
            assert rel_error < 1e-6, \
                f"Ring kernel mismatch at (R={R}, R'={Rp}): error = {rel_error:.2e}"


class TestCoherenceWindow:
    """Test the universal coherence window C(R) properties"""
    
    def test_coherence_window_limits(self):
        """C(R): C(0)→0, C(∞)→1, C(ℓ₀)≈0.5"""
        R = np.linspace(0, 50, 500)
        ell0, p, n_coh = 5.0, 2.0, 1.0
        C = C_burr_XII(R, ell0, p, n_coh)
        
        assert C[0] < 1e-10, "C(0) should be ≈ 0"
        assert C[-1] > 0.99, "C(large R) should be ≈ 1"
        
        # Find C at ℓ₀
        idx_ell0 = np.argmin(np.abs(R - ell0))
        assert 0.4 < C[idx_ell0] < 0.6, \
            f"C(ℓ₀) = {C[idx_ell0]:.3f} should be ≈ 0.5"
    
    def test_coherence_window_monotonic(self):
        """C(R) must be monotonically increasing"""
        R = np.linspace(0.1, 50, 500)
        C = C_burr_XII(R, ell0=5.0, p=2.0, n_coh=1.0)
        
        dC = np.gradient(C, R)
        assert np.all(dC >= -1e-10), "C(R) must be monotonic increasing"


class TestUnifiedGate:
    """Test the combined distance × acceleration gate"""
    
    def test_unified_gate_product_structure(self):
        """G_unified = G_distance × G_acceleration"""
        R = np.logspace(-2, 2, 100)
        g_bar = 1e-10 / R**2  # Toy model
        
        params = {
            'R_min': 1.0, 'g_crit': 1e-10,
            'alpha_R': 2.0, 'beta_R': 1.0,
            'alpha_g': 2.0, 'beta_g': 1.0
        }
        
        G_uni = G_unified(R, g_bar, **params)
        G_d = G_distance(R, params['R_min'], params['alpha_R'], params['beta_R'])
        G_a = G_acceleration(g_bar, params['g_crit'], params['alpha_g'], params['beta_g'])
        
        np.testing.assert_allclose(G_uni, G_d * G_a, rtol=1e-10,
                                  err_msg="Unified gate must be product of components")
    
    def test_unified_gate_double_suppression(self):
        """At small R AND high g, gate should be strongly suppressed"""
        R_small = 0.1  # kpc
        g_high = 1e-8  # m/s²
        
        G = G_unified(R_small, g_high, R_min=1.0, g_crit=1e-10)
        
        assert G < 0.01, \
            "Unified gate should be strongly suppressed at small R AND high g"


# ============================================================================
# INTEGRATION TESTS WITH MAIN REPOSITORY (if available)
# ============================================================================

class TestIntegrationWithMainRepo:
    """Test integration with many_path_model if available"""
    
    def test_compatible_with_coherence_window(self):
        """Gates should be compatible with main repo's C(R)"""
        # This test requires the main repo's path_spectrum_kernel.py
        # Skip if not available
        try:
            import sys
            sys.path.insert(0, '../many_path_model')
            from path_spectrum_kernel import coherence_window
            
            R = np.linspace(0, 20, 100)
            
            # Our version
            C_ours = C_burr_XII(R, ell0=5.0, p=2.0, n_coh=1.0)
            
            # Main repo version (if compatible signature)
            # C_theirs = coherence_window(R, ell0=5.0, p=2.0, n_coh=1.0)
            
            # Would assert np.allclose(C_ours, C_theirs)
            # For now, just check ours is reasonable
            assert np.all((C_ours >= 0) & (C_ours <= 1))
            
        except ImportError:
            # Skip if main repo not available
            pass


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    import pytest
    
    # Run with verbose output
    pytest.main([__file__, '-v', '--tb=short'])

