"""
Solar system tests: PPN parameters and screening mechanisms.

Verify that coherence field reduces to GR in high-density environments.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp
from scipy.optimize import brentq


class PPNCalculator:
    """
    Calculate Post-Newtonian parameters for coherence scalar field.
    
    For scalar-tensor theories, the key PPN parameters are:
    - γ: curvature parameter (GR: γ = 1)
    - β: nonlinearity parameter (GR: β = 1)
    
    Observational constraints:
    - |γ - 1| < 2.3×10⁻⁵ (Cassini)
    - |β - 1| < 8×10⁻⁵ (lunar laser ranging)
    """
    
    def __init__(self, V0=1e-6, lambda_param=1.0, coupling=1.0):
        """
        Parameters:
        -----------
        V0 : float
            Potential scale
        lambda_param : float
            Exponential slope
        coupling : float
            Matter-field coupling strength
        """
        self.V0 = V0
        self.lambda_param = lambda_param
        self.coupling = coupling
        
        # Physical constants (SI units)
        self.G = 6.674e-11  # m^3 kg^-1 s^-2
        self.c = 2.998e8    # m/s
        self.M_sun = 1.989e30  # kg
        
    def V(self, phi):
        """Potential V(φ)"""
        return self.V0 * np.exp(-self.lambda_param * phi)
    
    def dV_dphi(self, phi):
        """First derivative dV/dφ"""
        return -self.lambda_param * self.V(phi)
    
    def d2V_dphi2(self, phi):
        """Second derivative d²V/dφ²"""
        return self.lambda_param**2 * self.V(phi)
    
    def effective_mass_squared(self, phi, rho_matter=0):
        """
        Effective mass squared including matter coupling.
        
        m_eff² = d²V_eff/dφ²
        
        Parameters:
        -----------
        phi : float
            Field value
        rho_matter : float
            Matter density (kg/m³)
            
        Returns:
        --------
        m_eff_sq : float
            Effective mass squared (in natural units)
        """
        return self.d2V_dphi2(phi) + self.coupling * rho_matter
    
    def compute_ppn_parameters(self, phi_bg=0, rho_bg=0):
        """
        Compute PPN parameters γ and β.
        
        For scalar-tensor theories:
        γ - 1 ≈ -2 α₀² / (1 + α₀²)
        β - 1 ≈ (1/2) α₀² β₀ / (1 + α₀²)²
        
        where α₀ is the coupling strength
        
        Parameters:
        -----------
        phi_bg : float
            Background field value
        rho_bg : float
            Background matter density
            
        Returns:
        --------
        gamma, beta : float
            PPN parameters
        """
        # Effective coupling (simplified)
        alpha_eff = self.coupling / np.sqrt(self.effective_mass_squared(phi_bg, rho_bg))
        
        # PPN parameters
        gamma = 1 - 2 * alpha_eff**2 / (1 + alpha_eff**2)
        beta = 1 + 0.5 * alpha_eff**2 / (1 + alpha_eff**2)**2
        
        return gamma, beta
    
    def solve_field_around_sun(self, r_array):
        """
        Solve for scalar field φ(r) around the Sun.
        
        Equation: ∇²φ = dV_eff/dφ
        In spherical symmetry: (1/r²) d/dr(r² dφ/dr) = dV_eff/dφ
        
        Parameters:
        -----------
        r_array : array
            Radial coordinates (m)
            
        Returns:
        --------
        phi : array
            Field profile φ(r)
        """
        # Sun properties
        M_sun = self.M_sun
        R_sun = 6.96e8  # m
        rho_sun_avg = M_sun / (4/3 * np.pi * R_sun**3)
        
        # Asymptotic field value (in low-density region)
        phi_inf = 0.0
        
        def rho_sun(r):
            """Sun density profile (simplified)"""
            if r < R_sun:
                return rho_sun_avg * (1 - (r/R_sun)**2)
            else:
                return 0.0
        
        def field_equation(r, y):
            """
            ODE system: y = [φ, dφ/dr]
            dy/dr = [dφ/dr, d²φ/dr²]
            
            d²φ/dr² = -2/r dφ/dr + dV_eff/dφ
            """
            phi, dphi_dr = y
            
            d2phi_dr2 = -2 * dphi_dr / r + self.dV_dphi(phi) + self.coupling * rho_sun(r)
            
            return [dphi_dr, d2phi_dr2]
        
        # Initial conditions at large r
        y0 = [phi_inf, 0.0]
        
        # Integrate inward
        solution = odeint(field_equation, y0, r_array[::-1])
        phi = solution[::-1, 0]
        
        return phi
    
    def compute_fifth_force_strength(self, r):
        """
        Compute fifth force relative to Newtonian gravity.
        
        F_5th / F_Newton = α² exp(-m_eff r) (1 + m_eff r) / r²
        
        Parameters:
        -----------
        r : float
            Distance from source (m)
            
        Returns:
        --------
        ratio : float
            Fifth force to Newtonian force ratio
        """
        # Assume screened environment
        phi_bg = 0.0
        rho_bg = 1e-10  # Low density (space)
        
        m_eff_sq = self.effective_mass_squared(phi_bg, rho_bg)
        m_eff = np.sqrt(np.abs(m_eff_sq)) if m_eff_sq > 0 else 0.0
        
        alpha = self.coupling
        
        if m_eff > 0:
            ratio = alpha**2 * np.exp(-m_eff * r) * (1 + m_eff * r)
        else:
            ratio = alpha**2  # Unscreened
        
        return ratio
    
    def plot_solar_system_tests(self, savefig='ppn_solar_system.png'):
        """
        Plot solar system test results.
        
        Parameters:
        -----------
        savefig : str
            Save filename
        """
        # Compute PPN parameters
        gamma, beta = self.compute_ppn_parameters()
        
        print(f"\nPPN Parameters:")
        print(f"  gamma = {gamma:.6f}  (GR: 1.0)")
        print(f"  beta  = {beta:.6f}  (GR: 1.0)")
        print(f"  |gamma - 1| = {abs(gamma - 1):.2e}  (Constraint: < 2.3e-5)")
        print(f"  |beta - 1|  = {abs(beta - 1):.2e}  (Constraint: < 8e-5)")
        
        # Field profile around Sun
        r_array = np.logspace(8, 12, 200)  # 100 km to 10,000 km
        # phi_profile = self.solve_field_around_sun(r_array)
        
        # Fifth force strength
        fifth_force_ratio = np.array([self.compute_fifth_force_strength(r) 
                                     for r in r_array])
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Fifth force ratio
        ax = axes[0]
        ax.loglog(r_array / 1.496e11, fifth_force_ratio, linewidth=2.5)  # Convert to AU
        ax.set_xlabel('Distance (AU)', fontsize=12)
        ax.set_ylabel('$F_{5th} / F_{Newton}$', fontsize=12)
        ax.set_title('Fifth Force Strength', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, which='both')
        ax.axhline(1e-5, color='r', linestyle='--', alpha=0.7, 
                  label='Solar system constraint')
        ax.legend(fontsize=10)
        
        # Panel 2: PPN parameter comparison
        ax = axes[1]
        params_names = ['γ', 'β']
        params_values = [gamma, beta]
        params_GR = [1.0, 1.0]
        params_constraints = [2.3e-5, 8e-5]
        
        x = np.arange(len(params_names))
        width = 0.35
        
        ax.bar(x - width/2, params_GR, width, label='GR', alpha=0.7)
        ax.bar(x + width/2, params_values, width, label='Coherence field', alpha=0.7)
        
        # Error bars for constraints
        for i, constraint in enumerate(params_constraints):
            ax.errorbar(i, 1.0, yerr=constraint, fmt='none', 
                       color='red', capsize=5, capthick=2, 
                       linewidth=2, alpha=0.7)
        
        ax.set_ylabel('Parameter value', fontsize=12)
        ax.set_title('PPN Parameters', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(params_names, fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim([0.99, 1.01])
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(savefig, dpi=300)
        print(f"\nPlot saved: {savefig}")
        plt.show()


def test_solar_system():
    """Test solar system constraints."""
    print("=" * 70)
    print("Solar System PPN Tests")
    print("=" * 70)
    
    # Create calculator with small coupling (for screening)
    ppn = PPNCalculator(V0=1e-6, lambda_param=1.0, coupling=1e-3)
    
    # Compute and plot
    ppn.plot_solar_system_tests(savefig='solar_system_tests.png')
    
    print("\n" + "=" * 70)
    print("Solar system tests complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_solar_system()

