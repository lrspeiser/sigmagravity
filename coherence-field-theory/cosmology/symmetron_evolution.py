"""
Symmetron potential for coherence field theory.

Potential: V(φ) = -μ²φ²/2 + λφ⁴/4 + V₀
Coupling: A(φ) = exp(βφ²/2)  (or 1 + φ²/(2M²) depending on regime)

Key feature: Two minima that switch based on density
- High density (ρ > ρ_crit): φ = 0 (screened)
- Low density (ρ < ρ_crit): φ = ±φ₀ (active)

Critical density: ρ_crit ~ μ²M²
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class SymmetronCosmology:
    """
    Evolve cosmology with symmetron scalar field.
    
    Potential: V(φ) = -μ²φ²/2 + λφ⁴/4 + V₀
    Effective: V_eff(φ,ρ) = V(φ) + ρφ²/(2M²)
    
    Parameters
    ----------
    mu2 : float
        Mass-squared parameter (controls symmetry breaking)
    lambda_s : float
        Quartic coupling (self-interaction strength)
    M : float
        Coupling scale (M² controls matter coupling strength)
    V0 : float
        Constant energy offset
    rho_m0_guess : float
        Initial guess for present-day matter density
    """
    
    def __init__(self, mu2=1.0e-6, lambda_s=1.0, M=1.0, V0=1.0e-6, 
                 rho_m0_guess=1.0e-6):
        self.mu2 = mu2
        self.lambda_s = lambda_s
        self.M = M
        self.V0 = V0
        self.rho_m0_guess = rho_m0_guess
        
        # Derived quantities
        self.phi_0_vacuum = np.sqrt(mu2 / lambda_s) if lambda_s > 0 else 0
        self.rho_crit = mu2 * M**2
        
    def V(self, phi):
        """
        Bare potential V(φ) = -μ²φ²/2 + λφ⁴/4 + V₀
        
        Two minima at φ = ±√(μ²/λ) when isolated (ρ=0)
        Single minimum at φ = 0 when coupled to high density
        """
        return -0.5 * self.mu2 * phi**2 + 0.25 * self.lambda_s * phi**4 + self.V0
    
    def dV_dphi(self, phi):
        """Potential derivative dV/dφ = -μ²φ + λφ³"""
        return -self.mu2 * phi + self.lambda_s * phi**3
    
    def V_eff(self, phi, rho):
        """
        Effective potential including matter coupling:
        V_eff(φ,ρ) = V(φ) + ρφ²/(2M²)
        
        High density: matter term dominates → φ = 0
        Low density: bare potential dominates → φ = ±φ₀
        """
        return self.V(phi) + rho * phi**2 / (2 * self.M**2)
    
    def dV_eff_dphi(self, phi, rho):
        """Effective potential derivative"""
        return self.dV_dphi(phi) + rho * phi / self.M**2
    
    def phi_min(self, rho):
        """
        Field value at minimum of V_eff for given density.
        
        High density (ρ > ρ_crit): φ_min ≈ 0 (screened)
        Low density (ρ < ρ_crit): φ_min ≈ ±√((μ² - ρ/M²)/λ) (active)
        
        Returns
        -------
        phi_min : float
            Field value at minimum (picks positive branch)
        """
        # Solve: dV_eff/dφ = 0
        # -μ²φ + λφ³ + ρφ/M² = 0
        # φ(-μ² + λφ² + ρ/M²) = 0
        
        # Trivial solution
        if rho >= self.rho_crit:
            return 0.0
        
        # Non-trivial solution (positive branch)
        mu2_eff = self.mu2 - rho / self.M**2
        if mu2_eff <= 0:
            return 0.0
        
        phi_min_sq = mu2_eff / self.lambda_s
        return np.sqrt(phi_min_sq) if phi_min_sq > 0 else 0.0
    
    def m_eff(self, phi, rho):
        """
        Effective mass at field value φ in density ρ.
        
        m_eff² = d²V_eff/dφ² = -μ² + 3λφ² + ρ/M²
        """
        d2V = -self.mu2 + 3 * self.lambda_s * phi**2 + rho / self.M**2
        return np.sqrt(abs(d2V)) if d2V > 0 else 0.0
    
    def evolve(self, N_start=-7.0, N_end=0.0, n_steps=4000):
        """
        Evolve cosmology from early times to present.
        
        Parameters
        ----------
        N_start : float
            Starting e-fold number (ln a), typically -7 or less
        N_end : float
            Ending e-fold number, 0 corresponds to a=1 (today)
        n_steps : int
            Number of integration steps
            
        Returns
        -------
        results : dict
            Dictionary containing arrays: N, a, phi, phidot, H, rho_m, rho_phi
        """
        dN = (N_end - N_start) / n_steps
        N_vals = np.linspace(N_start, N_end, n_steps + 1)
        a_vals = np.exp(N_vals)
        
        # Initialize arrays
        phi_vals = np.zeros_like(N_vals)
        phidot_vals = np.zeros_like(N_vals)
        rho_m_vals = np.zeros_like(N_vals)
        rho_phi_vals = np.zeros_like(N_vals)
        H_vals = np.zeros_like(N_vals)
        
        # Initial conditions: start near vacuum minimum (high z)
        # At early times, density is high → φ ≈ 0
        phi_vals[0] = 0.0
        phidot_vals[0] = 0.0
        
        # Evolve using RK4 in N = ln(a)
        for i in range(len(N_vals)):
            N = N_vals[i]
            
            if i == 0:
                rho_m = self.rho_m0_guess * np.exp(-3 * N)
                rho_phi = 0.5 * phidot_vals[i]**2 + self.V(phi_vals[i])
                H = np.sqrt(rho_m + rho_phi)
            else:
                phi = phi_vals[i-1]
                phidot = phidot_vals[i-1]
                N_prev = N_vals[i-1]
                
                # RK4 integration
                k1_phi, k1_phidot = self._rhs(phi, phidot, N_prev)
                
                phi_mid = phi + 0.5 * dN * k1_phi
                phidot_mid = phidot + 0.5 * dN * k1_phidot
                N_mid = N_prev + 0.5 * dN
                k2_phi, k2_phidot = self._rhs(phi_mid, phidot_mid, N_mid)
                
                phi_mid2 = phi + 0.5 * dN * k2_phi
                phidot_mid2 = phidot + 0.5 * dN * k2_phidot
                k3_phi, k3_phidot = self._rhs(phi_mid2, phidot_mid2, N_mid)
                
                phi_end = phi + dN * k3_phi
                phidot_end = phidot + dN * k3_phidot
                N_end_step = N_prev + dN
                k4_phi, k4_phidot = self._rhs(phi_end, phidot_end, N_end_step)
                
                phi_vals[i] = phi + (dN/6.0) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
                phidot_vals[i] = phidot + (dN/6.0) * (k1_phidot + 2*k2_phidot + 2*k3_phidot + k4_phidot)
                
                rho_m = self.rho_m0_guess * np.exp(-3 * N)
                rho_phi = 0.5 * phidot_vals[i]**2 + self.V(phi_vals[i])
                H = np.sqrt(rho_m + rho_phi)
            
            rho_m_vals[i] = rho_m
            rho_phi_vals[i] = rho_phi
            H_vals[i] = H
        
        # Normalize so H(a=1) = 1
        H0 = H_vals[-1]
        H_vals /= H0
        rho_m_vals /= H0**2
        rho_phi_vals /= H0**2
        
        # Compute density parameters
        Omega_m0 = rho_m_vals[-1] / (rho_m_vals[-1] + rho_phi_vals[-1])
        Omega_phi0 = rho_phi_vals[-1] / (rho_m_vals[-1] + rho_phi_vals[-1])
        
        self.results = {
            'N': N_vals,
            'a': a_vals,
            'phi': phi_vals,
            'phidot': phidot_vals,
            'H': H_vals,
            'rho_m': rho_m_vals,
            'rho_phi': rho_phi_vals,
            'Omega_m0': Omega_m0,
            'Omega_phi0': Omega_phi0
        }
        
        return self.results
    
    def _rhs(self, phi, phidot, N):
        """Right-hand side of evolution equations in N = ln(a)"""
        rho_m = self.rho_m0_guess * np.exp(-3 * N)
        rho_phi = 0.5 * phidot**2 + self.V(phi)
        H = np.sqrt(rho_m + rho_phi)
        
        # Include matter coupling in effective potential
        dphi_dN = phidot / H
        dphidot_dN = -3.0 * phidot - self.dV_eff_dphi(phi, rho_m) / H
        
        return dphi_dN, dphidot_dN
    
    def get_phi_0(self):
        """Get today's scalar field value φ₀"""
        if 'phi' not in self.results:
            raise ValueError("Must run evolve() first")
        return self.results['phi'][-1]
    
    def plot_evolution(self, save_path=None):
        """Plot cosmological evolution"""
        if 'phi' not in self.results:
            raise ValueError("Must run evolve() first")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Field evolution
        ax = axes[0, 0]
        ax.plot(self.results['a'], self.results['phi'])
        ax.set_xlabel('Scale factor a')
        ax.set_ylabel('φ')
        ax.set_title('Field Evolution')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Hubble parameter
        ax = axes[0, 1]
        z = 1/self.results['a'] - 1
        ax.plot(z, self.results['H'])
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('H(z) / H₀')
        ax.set_title('Hubble Evolution')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Plot 3: Density parameters
        ax = axes[1, 0]
        Omega_m = self.results['rho_m'] / (self.results['rho_m'] + self.results['rho_phi'])
        Omega_phi = self.results['rho_phi'] / (self.results['rho_m'] + self.results['rho_phi'])
        ax.plot(self.results['a'], Omega_m, label='Ωₘ')
        ax.plot(self.results['a'], Omega_phi, label='Ω_φ')
        ax.set_xlabel('Scale factor a')
        ax.set_ylabel('Density parameter')
        ax.set_title(f'Ωₘ₀ = {self.results["Omega_m0"]:.3f}, Ω_φ₀ = {self.results["Omega_phi0"]:.3f}')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Effective potential shape
        ax = axes[1, 1]
        phi_range = np.linspace(-2*self.phi_0_vacuum, 2*self.phi_0_vacuum, 200)
        
        # Show V_eff at different densities
        rho_today = self.results['rho_m'][-1] * self.rho_m0_guess
        rho_early = self.results['rho_m'][0] * self.rho_m0_guess
        
        V_bare = np.array([self.V(p) for p in phi_range])
        V_eff_today = np.array([self.V_eff(p, rho_today) for p in phi_range])
        V_eff_early = np.array([self.V_eff(p, rho_early) for p in phi_range])
        
        # Normalize to make visible
        V_offset = np.min(V_bare)
        ax.plot(phi_range, V_bare - V_offset, 'k--', label='V(φ) bare', alpha=0.5)
        ax.plot(phi_range, V_eff_today - V_offset, 'b-', label='V_eff(φ) today')
        ax.plot(phi_range, V_eff_early - V_offset, 'r-', label='V_eff(φ) early', alpha=0.5)
        
        ax.axvline(self.results['phi'][-1], color='blue', linestyle=':', alpha=0.5, label='φ₀')
        ax.set_xlabel('φ')
        ax.set_ylabel('V - V_min')
        ax.set_title('Effective Potential')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == '__main__':
    # Test symmetron cosmology
    print("="*70)
    print("SYMMETRON COSMOLOGY TEST")
    print("="*70)
    
    # Try a few parameter sets
    test_params = [
        {'mu2': 1.0e-6, 'lambda_s': 1.0, 'M': 1.0, 'V0': 1.0e-6, 'name': 'Default'},
        {'mu2': 5.0e-7, 'lambda_s': 0.5, 'M': 0.8, 'V0': 5.0e-7, 'name': 'Soft'},
    ]
    
    for params in test_params:
        name = params.pop('name')
        print(f"\n{name} parameters:")
        for k, v in params.items():
            print(f"  {k} = {v}")
        
        cosmo = SymmetronCosmology(**params)
        
        print(f"\nDerived:")
        print(f"  φ₀ (vacuum) = {cosmo.phi_0_vacuum:.3e}")
        print(f"  ρ_crit = {cosmo.rho_crit:.3e}")
        
        # Evolve
        results = cosmo.evolve(N_start=-7.0, N_end=0.0, n_steps=2000)
        
        print(f"\nCosmology:")
        print(f"  Ωₘ₀ = {results['Omega_m0']:.4f}")
        print(f"  Ω_φ₀ = {results['Omega_phi0']:.4f}")
        print(f"  φ₀ (today) = {results['phi'][-1]:.3e}")
        
        # Plot
        cosmo.plot_evolution(f'outputs/symmetron_test_{name.lower()}.png')
    
    print("\n" + "="*70)
    print("Test complete. Plots saved to outputs/")
