"""
Cosmological background evolution for coherence scalar field.

Solves the coupled Friedmann + Klein-Gordon equations to compute:
- Scale factor a(t) evolution
- Hubble parameter H(z)
- Luminosity distance d_L(z)
- Density parameters Ω_m(a), Ω_φ(a)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz


class CoherenceCosmology:
    """
    Evolve cosmology with coherence scalar field.
    
    Potential: V(φ) = V0 * exp(-λφ)
    Units: 8πG/3 = 1, H0 = 1 at present
    """
    
    def __init__(self, V0=1.0e-6, lambda_param=1.0, rho_m0_guess=1.0e-6, M4=None):
        """
        Parameters:
        -----------
        V0 : float
            Potential energy scale
        lambda_param : float
            Exponential slope parameter
        rho_m0_guess : float
            Initial guess for present-day matter density
        M4 : float, optional
            Chameleon mass scale (M^5 term, if None, pure exponential)
            For n=1 chameleon: V(φ) = V₀e^(-λφ) + M^5/φ
        """
        self.V0 = V0
        self.lambda_param = lambda_param
        self.rho_m0_guess = rho_m0_guess
        self.M4 = M4
        
    def V(self, phi):
        """
        Potential V(φ).
        
        For chameleon (n=1): V(φ) = V₀e^(-λφ) + M^5/φ
        Without chameleon: V(φ) = V₀e^(-λφ)
        """
        if self.M4 is not None:
            phi_safe = np.maximum(np.abs(phi), 1e-10) * np.sign(phi) if phi != 0 else 1e-10
            return self.V0 * np.exp(-self.lambda_param * phi) + self.M4**5 / phi_safe
        return self.V0 * np.exp(-self.lambda_param * phi)
    
    def dV_dphi(self, phi):
        """
        Potential derivative dV/dφ.
        
        For chameleon: dV/dφ = -λV₀e^(-λφ) - M^5/φ²
        """
        if self.M4 is not None:
            phi_safe = np.maximum(np.abs(phi), 1e-10) * np.sign(phi) if phi != 0 else 1e-10
            return (-self.lambda_param * self.V0 * np.exp(-self.lambda_param * phi) - 
                    self.M4**5 / phi_safe**2)
        return -self.lambda_param * self.V(phi)
    
    def evolve(self, N_start=-7.0, N_end=0.0, n_steps=4000):
        """
        Evolve cosmology from early times to present.
        
        Parameters:
        -----------
        N_start : float
            Starting e-fold number (ln a), typically -7 or less
        N_end : float
            Ending e-fold number, 0 corresponds to a=1 (today)
        n_steps : int
            Number of integration steps
            
        Returns:
        --------
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
        
        # Initial conditions: scalar nearly frozen at early times
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
    
    def get_phi_0(self):
        """
        Get today's scalar field value φ₀.
        
        Returns:
        --------
        phi_0 : float
            Scalar field value at a=1 (today)
        """
        if 'phi' not in self.results:
            raise ValueError("Must run evolve() first")
        # Today corresponds to a=1, which is the last element
        return self.results['phi'][-1]
    
    def _rhs(self, phi, phidot, N):
        """Right-hand side of evolution equations in N = ln(a)"""
        rho_m = self.rho_m0_guess * np.exp(-3 * N)
        rho_phi = 0.5 * phidot**2 + self.V(phi)
        H = np.sqrt(rho_m + rho_phi)
        
        dphi_dN = phidot / H
        dphidot_dN = -3.0 * phidot - self.dV_dphi(phi) / H
        
        return dphi_dN, dphidot_dN
    
    def compute_H_of_z(self, z_array):
        """
        Compute H(z) at given redshifts.
        
        Parameters:
        -----------
        z_array : array
            Redshift values
            
        Returns:
        --------
        H_z : array
            Hubble parameter in units of H0
        """
        a_sample = 1.0 / (1.0 + z_array)
        H_z = np.interp(a_sample, self.results['a'], self.results['H'])
        return H_z
    
    def compute_dL(self, z_array):
        """
        Compute luminosity distance at given redshifts.
        
        Parameters:
        -----------
        z_array : array
            Redshift values
            
        Returns:
        --------
        dL : array
            Luminosity distance in units of c/H0
        """
        H_interp = interp1d(self.results['a'], self.results['H'], kind='cubic')
        
        dL_vals = []
        for z in z_array:
            zs = np.linspace(0.0, z, 800)
            a_s = 1.0 / (1.0 + zs)
            H_s = H_interp(a_s)
            chi = trapz(1.0 / H_s, zs)  # comoving distance
            dL_vals.append((1.0 + z) * chi)
        
        return np.array(dL_vals)
    
    def plot_density_evolution(self, savefig=None):
        """Plot matter vs coherence field density evolution."""
        a_vals = self.results['a']
        rho_m = self.results['rho_m']
        rho_phi = self.results['rho_phi']
        
        Omega_m = rho_m / (rho_m + rho_phi)
        Omega_phi = rho_phi / (rho_m + rho_phi)
        
        plt.figure(figsize=(10, 6))
        plt.plot(a_vals, Omega_m, label=r'$\Omega_m(a)$', linewidth=2)
        plt.plot(a_vals, Omega_phi, label=r'$\Omega_\phi(a)$', linewidth=2)
        plt.xscale('log')
        plt.xlabel('Scale factor a', fontsize=12)
        plt.ylabel('Density parameter', fontsize=12)
        plt.title('Matter vs Coherence Field Density Evolution', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.show()
    
    def compare_with_LCDM(self, z_max=2.0, savefig_prefix=None):
        """
        Compare H(z) and d_L(z) with matched ΛCDM model.
        
        Parameters:
        -----------
        z_max : float
            Maximum redshift for comparison
        savefig_prefix : str
            Prefix for saved figure filenames
        """
        z_vals = np.linspace(0.0, z_max, 200)
        
        # Coherence model
        H_scalar = self.compute_H_of_z(z_vals)
        dL_scalar = self.compute_dL(z_vals)
        
        # Matched ΛCDM
        Omega_m0 = self.results['Omega_m0']
        Omega_phi0 = self.results['Omega_phi0']
        a_sample = 1.0 / (1.0 + z_vals)
        H_LCDM = np.sqrt(Omega_m0 * a_sample**-3 + Omega_phi0)
        
        # ΛCDM luminosity distance
        dL_LCDM = []
        for z in z_vals:
            zs = np.linspace(0.0, z, 800)
            a_s = 1.0 / (1.0 + zs)
            H_s = np.sqrt(Omega_m0 * a_s**-3 + Omega_phi0)
            chi = trapz(1.0 / H_s, zs)
            dL_LCDM.append((1.0 + z) * chi)
        dL_LCDM = np.array(dL_LCDM)
        
        # Plot H(z)
        plt.figure(figsize=(10, 6))
        plt.plot(z_vals, H_scalar, label='Coherence field model', linewidth=2)
        plt.plot(z_vals, H_LCDM, '--', label=f'LCDM (Omega_m={Omega_m0:.3f})', linewidth=2)
        plt.xlabel('Redshift z', fontsize=12)
        plt.ylabel('H(z) / H₀', fontsize=12)
        plt.title('Expansion History Comparison', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if savefig_prefix:
            plt.savefig(f'{savefig_prefix}_H_z.png', dpi=300)
        plt.show()
        
        # Plot d_L(z)
        plt.figure(figsize=(10, 6))
        plt.plot(z_vals, dL_scalar, label='Coherence field model', linewidth=2)
        plt.plot(z_vals, dL_LCDM, '--', label=f'LCDM (Omega_m={Omega_m0:.3f})', linewidth=2)
        plt.xlabel('Redshift z', fontsize=12)
        plt.ylabel('Luminosity Distance $d_L$ (c/H₀)', fontsize=12)
        plt.title('Luminosity Distance Comparison', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if savefig_prefix:
            plt.savefig(f'{savefig_prefix}_dL_z.png', dpi=300)
        plt.show()
        
        # Residuals
        residual = (dL_scalar - dL_LCDM) / dL_LCDM * 100
        
        plt.figure(figsize=(10, 6))
        plt.plot(z_vals, residual, linewidth=2)
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Redshift z', fontsize=12)
        plt.ylabel('Residual (%)', fontsize=12)
        plt.title('Luminosity Distance Residual: (Coherence - ΛCDM) / ΛCDM', fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if savefig_prefix:
            plt.savefig(f'{savefig_prefix}_residual.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("Coherence Field Theory: Background Cosmology")
    print("=" * 60)
    
    # Initialize model
    cosmo = CoherenceCosmology(V0=1.0e-6, lambda_param=1.0)
    
    # Evolve background
    print("\nEvolving background cosmology...")
    results = cosmo.evolve(N_start=-7.0, N_end=0.0, n_steps=4000)
    
    print(f"\nPresent-day density parameters:")
    print(f"  Omega_m0  = {results['Omega_m0']:.4f}")
    print(f"  Omega_phi0  = {results['Omega_phi0']:.4f}")
    print(f"  Total = {results['Omega_m0'] + results['Omega_phi0']:.4f}")
    
    # Plot density evolution
    print("\nPlotting density evolution...")
    cosmo.plot_density_evolution(savefig='density_evolution.png')
    
    # Compare with ΛCDM
    print("\nComparing with LCDM...")
    cosmo.compare_with_LCDM(z_max=2.0, savefig_prefix='cosmology_comparison')
    
    print("\n" + "=" * 60)
    print("Background evolution complete!")
    print("=" * 60)

