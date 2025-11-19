"""
Cluster lensing profiles with coherence field.

Models gravitational lensing in galaxy clusters with coherence field contribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d


class ClusterLensing:
    """
    Model cluster lensing with coherence field.
    
    Compute:
    - Surface mass density Σ(R)
    - Convergence κ(R)
    - Shear γ(R)
    - Einstein radius
    """
    
    def __init__(self, z_lens=0.3, z_source=1.0):
        """
        Parameters:
        -----------
        z_lens : float
            Lens redshift
        z_source : float
            Source redshift
        """
        self.z_lens = z_lens
        self.z_source = z_source
        self.G = 4.30091e-6  # (km/s)^2 kpc / M_sun
        self.c = 299792.458  # km/s
        
        # Critical surface density (Σ_crit)
        # For now using simplified formula; should integrate with cosmology module
        self.compute_critical_density()
        
    def compute_critical_density(self):
        """Compute critical surface density for lensing."""
        # Simplified angular diameter distances (should use cosmology module)
        # D_L, D_S, D_LS in Mpc
        D_L = 1000.0 * self.z_lens  # Rough approximation
        D_S = 1000.0 * self.z_source
        D_LS = D_S - D_L
        
        # Critical density in M_sun / kpc^2
        # Σ_crit = c^2 / (4πG) * D_S / (D_L D_LS)
        self.Sigma_crit = (self.c**2 / (4 * np.pi * self.G)) * (D_S / (D_L * D_LS))
        
        print(f"Critical surface density: Sigma_crit = {self.Sigma_crit:.2e} M_sun/kpc^2")
    
    def set_baryonic_profile_NFW(self, M200, c, r_vir):
        """
        Set NFW profile for baryonic (ICM + galaxies) component.
        
        Parameters:
        -----------
        M200 : float
            Virial mass (M_sun)
        c : float
            Concentration parameter
        r_vir : float
            Virial radius (kpc)
        """
        self.M200 = M200
        self.c = c
        self.r_vir = r_vir
        self.r_s = r_vir / c
        
        # Normalization
        self.rho_s = M200 / (4 * np.pi * self.r_s**3 * (np.log(1 + c) - c / (1 + c)))
        
    def rho_NFW(self, r):
        """NFW density profile."""
        x = r / self.r_s
        return self.rho_s / (x * (1 + x)**2)
    
    def set_coherence_profile_simple(self, rho_c0, R_c):
        """
        Simple coherence halo profile.
        
        Parameters:
        -----------
        rho_c0 : float
            Central density
        R_c : float
            Core radius (kpc)
        """
        self.rho_c0 = rho_c0
        self.R_c = R_c
        
    def rho_coherence(self, r):
        """Coherence field density profile."""
        return self.rho_c0 / (1.0 + (r / self.R_c)**2)
    
    def surface_density(self, R, rho_func, r_max=10000):
        """
        Compute projected surface density Σ(R).
        
        Σ(R) = ∫ ρ(√(R² + z²)) dz  from -∞ to +∞
        
        Parameters:
        -----------
        R : float
            Projected radius (kpc)
        rho_func : callable
            3D density function ρ(r)
        r_max : float
            Integration cutoff
            
        Returns:
        --------
        Sigma : float
            Surface density (M_sun / kpc^2)
        """
        if R < 1e-6:
            R = 1e-6
        
        z_max = np.sqrt(r_max**2 - R**2) if r_max > R else 0.0
        if z_max < 1e-6:
            return 0.0
        
        z_vals = np.linspace(0, z_max, 500)
        r_vals = np.sqrt(R**2 + z_vals**2)
        integrand = rho_func(r_vals)
        
        # Factor of 2 for symmetry (integrate from 0 to z_max, multiply by 2)
        return 2.0 * trapz(integrand, z_vals)
    
    def convergence(self, R, Sigma):
        """
        Convergence κ = Σ / Σ_crit.
        
        Parameters:
        -----------
        R : array
            Projected radius
        Sigma : array
            Surface density
            
        Returns:
        --------
        kappa : array
            Convergence
        """
        return Sigma / self.Sigma_crit
    
    def average_convergence(self, R, kappa_func):
        """
        Average convergence inside R: κ̄(R) = (2/R²) ∫₀ᴿ κ(R') R' dR'
        
        Parameters:
        -----------
        R : float
            Radius
        kappa_func : callable
            Convergence as function of radius
            
        Returns:
        --------
        kappa_bar : float
            Average convergence
        """
        if R < 1e-6:
            return kappa_func(1e-6)
        
        R_vals = np.linspace(0, R, 300)
        kappa_vals = np.array([kappa_func(Ri) for Ri in R_vals])
        integrand = kappa_vals * R_vals
        
        return (2.0 / R**2) * trapz(integrand, R_vals)
    
    def shear(self, R, kappa, kappa_bar):
        """
        Tangential shear γ_t = κ̄(R) - κ(R).
        
        Parameters:
        -----------
        R : array
            Radius
        kappa : array
            Convergence
        kappa_bar : array
            Average convergence inside R
            
        Returns:
        --------
        gamma : array
            Tangential shear
        """
        return kappa_bar - kappa
    
    def compute_lensing_profile(self, R_array, include_coherence=True):
        """
        Compute full lensing profile.
        
        Parameters:
        -----------
        R_array : array
            Projected radii (kpc)
        include_coherence : bool
            Whether to include coherence field
            
        Returns:
        --------
        profiles : dict
            Dictionary with Sigma, kappa, gamma arrays
        """
        n = len(R_array)
        
        Sigma_NFW = np.zeros(n)
        Sigma_coh = np.zeros(n)
        
        print("Computing surface densities...")
        for i, R in enumerate(R_array):
            Sigma_NFW[i] = self.surface_density(R, self.rho_NFW)
            if include_coherence:
                Sigma_coh[i] = self.surface_density(R, self.rho_coherence)
            
            if i % 10 == 0:
                print(f"  Progress: {i+1}/{n}")
        
        Sigma_total = Sigma_NFW + (Sigma_coh if include_coherence else 0)
        
        kappa_total = self.convergence(R_array, Sigma_total)
        
        # Compute average convergence and shear
        print("Computing shear...")
        kappa_interp = interp1d(R_array, kappa_total, kind='cubic', 
                               fill_value='extrapolate')
        
        kappa_bar = np.array([self.average_convergence(R, kappa_interp) 
                             for R in R_array])
        
        gamma = self.shear(R_array, kappa_total, kappa_bar)
        
        return {
            'R': R_array,
            'Sigma_NFW': Sigma_NFW,
            'Sigma_coherence': Sigma_coh,
            'Sigma_total': Sigma_total,
            'kappa': kappa_total,
            'kappa_bar': kappa_bar,
            'gamma': gamma
        }
    
    def plot_lensing_profiles(self, profiles, savefig=None):
        """
        Plot lensing profiles.
        
        Parameters:
        -----------
        profiles : dict
            Output from compute_lensing_profile
        savefig : str
            Filename to save figure
        """
        R = profiles['R']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Surface density
        ax = axes[0, 0]
        ax.loglog(R, profiles['Sigma_NFW'], '--', label='NFW (baryons)', linewidth=2)
        if np.sum(profiles['Sigma_coherence']) > 0:
            ax.loglog(R, profiles['Sigma_coherence'], ':', 
                     label='Coherence field', linewidth=2)
        ax.loglog(R, profiles['Sigma_total'], '-', label='Total', linewidth=2.5)
        ax.axhline(self.Sigma_crit, color='k', linestyle='--', 
                  alpha=0.5, label='$\\Sigma_{\\rm crit}$')
        ax.set_xlabel('Projected Radius R (kpc)', fontsize=12)
        ax.set_ylabel('Surface Density $\\Sigma$ (M$_\\odot$/kpc$^2$)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        # Convergence
        ax = axes[0, 1]
        ax.loglog(R, profiles['kappa'], linewidth=2.5, color='C2')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='κ = 1')
        ax.set_xlabel('Projected Radius R (kpc)', fontsize=12)
        ax.set_ylabel('Convergence $\\kappa$', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        # Shear
        ax = axes[1, 0]
        ax.loglog(R, np.abs(profiles['gamma']), linewidth=2.5, color='C3')
        ax.set_xlabel('Projected Radius R (kpc)', fontsize=12)
        ax.set_ylabel('Tangential Shear $\\gamma_t$', fontsize=12)
        ax.grid(alpha=0.3)
        
        # Mass profile
        ax = axes[1, 1]
        M_enc = np.array([np.pi * R[i]**2 * profiles['Sigma_total'][i] 
                         for i in range(len(R))])
        ax.loglog(R, M_enc / 1e14, linewidth=2.5, color='C4')
        ax.set_xlabel('Projected Radius R (kpc)', fontsize=12)
        ax.set_ylabel('Enclosed Mass M(<R) ($10^{14}$ M$_\\odot$)', fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.suptitle(f'Cluster Lensing Profile (z_lens={self.z_lens}, z_src={self.z_source})', 
                    fontsize=14)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.show()


def example_cluster():
    """Run example cluster lensing calculation."""
    print("=" * 70)
    print("Cluster Lensing with Coherence Field")
    print("=" * 70)
    
    # Create lensing model
    lens = ClusterLensing(z_lens=0.3, z_source=1.0)
    
    # Set NFW profile (typical massive cluster)
    M200 = 1e15  # M_sun
    c = 4.0
    r_vir = 2000  # kpc
    lens.set_baryonic_profile_NFW(M200, c, r_vir)
    
    print(f"\nCluster parameters:")
    print(f"  M200 = {M200:.2e} M_sun")
    print(f"  c = {c}")
    print(f"  r_vir = {r_vir} kpc")
    print(f"  r_s = {lens.r_s:.2f} kpc")
    
    # Set coherence halo
    rho_c0 = 1e8  # M_sun / kpc^3
    R_c = 500  # kpc
    lens.set_coherence_profile_simple(rho_c0, R_c)
    
    print(f"\nCoherence halo:")
    print(f"  ρ_c0 = {rho_c0:.2e} M_sun/kpc^3")
    print(f"  R_c = {R_c} kpc")
    
    # Compute profiles
    print(f"\nComputing lensing profiles...")
    R_array = np.logspace(1, 3.5, 40)  # 10 kpc to ~3 Mpc
    profiles = lens.compute_lensing_profile(R_array, include_coherence=True)
    
    # Plot
    print("\nGenerating plots...")
    lens.plot_lensing_profiles(profiles, savefig='cluster_lensing_example.png')
    
    print("\n" + "=" * 70)
    print("Cluster lensing example complete!")
    print("=" * 70)


if __name__ == '__main__':
    example_cluster()

