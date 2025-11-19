"""
Galaxy rotation curve modeling with coherence field halos.

Computes circular velocities from baryonic matter + coherence scalar field.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d


class GalaxyRotationCurve:
    """
    Model galaxy rotation curves with coherence field halo.
    
    Components:
    - Baryonic matter (disk + bulge)
    - Coherence scalar field φ(r) providing extra gravitating mass
    """
    
    def __init__(self, G=4.30091e-6):  # G in (km/s)^2 kpc / M_sun
        """
        Parameters:
        -----------
        G : float
            Gravitational constant in convenient units
        """
        self.G = G
        
    def set_baryon_profile(self, M_disk, R_disk, M_bulge=0.0, R_bulge=0.0):
        """
        Set exponential disk + bulge profiles.
        
        Parameters:
        -----------
        M_disk : float
            Total disk mass (M_sun)
        R_disk : float
            Disk scale length (kpc)
        M_bulge : float
            Bulge mass (M_sun)
        R_bulge : float
            Bulge scale radius (kpc)
        """
        self.M_disk = M_disk
        self.R_disk = R_disk
        self.M_bulge = M_bulge
        self.R_bulge = R_bulge
        
    def rho_disk(self, r):
        """Exponential disk density profile."""
        rho_0 = self.M_disk / (4 * np.pi * self.R_disk**3)
        return rho_0 * np.exp(-r / self.R_disk)
    
    def rho_bulge(self, r):
        """Hernquist bulge density profile."""
        if self.M_bulge == 0:
            return 0.0
        a = self.R_bulge
        return (self.M_bulge / (2 * np.pi)) * (a / r) / (r + a)**3
    
    def set_coherence_halo_simple(self, rho_c0, R_c):
        """
        Set simple pseudo-isothermal coherence halo.
        
        Parameters:
        -----------
        rho_c0 : float
            Central coherence density parameter
        R_c : float
            Core radius (kpc)
        """
        self.rho_c0 = rho_c0
        self.R_c = R_c
        self.halo_type = 'simple'
        
    def rho_coherence_simple(self, r):
        """Simple pseudo-isothermal coherence halo."""
        return self.rho_c0 / (1.0 + (r / self.R_c)**2)
    
    def set_coherence_halo_field(self, phi_profile, V_func):
        """
        Set coherence halo from solved scalar field profile.
        
        Parameters:
        -----------
        phi_profile : callable
            φ(r) function
        V_func : callable
            Potential V(φ) function
        """
        self.phi_profile = phi_profile
        self.V_func = V_func
        self.halo_type = 'field'
        
    def rho_coherence_field(self, r):
        """Coherence density from scalar field energy."""
        phi = self.phi_profile(r)
        dphi_dr = self._numerical_derivative(self.phi_profile, r)
        return 0.5 * dphi_dr**2 + self.V_func(phi)
    
    def _numerical_derivative(self, func, x, h=1e-5):
        """Numerical derivative."""
        return (func(x + h) - func(x - h)) / (2 * h)
    
    def mass_enclosed(self, r, rho_func):
        """
        Compute enclosed mass from density profile.
        
        Parameters:
        -----------
        r : float
            Radius (kpc)
        rho_func : callable
            Density function ρ(r)
            
        Returns:
        --------
        M : float
            Enclosed mass (M_sun)
        """
        if r < 1e-6:
            return 0.0
        rs = np.linspace(1e-6, r, 800)
        integrand = 4.0 * np.pi * rs**2 * rho_func(rs)
        return trapz(integrand, rs)
    
    def circular_velocity(self, r):
        """
        Compute circular velocity at radius r.
        
        v^2(r) = G M(<r) / r
        
        Parameters:
        -----------
        r : float or array
            Radius (kpc)
            
        Returns:
        --------
        v : float or array
            Circular velocity (km/s)
        """
        scalar_input = np.isscalar(r)
        r = np.atleast_1d(r)
        
        v = np.zeros_like(r)
        
        for i, ri in enumerate(r):
            if ri < 1e-6:
                v[i] = 0.0
                continue
                
            # Baryonic mass
            M_baryon = (self.mass_enclosed(ri, self.rho_disk) + 
                       self.mass_enclosed(ri, self.rho_bulge))
            
            # Coherence halo mass
            if self.halo_type == 'simple':
                M_coh = self.mass_enclosed(ri, self.rho_coherence_simple)
            elif self.halo_type == 'field':
                M_coh = self.mass_enclosed(ri, self.rho_coherence_field)
            else:
                M_coh = 0.0
            
            M_total = M_baryon + M_coh
            v[i] = np.sqrt(self.G * M_total / ri)
        
        return v[0] if scalar_input else v
    
    def plot_rotation_curve(self, r_max=30.0, n_points=200, 
                           observed_r=None, observed_v=None, observed_err=None,
                           savefig=None):
        """
        Plot rotation curve with components.
        
        Parameters:
        -----------
        r_max : float
            Maximum radius (kpc)
        n_points : int
            Number of points
        observed_r, observed_v, observed_err : arrays
            Observed data for comparison (optional)
        savefig : str
            Save figure filename
        """
        r_vals = np.linspace(0.5, r_max, n_points)
        
        # Compute components
        v_baryon = np.zeros(n_points)
        v_total = np.zeros(n_points)
        
        for i, r in enumerate(r_vals):
            M_b = (self.mass_enclosed(r, self.rho_disk) + 
                  self.mass_enclosed(r, self.rho_bulge))
            v_baryon[i] = np.sqrt(self.G * M_b / r)
            v_total[i] = self.circular_velocity(r)
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        if observed_r is not None:
            if observed_err is not None:
                plt.errorbar(observed_r, observed_v, yerr=observed_err, 
                           fmt='o', color='black', label='Observed', 
                           markersize=4, capsize=3, alpha=0.6)
            else:
                plt.plot(observed_r, observed_v, 'ko', label='Observed', 
                        markersize=4, alpha=0.6)
        
        plt.plot(r_vals, v_baryon, '--', label='Baryons only', linewidth=2, alpha=0.7)
        plt.plot(r_vals, v_total, '-', label='Baryons + Coherence field', 
                linewidth=2.5)
        
        plt.xlabel('Radius (kpc)', fontsize=12)
        plt.ylabel('Circular Velocity (km/s)', fontsize=12)
        plt.title('Galaxy Rotation Curve', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.show()


def toy_example():
    """Run toy example from theory document."""
    print("=" * 60)
    print("Toy Galaxy Rotation Curve Example")
    print("=" * 60)
    
    # Create model
    galaxy = GalaxyRotationCurve(G=1.0)  # Arbitrary units
    
    # Baryons: exponential disk
    galaxy.set_baryon_profile(M_disk=1.0, R_disk=2.0)
    
    # Coherence halo: pseudo-isothermal
    galaxy.set_coherence_halo_simple(rho_c0=0.2, R_c=8.0)
    
    # Plot
    galaxy.plot_rotation_curve(r_max=30.0, savefig='toy_rotation_curve.png')
    
    print("\nToy example complete!")
    print("Generated: toy_rotation_curve.png")


if __name__ == '__main__':
    toy_example()

