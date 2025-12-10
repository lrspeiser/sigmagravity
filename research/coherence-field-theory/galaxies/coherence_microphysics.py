"""
Gravitational Polarization with Memory (GPM) - Microphysical Model

This implements a first-principles derivation of coherence density via
a gravitational dielectric response with memory and diffusion.

Core equation (steady state):
    (1 - ℓ² ∇²) ρ_coh = α · ρ_b

In spherical symmetry, this reduces to a Yukawa convolution:
    ρ_coh(r) = α ∫ G_ℓ(|r-s|) ρ_b(s) d³s
    
where G_ℓ(r) = exp(-r/ℓ) / (4π ℓ² r) is the Yukawa kernel.

Environmental gating naturally:
- Activates in cold, rotating disks (low Q, low σ_v)
- Suppresses in hot systems (high σ_v) → PPN safe
- Vanishes in homogeneous backgrounds (no disk structure) → cosmology safe

References:
- User's phenomenological Σ-Gravity (many_path_model/)
- This replaces per-galaxy K(R) tuning with global microphysics
"""

import numpy as np
from scipy.integrate import simpson
from typing import Callable, Tuple, Dict, Optional


class GravitationalPolarizationMemory:
    """
    Microphysical model for coherence density via gravitational polarization.
    
    Parameters
    ----------
    alpha0 : float
        Base susceptibility (dimensionless)
    ell0_kpc : float
        Base coherence length in kpc
    Qstar : float
        Toomre Q threshold for gating
    sigmastar : float
        Velocity dispersion threshold in km/s
    nQ : float
        Exponent for Q gating
    nsig : float
        Exponent for σ_v gating
    p : float
        Exponent for coherence length scaling with dynamics
    Mstar_Msun : float
        Mass scale for mass-dependent gating (M☉)
    nM : float
        Exponent for mass gating
    """
    
    def __init__(self, 
                 alpha0: float = 0.8,
                 ell0_kpc: float = 2.0,
                 Qstar: float = 2.0,
                 sigmastar: float = 30.0,
                 nQ: float = 2.0,
                 nsig: float = 2.0,
                 p: float = 1.0,
                 Mstar_Msun: float = 1e9,
                 nM: float = 1.0):
        self.alpha0 = alpha0
        self.ell0 = ell0_kpc
        self.Qstar = Qstar
        self.sigmastar = sigmastar
        self.nQ = nQ
        self.nsig = nsig
        self.p = p
        self.Mstar = Mstar_Msun
        self.nM = nM
    
    def environment_factors(self, 
                          Q: float = 2.0,
                          sigma_v: float = 10.0,
                          R_disk: float = 2.0,
                          M_total: Optional[float] = None,
                          cs: Optional[float] = None,
                          kappa: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute environmentally-gated susceptibility and coherence length.
        
        Parameters
        ----------
        Q : float
            Toomre stability parameter
        sigma_v : float
            Velocity dispersion (km/s)
        R_disk : float
            Disk scale length (kpc)
        M_total : float, optional
            Total baryon mass (M☉) for mass-dependent gating
        cs : float, optional
            Sound speed (km/s)
        kappa : float, optional
            Epicyclic frequency (km/s/kpc)
            
        Returns
        -------
        alpha : float
            Effective susceptibility (gated)
        ell : float
            Coherence length in kpc
        """
        # Susceptibility gate (suppresses in hot/stable systems)
        # α → 0 when Q >> Q* or σ_v >> σ*
        gate_env = 1.0 / (1.0 + (Q / self.Qstar)**self.nQ + 
                          (sigma_v / self.sigmastar)**self.nsig)
        
        # Mass-dependent gating (suppresses in massive galaxies)
        # α → 0 when M >> M*
        gate_mass = 1.0
        if M_total is not None and M_total > 0:
            gate_mass = 1.0 / (1.0 + (M_total / self.Mstar)**self.nM)
        
        # Coherence length scales with disk size
        # ℓ ∝ R_disk^p (larger disks → longer coherence lengths)
        ell = self.ell0 * (R_disk / 2.0)**self.p  # normalize to 2 kpc
        
        # Override with detailed dynamics if available
        if cs is not None and kappa is not None and R_disk > 0:
            # ℓ ∝ (c_s / κ R_disk)^p scales with dynamical properties
            ell = self.ell0 * ((cs / (kappa * R_disk + 1e-10)) ** self.p)
        
        # Combined gating
        alpha = self.alpha0 * gate_env * gate_mass
        
        return alpha, max(1e-3, ell)  # floor ell to avoid division by zero
    
    def yukawa_kernel(self, r: float, ell: float) -> float:
        """
        Yukawa kernel G_ℓ(r) = exp(-r/ℓ) / (4π ℓ² r)
        
        Parameters
        ----------
        r : float
            Distance (kpc)
        ell : float
            Coherence length (kpc)
            
        Returns
        -------
        G : float
            Kernel value
        """
        if r < 1e-6:  # regularization at origin
            # lim_{r→0} G_ℓ(r) = 1/(4π ℓ³) via L'Hopital
            return 1.0 / (4.0 * np.pi * ell**3)
        return np.exp(-r / ell) / (4.0 * np.pi * ell**2 * r)
    
    def yukawa_convolution_spherical(self,
                                    r_target: float,
                                    rho_b_func: Callable[[float], float],
                                    alpha: float,
                                    ell: float,
                                    r_max: float = 50.0,
                                    n_points: int = 1000) -> float:
        """
        Compute ρ_coh(r) = α ∫ G_ℓ(|r-s|) ρ_b(s) 4π s² ds
        
        In spherical symmetry with radial source ρ_b(s).
        
        Parameters
        ----------
        r_target : float
            Target radius (kpc)
        rho_b_func : callable
            Baryon density function ρ_b(r) in M☉/kpc³
        alpha : float
            Susceptibility
        ell : float
            Coherence length (kpc)
        r_max : float
            Maximum integration radius (kpc)
        n_points : int
            Number of integration points
            
        Returns
        -------
        rho_coh : float
            Coherence density at r_target (M☉/kpc³)
        """
        # Adaptive integration range
        r_max = max(r_max, 5.0 * ell + r_target, 2.0 * r_target)
        
        # Radial grid
        s = np.linspace(1e-4, r_max, n_points)
        
        # Distance from target point
        rs = np.abs(r_target - s)
        
        # Yukawa kernel
        G = np.array([self.yukawa_kernel(ri, ell) for ri in rs])
        
        # Baryon density at source points
        rho_b = np.array([rho_b_func(si) for si in s])
        
        # Integrand: 4π s² ρ_b(s) G_ℓ(|r-s|)
        integrand = 4.0 * np.pi * s**2 * rho_b * G
        
        # Integrate using Simpson's rule
        return alpha * simpson(integrand, x=s)
    
    def make_rho_coh(self,
                    rho_b_func: Callable[[float], float],
                    Q: float = 2.0,
                    sigma_v: float = 10.0,
                    R_disk: float = 2.0,
                    M_total: Optional[float] = None,
                    cs: Optional[float] = None,
                    kappa: Optional[float] = None,
                    r_max: float = 50.0,
                    use_axisymmetric: bool = False,
                    h_z: float = 0.3) -> Tuple[Callable[[float], float], Dict]:
        """
        Create coherence density function from baryon density.
        
        Uses analytic Yukawa convolution - either spherical (fast) or
        axisymmetric disk geometry (more accurate for spirals).
        
        Parameters
        ----------
        rho_b_func : callable
            Baryon density ρ_b(r) in M☉/kpc³
        Q : float
            Toomre parameter
        sigma_v : float
            Velocity dispersion (km/s)
        R_disk : float
            Disk scale length (kpc)
        M_total : float, optional
            Total baryon mass (M☉) for mass-dependent gating
        cs : float, optional
            Sound speed (km/s)
        kappa : float, optional
            Epicyclic frequency (km/s/kpc)
        r_max : float
            Maximum radius for integration (kpc)
        use_axisymmetric : bool
            If True, use axisymmetric disk convolution (more accurate, slightly slower)
            If False, use spherical convolution (faster, good for dwarfs)
        h_z : float
            Disk scale height (kpc), only used if use_axisymmetric=True
            
        Returns
        -------
        rho_coh_func : callable
            Function that computes ρ_coh(r)
        params : dict
            Dictionary with effective α, ℓ, and diagnostics
        """
        from scipy.integrate import cumulative_trapezoid
        
        # Get environment-gated parameters
        alpha, ell = self.environment_factors(Q, sigma_v, R_disk, M_total, cs, kappa)
        
        # Choose convolution method
        if use_axisymmetric:
            # Use axisymmetric disk convolution (more accurate for spirals)
            from galaxies.coherence_microphysics_axisym import AxiSymmetricYukawaConvolver
            
            convolver = AxiSymmetricYukawaConvolver(h_z=h_z)
            
            def rho_coh_of_r(r):
                """Axisymmetric disk convolution."""
                r_arr = np.atleast_1d(r)
                scalar_input = np.isscalar(r)
                
                rho_coh = convolver.convolve_volume_density(
                    rho_b_func, alpha, ell, r_arr, R_max=r_max
                )
                
                return float(rho_coh[0]) if scalar_input else rho_coh
            
            params = {
                'alpha': alpha,
                'ell_kpc': ell,
                'Q': Q,
                'sigma_v': sigma_v,
                'R_disk': R_disk,
                'M_total': M_total,
                'gate_strength': alpha / self.alpha0 if self.alpha0 > 0 else 0.0,
                'coherence_scale': ell / R_disk if R_disk > 0 else 0.0,
                'geometry': 'axisymmetric',
                'h_z': h_z
            }
            
            return rho_coh_of_r, params
        
        # Pre-compute on fixed grid for stability
        r_max_safe = max(r_max, 10.0*ell, 5.0*R_disk)
        grid = np.geomspace(1e-4, r_max_safe, 2048)
        
        # Evaluate baryon density on grid
        rho_b_grid = np.array([rho_b_func(g) for g in grid])
        
        # Cumulative integral for r < s: J_<(r) = ∫_0^r s sinh(s/ℓ) ρ_b(s) ds
        integrand_lt = grid * np.sinh(grid/ell) * rho_b_grid
        Jlt = cumulative_trapezoid(integrand_lt, grid, initial=0.0)
        
        # Cumulative integral for r > s: J_>(r) = ∫_r^∞ s exp(-s/ℓ) ρ_b(s) ds
        # Compute total integral, then subtract cumulative from left
        integrand_gt = grid * np.exp(-grid/ell) * rho_b_grid
        Jgt_cumulative = cumulative_trapezoid(integrand_gt, grid, initial=0.0)
        Jgt_total = Jgt_cumulative[-1]
        Jgt = Jgt_total - Jgt_cumulative  # Integral from r to infinity
        
        # Create ρ_coh function using cached integrals
        def rho_coh_of_r(r):
            """
            Analytic formula: ρ_coh(r) = α/(ℓ²r) [e^(-r/ℓ) J_<(r) + sinh(r/ℓ) J_>(r)]
            """
            r_safe = np.maximum(np.asarray(r), 1e-6)
            scalar_input = np.isscalar(r)
            r_safe = np.atleast_1d(r_safe)
            
            # Interpolate pre-computed integrals
            Jlt_r = np.interp(r_safe, grid, Jlt)
            Jgt_r = np.interp(r_safe, grid, Jgt)
            
            # Apply analytic formula
            pref = alpha / (ell**2 * r_safe)
            result = pref * (np.exp(-r_safe/ell) * Jlt_r + np.sinh(r_safe/ell) * Jgt_r)
            
            return float(result[0]) if scalar_input else result
        
        # Diagnostic parameters
        params = {
            'alpha': alpha,
            'ell_kpc': ell,
            'Q': Q,
            'sigma_v': sigma_v,
            'R_disk': R_disk,
            'M_total': M_total,
            'gate_strength': alpha / self.alpha0 if self.alpha0 > 0 else 0.0,
            'coherence_scale': ell / R_disk if R_disk > 0 else 0.0
        }
        
        return rho_coh_of_r, params


# ============================================================================
# HELPER FUNCTIONS FOR COMMON PROFILES
# ============================================================================

def exponential_disk_density(r: float, 
                            Sigma0: float, 
                            R_d: float, 
                            h_z: float = 0.3) -> float:
    """
    Exponential disk density: ρ(r,z) = (Σ₀/2h_z) exp(-r/R_d)
    
    Parameters
    ----------
    r : float
        Radius (kpc)
    Sigma0 : float
        Central surface density (M☉/kpc²)
    R_d : float
        Disk scale length (kpc)
    h_z : float
        Scale height (kpc)
        
    Returns
    -------
    rho : float
        Volume density at r (M☉/kpc³)
    """
    return (Sigma0 / (2.0 * h_z)) * np.exp(-r / R_d)


def plummer_sphere_density(r: float, M_tot: float, a: float) -> float:
    """
    Plummer sphere density: ρ(r) = (3M/4πa³) (1 + r²/a²)^(-5/2)
    
    Parameters
    ----------
    r : float
        Radius (kpc)
    M_tot : float
        Total mass (M☉)
    a : float
        Plummer scale length (kpc)
        
    Returns
    -------
    rho : float
        Volume density at r (M☉/kpc³)
    """
    return (3.0 * M_tot / (4.0 * np.pi * a**3)) * (1.0 + (r / a)**2)**(-2.5)


def nfw_density(r: float, rho_s: float, r_s: float) -> float:
    """
    NFW profile density: ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
    
    Parameters
    ----------
    r : float
        Radius (kpc)
    rho_s : float
        Characteristic density (M☉/kpc³)
    r_s : float
        Scale radius (kpc)
        
    Returns
    -------
    rho : float
        Volume density at r (M☉/kpc³)
    """
    x = r / r_s
    return rho_s / (x * (1.0 + x)**2)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_ddo154():
    """Example: Apply GPM to DDO154-like dwarf galaxy."""
    print("="*80)
    print("GPM Example: DDO154-like Dwarf")
    print("="*80)
    
    # Baryon profile (exponential disk)
    M_disk = 1.2e9  # M☉
    R_disk = 1.6    # kpc
    Sigma0 = M_disk / (2.0 * np.pi * R_disk**2)  # M☉/kpc²
    
    def rho_b(r):
        return exponential_disk_density(r, Sigma0, R_disk, h_z=0.3)
    
    # Environment (cold dwarf)
    Q = 1.5
    sigma_v = 8.0  # km/s
    
    # Create GPM model
    gpm = GravitationalPolarizationMemory(
        alpha0=0.9,
        ell0_kpc=2.0,
        Qstar=2.0,
        sigmastar=25.0
    )
    
    # Generate coherence density
    rho_coh_func, params = gpm.make_rho_coh(
        rho_b, Q=Q, sigma_v=sigma_v, R_disk=R_disk
    )
    
    print(f"\nEnvironmental Parameters:")
    print(f"  Q = {Q:.2f}")
    print(f"  σ_v = {sigma_v:.1f} km/s")
    print(f"  R_disk = {R_disk:.2f} kpc")
    
    print(f"\nEffective GPM Parameters:")
    print(f"  α = {params['alpha']:.3f} (gate strength: {params['gate_strength']:.1%})")
    print(f"  ℓ = {params['ell_kpc']:.2f} kpc (ℓ/R_d = {params['coherence_scale']:.2f})")
    
    # Evaluate at sample radii
    r_sample = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    rho_b_sample = np.array([rho_b(r) for r in r_sample])
    rho_coh_sample = rho_coh_func(r_sample)
    
    print(f"\nDensity Profiles:")
    print(f"{'r (kpc)':<10} {'ρ_b':<15} {'ρ_coh':<15} {'ρ_coh/ρ_b':<15}")
    print("-"*60)
    for i in range(len(r_sample)):
        ratio = rho_coh_sample[i] / rho_b_sample[i] if rho_b_sample[i] > 0 else 0
        print(f"{r_sample[i]:<10.2f} {rho_b_sample[i]:<15.2e} {rho_coh_sample[i]:<15.2e} {ratio:<15.3f}")
    
    print("\n" + "="*80)
    print("GPM successfully generates coherence halo from baryons!")
    print("="*80)


if __name__ == '__main__':
    example_ddo154()
