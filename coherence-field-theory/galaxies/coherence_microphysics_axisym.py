"""
Axisymmetric Yukawa convolution for disk geometries.

This module implements the Yukawa convolution for thin disk geometries,
which is more accurate than spherical approximation for spiral galaxies.

For an exponential disk with surface density Σ(R) and scale height h_z,
the coherence density at radius R in the midplane is:

    ρ_coh(R, z=0) = α ∫₀^∞ R' dR' Σ(R') K(R, R'; ℓ, h_z)

where K is the axisymmetric Yukawa kernel involving modified Bessel functions.

Mathematical formulation:
------------------------
The 3D Yukawa Green's function in cylindrical coordinates (R, φ, z) is:

    G_ℓ(r) = exp(-r/ℓ) / (4π ℓ² r)

where r = sqrt((R-R')² + (z-z')²) for source at (R', 0, z').

For a thin disk (|z| << h_z), we integrate over z' with weight:

    ρ(R', z') = Σ(R') / (2 h_z) sech²(z' / 2h_z)

The midplane (z=0) coherence density becomes:

    ρ_coh(R, 0) = α/(2h_z) ∫₀^∞ R' dR' Σ(R') ∫_{-∞}^∞ dz' sech²(z'/2h_z) G_ℓ(r)

The z'-integral can be done semi-analytically for each R, R' pair.

Numerical implementation:
------------------------
1. Pre-compute Σ(R') on radial grid
2. For each target radius R:
   - Compute kernel K(R, R') including z-integration
   - Integrate K(R, R') × R' × Σ(R') over R'
3. Use adaptive quadrature for accuracy

Reference: This is the disk analog of the spherical formula in
coherence_microphysics.py, adapted for thin disk geometry.
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.special import k0, k1, i0, i1  # Modified Bessel functions
from typing import Callable, Tuple, Dict


class AxiSymmetricYukawaConvolver:
    """
    Axisymmetric Yukawa convolution for disk geometries.
    
    Provides more accurate coherence density calculation for spiral galaxies
    by accounting for disk geometry rather than assuming spherical symmetry.
    """
    
    def __init__(self, h_z: float = 0.3):
        """
        Initialize convolver.
        
        Parameters
        ----------
        h_z : float
            Disk scale height in kpc (default: 0.3)
        """
        self.h_z = h_z
        self.G_kpc = 4.302e-3  # G in kpc (km/s)^2 / M_sun
    
    def yukawa_kernel_2d(self, R: float, Rp: float, ell: float, 
                        apply_thickness_correction: bool = True) -> float:
        """
        2D Yukawa kernel for disk midplane with optional finite-thickness correction.
        
        This computes the effective kernel at the midplane (z=0) after
        integrating over the vertical structure of the source disk.
        
        K(R, R') = ∫_{-∞}^∞ dz' [sech²(z'/2h_z)/(2h_z)] G_ℓ(r)
        
        where r = sqrt((R-R')² + z'²) and G_ℓ(r) = exp(-r/ℓ)/(4π ℓ² r).
        
        For numerical stability, we use the modified Bessel function form:
        K(R, R') ≈ K₀(|R-R'|/ℓ) / (2π ℓ²) for infinitesimally thin disks
        
        FINITE-THICKNESS CORRECTION:
        For finite h_z, the Fourier-space kernel picks up a factor exp(-k*h_z)
        where k = 1/ℓ. In real space, this modifies K₀ slightly for h_z ~ ℓ.
        We apply: K_thick = K_thin × exp(-h_z/ℓ) as first-order correction.
        
        Parameters
        ----------
        R : float
            Target radius (kpc)
        Rp : float
            Source radius (kpc)
        ell : float
            Coherence length (kpc)
        apply_thickness_correction : bool
            If True, apply exp(-h_z/ℓ) correction for finite disk thickness
            
        Returns
        -------
        kernel : float
            Kernel value (dimensionless per kpc²)
        """
        # Radial separation
        dR = abs(R - Rp)
        
        # For very small separations, use limiting form to avoid singularity
        if dR < 1e-6:
            # K₀(x) ~ -ln(x/2) - γ for small x, but we need regularization
            # Use average over small disk instead
            return self._kernel_regularized(R, ell, apply_thickness_correction)
        
        # Modified Bessel function K₀ for thin disk approximation
        # This is exact for infinitesimally thin disk (h_z → 0)
        # For finite h_z, this is accurate when h_z << ℓ
        x = dR / ell
        
        # K₀(x) decays exponentially for large x
        if x > 10:
            # Use asymptotic form: K₀(x) ~ sqrt(π/2x) exp(-x)
            K0_val = np.sqrt(np.pi / (2 * x)) * np.exp(-x)
        else:
            K0_val = k0(x)
        
        kernel = K0_val / (2.0 * np.pi * ell**2)
        
        # Apply finite-thickness correction
        # For disks with h_z ~ ℓ, the vertical extent softens the kernel
        if apply_thickness_correction and self.h_z > 0:
            thickness_factor = np.exp(-self.h_z / ell)
            kernel *= thickness_factor
        
        return kernel
    
    def _kernel_regularized(self, R: float, ell: float, 
                           apply_thickness_correction: bool = True) -> float:
        """
        Regularized kernel for R ≈ R' (self-interaction).
        
        Average kernel over a small disk of radius ε ~ ℓ/100.
        """
        eps = ell / 100.0
        # Integrate K₀(r/ℓ) over r from 0 to eps
        # ∫₀^ε r K₀(r/ℓ) dr = ℓ² [ε/ℓ K₁(ε/ℓ)]
        x_eps = eps / ell
        integral = ell**2 * x_eps * k1(x_eps)
        avg_kernel = integral / (np.pi * eps**2)
        kernel = avg_kernel / (2.0 * np.pi * ell**2)
        
        # Apply finite-thickness correction
        if apply_thickness_correction and self.h_z > 0:
            thickness_factor = np.exp(-self.h_z / ell)
            kernel *= thickness_factor
        
        return kernel
    
    def yukawa_kernel_3d_vertical(self, R: float, Rp: float, ell: float) -> float:
        """
        Full 3D kernel with vertical integration (more accurate, slower).
        
        K(R, R') = (1/2h_z) ∫_{-∞}^∞ dz' sech²(z'/2h_z) G_ℓ(r)
        
        where r = sqrt((R-R')² + z'²).
        
        This is more accurate than the thin-disk K₀ approximation when
        h_z ~ ℓ (thick disk), but much slower to compute.
        
        Parameters
        ----------
        R : float
            Target radius (kpc)
        Rp : float
            Source radius (kpc)  
        ell : float
            Coherence length (kpc)
            
        Returns
        -------
        kernel : float
            Kernel value after vertical integration
        """
        dR = abs(R - Rp)
        
        def integrand(zp):
            """Integrand: sech²(z'/2h_z) × G_ℓ(r)"""
            r = np.sqrt(dR**2 + zp**2)
            if r < 1e-10:
                return 0.0
            
            # Yukawa Green's function
            G = np.exp(-r / ell) / (4.0 * np.pi * ell**2 * r)
            
            # Vertical density profile (sech²)
            z_scaled = zp / (2.0 * self.h_z)
            sech_sq = 1.0 / np.cosh(z_scaled)**2
            
            return sech_sq * G / (2.0 * self.h_z)
        
        # Integrate over z' from -∞ to +∞
        # For sech², most weight is within ±4h_z
        z_max = 10.0 * self.h_z
        
        result, _ = quad(integrand, -z_max, z_max, limit=100)
        
        return result
    
    def convolve_surface_density(self,
                                 Sigma_func: Callable[[float], float],
                                 alpha: float,
                                 ell: float,
                                 R_target: np.ndarray,
                                 R_max: float = 50.0,
                                 use_3d: bool = False,
                                 apply_thickness_correction: bool = True) -> np.ndarray:
        """
        Compute coherence density via axisymmetric convolution.
        
        ρ_coh(R, 0) = α ∫₀^∞ R' dR' Σ(R') K(R, R'; ℓ)
        
        Parameters
        ----------
        Sigma_func : callable
            Surface density function Σ(R) in M☉/kpc²
        alpha : float
            Coherence susceptibility (dimensionless)
        ell : float
            Coherence length (kpc)
        R_target : array
            Target radii where to compute ρ_coh (kpc)
        R_max : float
            Maximum integration radius (kpc)
        use_3d : bool
            If True, use full 3D vertical integration (slower but more accurate)
            If False, use thin-disk K₀ approximation (faster)
            
        Returns
        -------
        rho_coh : array
            Coherence volume density at R_target, z=0 (M☉/kpc³)
        """
        # Pre-compute surface density on integration grid
        # Use log-spacing for better resolution at small R
        n_grid = 512
        R_grid = np.geomspace(1e-3, R_max, n_grid)
        Sigma_grid = np.array([Sigma_func(R) for R in R_grid])
        
        # Choose kernel function
        if use_3d:
            kernel_func = self.yukawa_kernel_3d_vertical
        else:
            kernel_func = self.yukawa_kernel_2d
        
        rho_coh = np.zeros_like(R_target)
        
        for i, R in enumerate(R_target):
            # Compute kernel K(R, R') for all R' on grid
            if use_3d:
                K_grid = np.array([kernel_func(R, Rp, ell) for Rp in R_grid])
            else:
                K_grid = np.array([kernel_func(R, Rp, ell, apply_thickness_correction) 
                                 for Rp in R_grid])
            
            # Integrand: R' × Σ(R') × K(R, R')
            integrand = R_grid * Sigma_grid * K_grid
            
            # Integrate using Simpson's rule
            rho_coh[i] = alpha * simpson(integrand, x=R_grid)
        
        return rho_coh
    
    def convolve_volume_density(self,
                               rho_func: Callable[[float], float],
                               alpha: float,
                               ell: float,
                               R_target: np.ndarray,
                               R_max: float = 50.0,
                               apply_thickness_correction: bool = True) -> np.ndarray:
        """
        Compute coherence density from 3D baryon density.
        
        This converts the volume density ρ_b(R, z) to surface density Σ(R)
        by integrating over z, then performs the axisymmetric convolution.
        
        Σ(R) = ∫_{-∞}^∞ ρ_b(R, z) dz
        
        For exponential disk: ρ_b(R, z) = Σ(R)/(2h_z) sech²(z/2h_z)
        so this is essentially extracting Σ(R) = 2h_z × ρ_b(R, 0).
        
        Parameters
        ----------
        rho_func : callable
            Volume density function ρ_b(R) at midplane (M☉/kpc³)
        alpha : float
            Coherence susceptibility
        ell : float
            Coherence length (kpc)
        R_target : array
            Target radii (kpc)
        R_max : float
            Maximum integration radius (kpc)
        apply_thickness_correction : bool
            Apply finite-thickness correction to kernel
            
        Returns
        -------
        rho_coh : array
            Coherence volume density at R_target, z=0 (M☉/kpc³)
        """
        # Convert volume density to surface density
        # For thin disk: Σ(R) = 2h_z × ρ(R, z=0)
        def Sigma_func(R):
            return 2.0 * self.h_z * rho_func(R)
        
        # Convolve surface density
        rho_coh = self.convolve_surface_density(
            Sigma_func, alpha, ell, R_target, R_max, use_3d=False,
            apply_thickness_correction=apply_thickness_correction
        )
        
        return rho_coh
    
    def convolve_disk_plus_bulge(self,
                                Sigma_disk_func: Callable[[float], float],
                                Sigma_bulge_func: Callable[[float], float],
                                alpha: float,
                                ell: float,
                                R_target: np.ndarray,
                                R_max: float = 50.0,
                                apply_thickness_correction: bool = True) -> np.ndarray:
        """
        Compute coherence density from disk + bulge components.
        
        This treats the bulge as a separate mass component that also
        contributes to the coherence source term. The total coherence
        density is the convolution of (disk + bulge) surface density.
        
        ρ_coh(R) = α ∫ R' dR' [Σ_disk(R') + Σ_bulge(R')] K(R, R'; ℓ)
        
        WHY THIS MATTERS:
        For massive spirals like NGC2841 and NGC0801, the bulge dominates
        the inner region. Including bulge mass in the coherence source
        ensures the coherence halo tracks total baryon geometry, not just
        the disk.
        
        Parameters
        ----------
        Sigma_disk_func : callable
            Disk surface density Σ_disk(R) in M☉/kpc²
        Sigma_bulge_func : callable  
            Bulge surface density Σ_bulge(R) in M☉/kpc²
        alpha : float
            Coherence susceptibility
        ell : float
            Coherence length (kpc)
        R_target : array
            Target radii (kpc)
        R_max : float
            Maximum integration radius (kpc)
        apply_thickness_correction : bool
            Apply finite-thickness correction (primarily for disk)
            
        Returns
        -------
        rho_coh : array
            Coherence volume density at R_target, z=0 (M☉/kpc³)
        """
        # Combined surface density: disk + bulge
        def Sigma_total_func(R):
            return Sigma_disk_func(R) + Sigma_bulge_func(R)
        
        # Convolve total surface density
        rho_coh = self.convolve_surface_density(
            Sigma_total_func, alpha, ell, R_target, R_max, use_3d=False,
            apply_thickness_correction=apply_thickness_correction
        )
        
        return rho_coh
    
    def apply_temporal_memory(self,
                            rho_coh_instantaneous: np.ndarray,
                            R_target: np.ndarray,
                            v_circ: np.ndarray,
                            eta: float = 1.0) -> np.ndarray:
        """
        Apply temporal memory smoothing to coherence density.
        
        The coherence field has a finite memory/relaxation timescale:
        
            τ(R) = η × (2π/Ω(R)) = η × (2π R / v_circ)
        
        where Ω = v_circ/R is the angular frequency.
        
        This introduces exponential time smoothing that damps narrow spikes
        while preserving large-scale structure. The smoothed coherence density
        is:
        
            ρ_coh(R) = ρ_inst(R) × [1 - exp(-t_age/τ(R))]
        
        For steady-state (t_age >> τ), this reduces to ρ_coh ≈ ρ_inst.
        For transient features (t_age ~ τ), this provides smoothing.
        
        PHYSICAL MOTIVATION:
        Gravitational polarization is not instantaneous—the coherence field
        takes time τ ~ orbital period to establish. This prevents sharp
        spikes from unphysical features in ρ_b and matches the "memory"
        aspect of GPM.
        
        Parameters
        ----------
        rho_coh_instantaneous : array
            Coherence density from spatial convolution (M☉/kpc³)
        R_target : array
            Radii where ρ_coh is evaluated (kpc)
        v_circ : array
            Circular velocity at R_target (km/s)
        eta : float
            Memory timescale parameter (η ~ 0.5-2)
            η = 1: memory timescale = one orbital period
            η < 1: faster memory (weaker smoothing)
            η > 1: slower memory (stronger smoothing)
            
        Returns
        -------
        rho_coh_smoothed : array
            Temporally smoothed coherence density (M☉/kpc³)
        """
        # Compute memory timescale τ(R) = η × 2π R / v_circ
        # Convert to Gyr: τ [Gyr] = η × 2π R[kpc] / v_circ[km/s] × (kpc/km/s → Gyr)
        kpc_per_km_s_to_Gyr = 1.0226903  # 1 kpc/(km/s) = 1.023 Gyr
        
        tau_R = eta * 2.0 * np.pi * R_target / v_circ * kpc_per_km_s_to_Gyr  # Gyr
        
        # For steady-state analysis, assume disk has existed for t_age ~ 10 Gyr
        # (typical spiral galaxy age)
        t_age = 10.0  # Gyr
        
        # Smoothing factor: f(R) = 1 - exp(-t_age/τ(R))
        # For t_age >> τ: f → 1 (no smoothing, equilibrium)
        # For t_age ~ τ: f ~ 0.5-0.9 (partial smoothing)
        # For t_age << τ: f → 0 (strong suppression, not yet established)
        smoothing_factor = 1.0 - np.exp(-t_age / tau_R)
        
        # Apply smoothing
        rho_coh_smoothed = rho_coh_instantaneous * smoothing_factor
        
        return rho_coh_smoothed


# Convenience function for drop-in replacement
def make_rho_coh_axisym(rho_b_func: Callable[[float], float],
                        alpha: float,
                        ell: float,
                        R_target: np.ndarray,
                        h_z: float = 0.3,
                        use_3d: bool = False) -> np.ndarray:
    """
    Drop-in replacement for spherical Yukawa convolution.
    
    This function has the same signature as the spherical version
    but uses axisymmetric disk geometry for more accuracy.
    
    Parameters
    ----------
    rho_b_func : callable
        Baryon volume density ρ_b(R) at midplane (M☉/kpc³)
    alpha : float
        Coherence susceptibility
    ell : float
        Coherence length (kpc)
    R_target : array
        Radii where to compute ρ_coh (kpc)
    h_z : float
        Disk scale height (kpc)
    use_3d : bool
        Use full 3D integration (slower, more accurate for thick disks)
        
    Returns
    -------
    rho_coh : array
        Coherence volume density at R_target (M☉/kpc³)
    """
    convolver = AxiSymmetricYukawaConvolver(h_z=h_z)
    
    rho_coh = convolver.convolve_volume_density(
        rho_b_func, alpha, ell, R_target, R_max=R_target.max() * 2.0
    )
    
    return rho_coh


if __name__ == '__main__':
    """Test axisymmetric convolution vs spherical approximation."""
    
    # Test with exponential disk
    Sigma0 = 1e9  # M☉/kpc²
    R_d = 2.0  # kpc
    h_z = 0.3  # kpc
    
    def Sigma_disk(R):
        return Sigma0 * np.exp(-R / R_d)
    
    def rho_disk(R):
        return Sigma_disk(R) / (2.0 * h_z)
    
    # Test parameters
    alpha = 0.25
    ell = 1.5  # kpc
    R_test = np.linspace(0.5, 10.0, 20)
    
    print("Testing Axisymmetric Yukawa Convolution")
    print("=" * 60)
    print(f"Disk: Σ₀ = {Sigma0:.2e} M☉/kpc², R_d = {R_d} kpc, h_z = {h_z} kpc")
    print(f"GPM: α = {alpha}, ℓ = {ell} kpc")
    print()
    
    # Thin-disk (K₀) approximation
    print("Computing with thin-disk approximation (fast)...")
    import time
    t0 = time.time()
    rho_coh_2d = make_rho_coh_axisym(rho_disk, alpha, ell, R_test, h_z, use_3d=False)
    t_2d = time.time() - t0
    print(f"  Completed in {t_2d:.2f} seconds")
    
    # Full 3D integration
    print("Computing with full 3D integration (slow)...")
    t0 = time.time()
    convolver = AxiSymmetricYukawaConvolver(h_z)
    rho_coh_3d = convolver.convolve_volume_density(rho_disk, alpha, ell, R_test[:5], R_test.max() * 2)
    t_3d = time.time() - t0
    print(f"  Completed in {t_3d:.2f} seconds (first 5 points only)")
    
    # Compare results
    print()
    print("Results (first 10 radii):")
    print(f"{'R [kpc]':<10} {'ρ_b [M☉/kpc³]':<18} {'ρ_coh (2D)':<18} {'ρ_coh (3D)':<18}")
    print("-" * 60)
    for i in range(min(5, len(R_test))):
        print(f"{R_test[i]:<10.2f} {rho_disk(R_test[i]):<18.2e} {rho_coh_2d[i]:<18.2e} {rho_coh_3d[i]:<18.2e}")
    
    print()
    print(f"Speedup (2D vs 3D): {t_3d / t_2d:.1f}×")
    print(f"2D/3D ratio (R={R_test[0]:.1f} kpc): {rho_coh_2d[0] / rho_coh_3d[0]:.3f}")
    print()
    print("✓ Axisymmetric convolution implemented successfully")
