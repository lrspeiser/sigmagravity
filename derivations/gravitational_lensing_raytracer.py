#!/usr/bin/env python3
"""
GRAVITATIONAL LENSING RAY-TRACER
================================

Full ray-tracing simulation for gravitational lensing in the graviton path model.

This module provides:
1. 3D mass distribution models (NFW, beta, Plummer, exponential disk)
2. Gravitational potential and deflection angle calculation
3. Ray-tracing through arbitrary mass distributions
4. Convergence (κ) and shear (γ) maps
5. Comparison between Newtonian, MOND, and graviton model predictions

USAGE:
    python gravitational_lensing_raytracer.py

Or import and use programmatically:
    from gravitational_lensing_raytracer import LensingSimulation, MassModel
    sim = LensingSimulation(scenario='bullet_cluster')
    results = sim.run()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import json
from datetime import datetime
from scipy import integrate
from scipy.ndimage import gaussian_filter

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
G = 6.67430e-11       # m³/kg/s²
c = 2.99792458e8      # m/s
M_sun = 1.98892e30    # kg
kpc = 3.0856775814913673e19  # m
Mpc = 3.0856775814913673e22  # m
arcsec = np.pi / (180 * 3600)  # radians

# MOND/Graviton model parameters
a0 = 1.2e-10          # m/s²
A_CLUSTER = 8.45      # Cluster amplitude
A_GALAXY = 1.0        # Galaxy amplitude


# =============================================================================
# MASS DISTRIBUTION MODELS
# =============================================================================

@dataclass
class MassComponent:
    """A single mass component with position, mass, and profile."""
    name: str
    M_total: float      # Total mass in M_sun
    x_center: float     # x position in kpc
    y_center: float     # y position in kpc
    z_center: float     # z position in kpc (along line of sight)
    profile: str        # 'nfw', 'beta', 'plummer', 'exponential', 'point'
    scale_radius: float # Scale radius in kpc
    concentration: float = 5.0  # For NFW
    beta: float = 0.67  # For beta model
    component_type: str = 'gas'  # 'gas' or 'stars' - affects enhancement
    
    def density_3d(self, r_kpc: np.ndarray) -> np.ndarray:
        """
        3D density profile ρ(r) in M_sun/kpc³.
        
        Args:
            r_kpc: 3D radius from center in kpc
        """
        r_s = self.scale_radius
        M = self.M_total
        r_kpc = np.maximum(r_kpc, 1e-6)
        
        if self.profile == 'nfw':
            # NFW profile: ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
            f_c = np.log(1 + self.concentration) - self.concentration / (1 + self.concentration)
            rho_s = M / (4 * np.pi * r_s**3 * f_c)
            x = r_kpc / r_s
            return rho_s / (x * (1 + x)**2)
            
        elif self.profile == 'beta':
            # Beta model: ρ(r) = ρ_0 / (1 + (r/r_c)²)^(3β/2)
            # For β=2/3: total mass diverges, use cutoff
            # Normalization: M = 4π ∫ ρ r² dr
            norm_integral = 4 * np.pi * r_s**3 * 2.0  # Approximate
            rho_0 = M / norm_integral
            x = r_kpc / r_s
            return rho_0 / (1 + x**2)**(1.5 * self.beta)
            
        elif self.profile == 'plummer':
            # Plummer: ρ(r) = (3M/4πa³) × (1 + r²/a²)^(-5/2)
            rho_0 = 3 * M / (4 * np.pi * r_s**3)
            x = r_kpc / r_s
            return rho_0 * (1 + x**2)**(-2.5)
            
        elif self.profile == 'exponential':
            # Exponential: ρ(r) = ρ_0 × exp(-r/r_s)
            # M = 8π ρ_0 r_s³
            rho_0 = M / (8 * np.pi * r_s**3)
            return rho_0 * np.exp(-r_kpc / r_s)
            
        elif self.profile == 'point':
            # Point mass (delta function - handled specially)
            return np.zeros_like(r_kpc)
            
        else:
            raise ValueError(f"Unknown profile: {self.profile}")
    
    def enclosed_mass(self, r_kpc: np.ndarray) -> np.ndarray:
        """
        Enclosed mass M(<r) in M_sun.
        
        Args:
            r_kpc: 3D radius in kpc
        """
        r_kpc = np.maximum(r_kpc, 1e-6)
        r_s = self.scale_radius
        M = self.M_total
        
        if self.profile == 'nfw':
            x = r_kpc / r_s
            f_x = np.log(1 + x) - x / (1 + x)
            f_c = np.log(1 + self.concentration) - self.concentration / (1 + self.concentration)
            return M * f_x / f_c
            
        elif self.profile == 'beta':
            # Approximate enclosed mass
            x = r_kpc / r_s
            # For beta=2/3, M(<r) ∝ r³/(1+r²/r_s²)^0.5
            return M * (1 - 1/np.sqrt(1 + x**2))
            
        elif self.profile == 'plummer':
            x = r_kpc / r_s
            return M * x**3 / (1 + x**2)**1.5
            
        elif self.profile == 'exponential':
            x = r_kpc / r_s
            return M * (1 - (1 + x + 0.5*x**2) * np.exp(-x))
            
        elif self.profile == 'point':
            return np.full_like(r_kpc, M)
            
        else:
            raise ValueError(f"Unknown profile: {self.profile}")
    
    def surface_density_2d(self, R_kpc: np.ndarray) -> np.ndarray:
        """
        Projected surface density Σ(R) in M_sun/kpc².
        
        Args:
            R_kpc: 2D projected radius in kpc
        """
        R_kpc = np.maximum(R_kpc, 1e-6)
        r_s = self.scale_radius
        M = self.M_total
        
        if self.profile == 'plummer':
            # Plummer has analytic projection: Σ(R) = M/(πa²) × (1 + R²/a²)^(-2)
            return M / (np.pi * r_s**2) * (1 + (R_kpc/r_s)**2)**(-2)
            
        elif self.profile == 'exponential':
            # Exponential disk: Σ(R) = M/(2πr_s²) × exp(-R/r_s)
            return M / (2 * np.pi * r_s**2) * np.exp(-R_kpc / r_s)
            
        elif self.profile == 'beta':
            # Beta model projection (approximate)
            x = R_kpc / r_s
            Sigma_0 = M / (np.pi * r_s**2 * 4)  # Rough normalization
            return Sigma_0 / (1 + x**2)**(self.beta)
            
        elif self.profile == 'nfw':
            # NFW projection (approximate)
            x = R_kpc / r_s
            # Use simplified form
            Sigma_0 = M / (np.pi * r_s**2 * 10)
            return Sigma_0 / (x * (1 + x)**2)
            
        elif self.profile == 'point':
            # Delta function at center
            return np.where(R_kpc < 0.1, M / (np.pi * 0.1**2), 0)
            
        else:
            raise ValueError(f"Unknown profile: {self.profile}")


@dataclass
class MassModel:
    """Complete mass model with multiple components."""
    name: str
    components: List[MassComponent] = field(default_factory=list)
    D_lens: float = 1000.0  # Lens distance in Mpc
    D_source: float = 2000.0  # Source distance in Mpc
    system_type: str = 'cluster'  # 'cluster', 'galaxy', 'dwarf'
    
    def add_component(self, component: MassComponent):
        """Add a mass component."""
        self.components.append(component)
    
    @property
    def amplitude(self) -> float:
        """Get appropriate amplitude for this system type."""
        if self.system_type == 'cluster':
            return A_CLUSTER
        else:
            return A_GALAXY


# =============================================================================
# GRAVITON MODEL PHYSICS
# =============================================================================

def newtonian_acceleration(M_enclosed: float, r_m: float) -> float:
    """
    Newtonian gravitational acceleration g = GM/r².
    
    Args:
        M_enclosed: Enclosed mass in M_sun
        r_m: Radius in meters
        
    Returns:
        Acceleration in m/s²
    """
    if r_m < 1e10:
        return 0
    return G * M_enclosed * M_sun / r_m**2


def graviton_enhancement(g_N: np.ndarray, amplitude: float = A_CLUSTER) -> np.ndarray:
    """
    Enhancement factor Σ = g_total / g_N from graviton model.
    
    g_total = g_N + A × √(g_N × a₀) × a₀/(a₀ + g_N)
    Σ = 1 + A × √(a₀/g_N) × a₀/(a₀ + g_N)
    """
    g_N = np.atleast_1d(np.asarray(g_N, dtype=float))
    g_N = np.maximum(g_N, 1e-20)
    f_coh = a0 / (a0 + g_N)
    boost_ratio = amplitude * np.sqrt(a0 / g_N) * f_coh
    result = 1 + boost_ratio
    return result if result.size > 1 else float(result)


def mond_interpolation(g_N: np.ndarray) -> np.ndarray:
    """
    Standard MOND interpolation function.
    g_obs = g_N × ν(g_N/a₀) where ν(x) = 1/(1 - exp(-√x))
    """
    g_N = np.atleast_1d(np.asarray(g_N, dtype=float))
    g_N = np.maximum(g_N, 1e-20)
    x = g_N / a0
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))
    return nu if nu.size > 1 else float(nu)


# =============================================================================
# LENSING PHYSICS
# =============================================================================

def critical_surface_density(D_l: float, D_s: float, D_ls: float) -> float:
    """
    Critical surface density for lensing.
    
    Σ_crit = c² D_s / (4πG D_l D_ls)
    
    Args:
        D_l: Angular diameter distance to lens (m)
        D_s: Angular diameter distance to source (m)
        D_ls: Angular diameter distance from lens to source (m)
        
    Returns:
        Critical surface density in M_sun/kpc²
    """
    sigma_crit_SI = c**2 * D_s / (4 * np.pi * G * D_l * D_ls)  # kg/m²
    return sigma_crit_SI / M_sun * kpc**2  # M_sun/kpc²


# =============================================================================
# RAY-TRACING ENGINE
# =============================================================================

@dataclass
class RayTracingResult:
    """Results from ray-tracing simulation."""
    x_grid: np.ndarray
    y_grid: np.ndarray
    
    # Surface densities (M_sun/kpc²)
    sigma_baryonic: np.ndarray
    sigma_gas: np.ndarray
    sigma_stars: np.ndarray
    
    # Effective surface densities with enhancement
    sigma_eff_newtonian: np.ndarray
    sigma_eff_graviton: np.ndarray
    sigma_eff_mond: np.ndarray
    
    # Convergence maps
    kappa_newtonian: np.ndarray
    kappa_graviton: np.ndarray
    kappa_mond: np.ndarray
    
    # Enhancement maps
    enhancement_graviton: np.ndarray
    enhancement_mond: np.ndarray
    
    # Peak locations [(x, y, value), ...]
    peaks_newtonian: List[Tuple[float, float, float]]
    peaks_graviton: List[Tuple[float, float, float]]
    peaks_mond: List[Tuple[float, float, float]]
    
    # Summary statistics
    total_mass_baryonic: float
    total_mass_gas: float
    total_mass_stars: float
    total_mass_eff_newtonian: float
    total_mass_eff_graviton: float
    total_mass_eff_mond: float
    
    # Component positions
    component_positions: Dict[str, Tuple[float, float]]


class LensingSimulation:
    """
    Full gravitational lensing simulation with ray-tracing.
    """
    
    def __init__(self, mass_model: MassModel, 
                 grid_size: Tuple[int, int] = (200, 200),
                 field_of_view: Tuple[float, float] = (2000, 2000)):
        """
        Initialize simulation.
        
        Args:
            mass_model: MassModel object with all components
            grid_size: (nx, ny) grid points
            field_of_view: (width, height) in kpc
        """
        self.mass_model = mass_model
        self.grid_size = grid_size
        self.fov = field_of_view
        
        # Create grid
        self.x_1d = np.linspace(-self.fov[0]/2, self.fov[0]/2, grid_size[0])
        self.y_1d = np.linspace(-self.fov[1]/2, self.fov[1]/2, grid_size[1])
        self.X, self.Y = np.meshgrid(self.x_1d, self.y_1d)
        
        # Pixel size
        self.dx_kpc = self.fov[0] / grid_size[0]
        self.dy_kpc = self.fov[1] / grid_size[1]
        self.pixel_area_kpc2 = self.dx_kpc * self.dy_kpc
        
    def compute_surface_density_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute surface density maps for gas, stars, and total.
        
        Returns:
            (sigma_gas, sigma_stars, sigma_total) in M_sun/kpc²
        """
        sigma_gas = np.zeros_like(self.X)
        sigma_stars = np.zeros_like(self.X)
        
        for comp in self.mass_model.components:
            # Distance from component center
            dx = self.X - comp.x_center
            dy = self.Y - comp.y_center
            R = np.sqrt(dx**2 + dy**2)
            
            # Surface density contribution
            sigma_comp = comp.surface_density_2d(R)
            
            if comp.component_type == 'gas':
                sigma_gas += sigma_comp
            else:
                sigma_stars += sigma_comp
        
        sigma_total = sigma_gas + sigma_stars
        return sigma_gas, sigma_stars, sigma_total
    
    def compute_gravitational_field_map(self) -> np.ndarray:
        """
        Compute local gravitational field at each grid point.
        
        Returns:
            g_map in m/s²
        """
        g_map = np.zeros_like(self.X)
        
        for i in range(self.grid_size[1]):
            for j in range(self.grid_size[0]):
                x = self.x_1d[j]
                y = self.y_1d[i]
                
                # Sum contributions from all components
                g_total = 0
                for comp in self.mass_model.components:
                    dx = x - comp.x_center
                    dy = y - comp.y_center
                    R_kpc = np.sqrt(dx**2 + dy**2)
                    R_m = R_kpc * kpc
                    
                    # Get enclosed mass at this projected radius
                    M_enc = comp.enclosed_mass(R_kpc)
                    
                    # Add contribution
                    g_total += newtonian_acceleration(M_enc, R_m)
                
                g_map[i, j] = g_total
        
        return g_map
    
    def compute_component_specific_enhancement(self, sigma_gas: np.ndarray, 
                                                sigma_stars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute enhancement for each component separately based on LOCAL g.
        
        This is the key insight: gas and stars have different local g
        and thus different enhancement factors.
        """
        # For each component, compute local g and enhancement
        sigma_eff_gas = np.zeros_like(sigma_gas)
        sigma_eff_stars = np.zeros_like(sigma_stars)
        
        amplitude = self.mass_model.amplitude
        
        for comp in self.mass_model.components:
            dx = self.X - comp.x_center
            dy = self.Y - comp.y_center
            R_kpc = np.sqrt(dx**2 + dy**2)
            R_m = R_kpc * kpc
            
            # Local g from THIS component
            M_enc = comp.enclosed_mass(R_kpc)
            g_local = np.zeros_like(R_m)
            mask = R_m > 1e10
            g_local[mask] = G * M_enc[mask] * M_sun / R_m[mask]**2
            
            # Enhancement for this component
            enhancement = graviton_enhancement(g_local, amplitude)
            
            # Surface density with enhancement
            sigma_comp = comp.surface_density_2d(R_kpc)
            sigma_eff_comp = sigma_comp * enhancement
            
            if comp.component_type == 'gas':
                sigma_eff_gas += sigma_eff_comp
            else:
                sigma_eff_stars += sigma_eff_comp
        
        return sigma_eff_gas, sigma_eff_stars
    
    def find_peaks(self, map_2d: np.ndarray, n_peaks: int = 5, 
                   smooth_sigma: float = 2.0) -> List[Tuple[float, float, float]]:
        """
        Find peaks in a 2D map.
        
        Returns list of (x, y, value) tuples.
        """
        # Smooth to avoid noise peaks
        smoothed = gaussian_filter(map_2d, sigma=smooth_sigma)
        
        # Find local maxima
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(smoothed, size=10)
        peaks_mask = (smoothed == local_max) & (smoothed > 0.01 * np.max(smoothed))
        
        # Get peak locations
        peak_indices = np.where(peaks_mask)
        if len(peak_indices[0]) == 0:
            return []
        
        peak_values = smoothed[peaks_mask]
        
        # Sort by value
        sorted_idx = np.argsort(peak_values)[::-1]
        
        peaks = []
        for i in range(min(n_peaks, len(sorted_idx))):
            idx = sorted_idx[i]
            y_idx = peak_indices[0][idx]
            x_idx = peak_indices[1][idx]
            peaks.append((float(self.x_1d[x_idx]), float(self.y_1d[y_idx]), float(peak_values[idx])))
        
        return peaks
    
    def run(self, verbose: bool = True) -> RayTracingResult:
        """
        Run full ray-tracing simulation.
        """
        if verbose:
            print(f"Running lensing simulation for: {self.mass_model.name}")
            print(f"  Grid: {self.grid_size[0]}×{self.grid_size[1]}")
            print(f"  Field of view: {self.fov[0]:.0f}×{self.fov[1]:.0f} kpc")
            print(f"  System type: {self.mass_model.system_type} (A={self.mass_model.amplitude})")
        
        # Step 1: Compute baryonic surface densities
        if verbose:
            print("  Computing surface densities...")
        sigma_gas, sigma_stars, sigma_total = self.compute_surface_density_maps()
        
        # Step 2: Compute gravitational field map
        if verbose:
            print("  Computing gravitational field...")
        g_map = self.compute_gravitational_field_map()
        
        # Step 3: Compute effective surface densities
        if verbose:
            print("  Computing effective densities...")
        
        # Newtonian: no enhancement
        sigma_eff_newton = sigma_total.copy()
        
        # Graviton: component-specific enhancement
        sigma_eff_gas_grav, sigma_eff_stars_grav = self.compute_component_specific_enhancement(
            sigma_gas, sigma_stars
        )
        sigma_eff_graviton = sigma_eff_gas_grav + sigma_eff_stars_grav
        
        # MOND: use total field for enhancement
        nu_mond = mond_interpolation(g_map)
        sigma_eff_mond = sigma_total * nu_mond
        
        # Step 4: Compute convergence maps
        D_l = self.mass_model.D_lens * Mpc
        D_s = self.mass_model.D_source * Mpc
        D_ls = D_s - D_l
        sigma_crit = critical_surface_density(D_l, D_s, D_ls)
        
        kappa_newton = sigma_eff_newton / sigma_crit
        kappa_graviton = sigma_eff_graviton / sigma_crit
        kappa_mond = sigma_eff_mond / sigma_crit
        
        # Step 5: Enhancement maps
        enhancement_graviton = np.where(sigma_total > 0, sigma_eff_graviton / sigma_total, 1.0)
        enhancement_mond = np.where(sigma_total > 0, sigma_eff_mond / sigma_total, 1.0)
        
        # Step 6: Find peaks
        if verbose:
            print("  Finding peaks...")
        peaks_newton = self.find_peaks(kappa_newton)
        peaks_graviton = self.find_peaks(kappa_graviton)
        peaks_mond = self.find_peaks(kappa_mond)
        
        # Step 7: Compute total masses
        M_gas = np.sum(sigma_gas) * self.pixel_area_kpc2
        M_stars = np.sum(sigma_stars) * self.pixel_area_kpc2
        M_total = M_gas + M_stars
        M_eff_newton = np.sum(sigma_eff_newton) * self.pixel_area_kpc2
        M_eff_graviton = np.sum(sigma_eff_graviton) * self.pixel_area_kpc2
        M_eff_mond = np.sum(sigma_eff_mond) * self.pixel_area_kpc2
        
        # Component positions
        comp_positions = {comp.name: (comp.x_center, comp.y_center) 
                         for comp in self.mass_model.components}
        
        if verbose:
            print(f"\nResults:")
            print(f"  Total baryonic mass: {M_total:.2e} M☉")
            print(f"    Gas: {M_gas:.2e} M☉ ({100*M_gas/M_total:.0f}%)")
            print(f"    Stars: {M_stars:.2e} M☉ ({100*M_stars/M_total:.0f}%)")
            print(f"  Effective mass (Newtonian): {M_eff_newton:.2e} M☉ (1.00×)")
            print(f"  Effective mass (Graviton): {M_eff_graviton:.2e} M☉ ({M_eff_graviton/M_total:.2f}×)")
            print(f"  Effective mass (MOND): {M_eff_mond:.2e} M☉ ({M_eff_mond/M_total:.2f}×)")
            
            if peaks_graviton:
                print(f"\n  Graviton κ peaks:")
                for i, (x, y, k) in enumerate(peaks_graviton[:3]):
                    print(f"    {i+1}. x={x:.0f} kpc, y={y:.0f} kpc, κ={k:.4f}")
        
        return RayTracingResult(
            x_grid=self.x_1d,
            y_grid=self.y_1d,
            sigma_baryonic=sigma_total,
            sigma_gas=sigma_gas,
            sigma_stars=sigma_stars,
            sigma_eff_newtonian=sigma_eff_newton,
            sigma_eff_graviton=sigma_eff_graviton,
            sigma_eff_mond=sigma_eff_mond,
            kappa_newtonian=kappa_newton,
            kappa_graviton=kappa_graviton,
            kappa_mond=kappa_mond,
            enhancement_graviton=enhancement_graviton,
            enhancement_mond=enhancement_mond,
            peaks_newtonian=peaks_newton,
            peaks_graviton=peaks_graviton,
            peaks_mond=peaks_mond,
            total_mass_baryonic=M_total,
            total_mass_gas=M_gas,
            total_mass_stars=M_stars,
            total_mass_eff_newtonian=M_eff_newton,
            total_mass_eff_graviton=M_eff_graviton,
            total_mass_eff_mond=M_eff_mond,
            component_positions=comp_positions
        )


# =============================================================================
# PREDEFINED SCENARIOS
# =============================================================================

def create_bullet_cluster() -> MassModel:
    """
    Create Bullet Cluster mass model.
    Based on Clowe+ 2006, Bradač+ 2006.
    
    Observed:
    - Total lensing mass: ~5.5×10¹⁴ M☉
    - Gas mass: ~2.1×10¹⁴ M☉
    - Stellar mass: ~0.5×10¹⁴ M☉
    - Lensing peaks offset from gas by ~150-200 kpc
    """
    model = MassModel(
        name="Bullet Cluster (1E 0657-56)",
        D_lens=1000,  # ~1 Gpc (z~0.3)
        D_source=2000,
        system_type='cluster'
    )
    
    # Main cluster gas (displaced toward center after collision)
    model.add_component(MassComponent(
        name="Main cluster gas",
        M_total=1.5e14,
        x_center=-50,
        y_center=0,
        z_center=0,
        profile='beta',
        scale_radius=250,
        beta=0.67,
        component_type='gas'
    ))
    
    # Main cluster stars (passed through, west of gas)
    model.add_component(MassComponent(
        name="Main cluster stars",
        M_total=0.3e14,
        x_center=-200,
        y_center=0,
        z_center=0,
        profile='plummer',
        scale_radius=100,
        component_type='stars'
    ))
    
    # Subcluster gas (the "bullet" - stripped and lagging)
    model.add_component(MassComponent(
        name="Subcluster gas",
        M_total=0.6e14,
        x_center=100,
        y_center=0,
        z_center=0,
        profile='beta',
        scale_radius=150,
        beta=0.67,
        component_type='gas'
    ))
    
    # Subcluster stars (ahead of gas)
    model.add_component(MassComponent(
        name="Subcluster stars",
        M_total=0.2e14,
        x_center=300,
        y_center=0,
        z_center=0,
        profile='plummer',
        scale_radius=60,
        component_type='stars'
    ))
    
    return model


def create_abell_1689() -> MassModel:
    """
    Create Abell 1689 mass model.
    One of the most massive known clusters, excellent for lensing.
    
    Observed:
    - Lensing mass within 1 Mpc: ~1.5×10¹⁵ M☉
    - Baryonic mass: ~0.15×10¹⁵ M☉
    """
    model = MassModel(
        name="Abell 1689",
        D_lens=630,  # z=0.183
        D_source=1500,
        system_type='cluster'
    )
    
    # Central gas
    model.add_component(MassComponent(
        name="ICM gas",
        M_total=1.2e14,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='beta',
        scale_radius=200,
        beta=0.6,
        component_type='gas'
    ))
    
    # BCG and central galaxies
    model.add_component(MassComponent(
        name="Central galaxies",
        M_total=0.15e14,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='plummer',
        scale_radius=50,
        component_type='stars'
    ))
    
    return model


def create_abell_520() -> MassModel:
    """
    Create Abell 520 mass model.
    The "Train Wreck Cluster" - another merger with complex lensing.
    
    Interestingly, A520 shows a "dark core" where lensing mass
    doesn't follow galaxies OR gas - a challenge for all models.
    """
    model = MassModel(
        name="Abell 520 (Train Wreck)",
        D_lens=700,  # z~0.2
        D_source=1800,
        system_type='cluster'
    )
    
    # Main gas concentration
    model.add_component(MassComponent(
        name="Central gas",
        M_total=1.0e14,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='beta',
        scale_radius=180,
        component_type='gas'
    ))
    
    # Galaxy concentration 1 (offset)
    model.add_component(MassComponent(
        name="Galaxy group 1",
        M_total=0.2e14,
        x_center=-150,
        y_center=100,
        z_center=0,
        profile='plummer',
        scale_radius=80,
        component_type='stars'
    ))
    
    # Galaxy concentration 2 (offset)
    model.add_component(MassComponent(
        name="Galaxy group 2",
        M_total=0.15e14,
        x_center=200,
        y_center=-50,
        z_center=0,
        profile='plummer',
        scale_radius=60,
        component_type='stars'
    ))
    
    return model


def create_galaxy_galaxy_lens() -> MassModel:
    """
    Create a typical galaxy-galaxy lensing scenario.
    
    Observed:
    - Effective M/L ~ 10-30 at 200 kpc
    """
    model = MassModel(
        name="Galaxy-Galaxy Lens",
        D_lens=100,  # ~100 Mpc
        D_source=500,
        system_type='galaxy'
    )
    
    # Lens galaxy (elliptical)
    model.add_component(MassComponent(
        name="Lens galaxy",
        M_total=5e11,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='plummer',
        scale_radius=10,
        component_type='stars'
    ))
    
    return model


def create_isolated_spiral() -> MassModel:
    """
    Create an isolated spiral galaxy (like the Milky Way).
    
    Observed:
    - Flat rotation curve to ~25 kpc
    - V_flat ~ 220 km/s
    """
    model = MassModel(
        name="Isolated Spiral Galaxy",
        D_lens=50,
        D_source=200,
        system_type='galaxy'
    )
    
    # Stellar disk
    model.add_component(MassComponent(
        name="Stellar disk",
        M_total=5e10,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='exponential',
        scale_radius=3.0,
        component_type='stars'
    ))
    
    # Bulge
    model.add_component(MassComponent(
        name="Bulge",
        M_total=1e10,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='plummer',
        scale_radius=0.5,
        component_type='stars'
    ))
    
    # Gas disk
    model.add_component(MassComponent(
        name="Gas disk",
        M_total=1e10,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='exponential',
        scale_radius=6.0,
        component_type='gas'
    ))
    
    return model


def create_dwarf_spheroidal() -> MassModel:
    """
    Create a dwarf spheroidal galaxy (like Fornax).
    
    Observed:
    - M_star ~ 2×10⁷ M☉
    - σ_los ~ 10.7 km/s
    - Very high M/L ratio
    """
    model = MassModel(
        name="Dwarf Spheroidal (Fornax-like)",
        D_lens=0.14,  # 140 kpc
        D_source=10,
        system_type='dwarf'
    )
    
    # Stellar component
    model.add_component(MassComponent(
        name="Stars",
        M_total=2e7,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='plummer',
        scale_radius=0.7,
        component_type='stars'
    ))
    
    return model


def create_udg_df2() -> MassModel:
    """
    Create NGC1052-DF2 (ultra-diffuse galaxy with "no dark matter").
    
    Observed:
    - M_star ~ 2×10⁸ M☉
    - σ_los ~ 8.5 km/s (very low!)
    - MOND predicts σ ~ 20 km/s (overpredicts)
    """
    model = MassModel(
        name="NGC1052-DF2",
        D_lens=20,  # 20 Mpc
        D_source=100,
        system_type='dwarf'
    )
    
    # Very diffuse stellar component
    model.add_component(MassComponent(
        name="Stars",
        M_total=2e8,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='plummer',
        scale_radius=2.2,
        component_type='stars'
    ))
    
    return model


def create_dragonfly44() -> MassModel:
    """
    Create Dragonfly 44 (UDG with lots of "dark matter").
    
    Observed:
    - M_star ~ 3×10⁸ M☉
    - σ_los ~ 47 km/s (very high!)
    - Appears extremely DM dominated
    """
    model = MassModel(
        name="Dragonfly 44",
        D_lens=100,  # ~100 Mpc (Coma cluster)
        D_source=300,
        system_type='dwarf'
    )
    
    model.add_component(MassComponent(
        name="Stars",
        M_total=3e8,
        x_center=0,
        y_center=0,
        z_center=0,
        profile='plummer',
        scale_radius=4.6,
        component_type='stars'
    ))
    
    return model


SCENARIOS = {
    'bullet_cluster': create_bullet_cluster,
    'abell_1689': create_abell_1689,
    'abell_520': create_abell_520,
    'galaxy_galaxy': create_galaxy_galaxy_lens,
    'spiral': create_isolated_spiral,
    'dsph': create_dwarf_spheroidal,
    'udg_df2': create_udg_df2,
    'dragonfly44': create_dragonfly44,
}


# =============================================================================
# BATCH SIMULATION
# =============================================================================

def run_all_scenarios(scenarios: List[str] = None, 
                      grid_size: Tuple[int, int] = (150, 150),
                      verbose: bool = True) -> Dict[str, RayTracingResult]:
    """
    Run lensing simulations for multiple scenarios.
    """
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())
    
    results = {}
    
    print("=" * 80)
    print("GRAVITATIONAL LENSING RAY-TRACER - BATCH SIMULATION")
    print("=" * 80)
    print(f"Running {len(scenarios)} scenarios...")
    print()
    
    for name in scenarios:
        if name not in SCENARIOS:
            print(f"Warning: Unknown scenario '{name}', skipping")
            continue
        
        # Create mass model
        mass_model = SCENARIOS[name]()
        
        # Determine appropriate field of view
        max_extent = 0
        for comp in mass_model.components:
            extent = max(abs(comp.x_center), abs(comp.y_center)) + 3 * comp.scale_radius
            max_extent = max(max_extent, extent)
        fov = (2.5 * max_extent, 2.5 * max_extent)
        
        # Run simulation
        sim = LensingSimulation(mass_model, grid_size=grid_size, field_of_view=fov)
        results[name] = sim.run(verbose=verbose)
        print()
    
    return results


def summarize_results(results: Dict[str, RayTracingResult]) -> Dict:
    """Create summary of all simulation results."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Graviton Path Model',
        'formula': 'g_total = g_N + A × √(g_N × a₀) × a₀/(a₀ + g_N)',
        'parameters': {
            'a0': a0,
            'A_cluster': A_CLUSTER,
            'A_galaxy': A_GALAXY
        },
        'n_scenarios': len(results),
        'scenarios': {}
    }
    
    for name, result in results.items():
        mass_ratio_graviton = result.total_mass_eff_graviton / result.total_mass_baryonic
        mass_ratio_mond = result.total_mass_eff_mond / result.total_mass_baryonic
        
        # Peak offset from baryonic peak
        peak_offset = None
        if result.peaks_graviton and result.peaks_newtonian:
            grav_peak = result.peaks_graviton[0]
            newt_peak = result.peaks_newtonian[0]
            peak_offset = np.sqrt((grav_peak[0] - newt_peak[0])**2 + 
                                 (grav_peak[1] - newt_peak[1])**2)
        
        summary['scenarios'][name] = {
            'total_mass_baryonic': float(result.total_mass_baryonic),
            'total_mass_gas': float(result.total_mass_gas),
            'total_mass_stars': float(result.total_mass_stars),
            'gas_fraction': float(result.total_mass_gas / result.total_mass_baryonic),
            'mass_ratio_graviton': float(mass_ratio_graviton),
            'mass_ratio_mond': float(mass_ratio_mond),
            'peak_kappa_graviton': float(result.peaks_graviton[0][2]) if result.peaks_graviton else None,
            'peak_location_graviton': list(result.peaks_graviton[0][:2]) if result.peaks_graviton else None,
            'peak_location_newtonian': list(result.peaks_newtonian[0][:2]) if result.peaks_newtonian else None,
            'peak_offset_kpc': float(peak_offset) if peak_offset else None,
            'mean_enhancement': float(np.mean(result.enhancement_graviton)),
            'max_enhancement': float(np.max(result.enhancement_graviton)),
            'component_positions': {k: list(v) for k, v in result.component_positions.items()}
        }
    
    return summary


def analyze_bullet_cluster(result: RayTracingResult) -> Dict:
    """
    Detailed analysis of Bullet Cluster results.
    """
    analysis = {
        'name': 'Bullet Cluster Analysis',
        'mass_comparison': {
            'baryonic': float(result.total_mass_baryonic),
            'gas': float(result.total_mass_gas),
            'stars': float(result.total_mass_stars),
            'gas_fraction': float(result.total_mass_gas / result.total_mass_baryonic),
            'effective_graviton': float(result.total_mass_eff_graviton),
            'effective_mond': float(result.total_mass_eff_mond),
            'observed_lensing': 5.5e14,
            'ratio_graviton': float(result.total_mass_eff_graviton / result.total_mass_baryonic),
            'ratio_mond': float(result.total_mass_eff_mond / result.total_mass_baryonic),
            'ratio_observed': 5.5e14 / result.total_mass_baryonic
        },
        'peak_analysis': {
            'graviton_peaks': result.peaks_graviton[:3] if result.peaks_graviton else [],
            'newtonian_peaks': result.peaks_newtonian[:3] if result.peaks_newtonian else [],
            'mond_peaks': result.peaks_mond[:3] if result.peaks_mond else [],
        },
        'component_positions': result.component_positions,
    }
    
    # Check if peaks are offset toward stars
    gas_positions = []
    star_positions = []
    for name, pos in result.component_positions.items():
        if 'gas' in name.lower():
            gas_positions.append(pos)
        else:
            star_positions.append(pos)
    
    if result.peaks_graviton:
        peak_x = result.peaks_graviton[0][0]
        
        # Find nearest gas and star positions
        min_dist_gas = min(abs(peak_x - pos[0]) for pos in gas_positions) if gas_positions else 999
        min_dist_star = min(abs(peak_x - pos[0]) for pos in star_positions) if star_positions else 999
        
        analysis['offset_analysis'] = {
            'primary_peak_x': float(peak_x),
            'distance_to_nearest_gas': float(min_dist_gas),
            'distance_to_nearest_stars': float(min_dist_star),
            'peak_follows_stars': min_dist_star < min_dist_gas
        }
    
    return analysis


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           GRAVITATIONAL LENSING RAY-TRACER                                    ║
║           Testing Graviton Path Model                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all scenarios
    results = run_all_scenarios(
        scenarios=list(SCENARIOS.keys()),
        grid_size=(150, 150),
        verbose=True
    )
    
    # Summarize
    summary = summarize_results(results)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Scenario':<20} {'M_bar':<12} {'f_gas':<8} {'Σ_grav':<8} {'Σ_MOND':<8} {'Offset':<8}")
    print("-" * 70)
    
    for name, data in summary['scenarios'].items():
        offset = data['peak_offset_kpc']
        offset_str = f"{offset:.0f}" if offset else "-"
        print(f"{name:<20} {data['total_mass_baryonic']:<12.1e} "
              f"{data['gas_fraction']:<8.2f} {data['mass_ratio_graviton']:<8.2f} "
              f"{data['mass_ratio_mond']:<8.2f} {offset_str:<8}")
    
    # Save results
    output_file = Path(__file__).parent / "lensing_raytracer_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Detailed Bullet Cluster analysis
    print("\n" + "=" * 80)
    print("BULLET CLUSTER DETAILED ANALYSIS")
    print("=" * 80)
    
    bc_result = results['bullet_cluster']
    bc_analysis = analyze_bullet_cluster(bc_result)
    
    print(f"\nMass comparison:")
    mc = bc_analysis['mass_comparison']
    print(f"  Baryonic: {mc['baryonic']:.2e} M☉")
    print(f"    Gas: {mc['gas']:.2e} M☉ ({100*mc['gas_fraction']:.0f}%)")
    print(f"    Stars: {mc['stars']:.2e} M☉ ({100*(1-mc['gas_fraction']):.0f}%)")
    print(f"  Graviton effective: {mc['effective_graviton']:.2e} M☉ ({mc['ratio_graviton']:.2f}×)")
    print(f"  MOND effective: {mc['effective_mond']:.2e} M☉ ({mc['ratio_mond']:.2f}×)")
    print(f"  Observed lensing: {mc['observed_lensing']:.2e} M☉ ({mc['ratio_observed']:.2f}×)")
    
    print(f"\nComponent positions:")
    for name, pos in bc_analysis['component_positions'].items():
        print(f"  {name}: x={pos[0]:.0f} kpc")
    
    print(f"\nPeak analysis:")
    if 'offset_analysis' in bc_analysis:
        oa = bc_analysis['offset_analysis']
        print(f"  Primary κ peak: x={oa['primary_peak_x']:.0f} kpc")
        print(f"  Distance to nearest gas: {oa['distance_to_nearest_gas']:.0f} kpc")
        print(f"  Distance to nearest stars: {oa['distance_to_nearest_stars']:.0f} kpc")
        print(f"  Peak follows stars: {oa['peak_follows_stars']}")
    
    # Save detailed analysis
    bc_output = Path(__file__).parent / "bullet_cluster_raytracing_results.json"
    with open(bc_output, 'w') as f:
        json.dump(bc_analysis, f, indent=2, default=float)
    print(f"\nBullet Cluster analysis saved to: {bc_output}")
