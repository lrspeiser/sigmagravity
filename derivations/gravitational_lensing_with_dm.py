#!/usr/bin/env python3
"""
GRAVITATIONAL LENSING: GRAVITON MODEL vs ΛCDM DARK MATTER
=========================================================

Compare the graviton path model predictions against standard ΛCDM
dark matter halo predictions for multiple scenarios.

DARK MATTER PARAMETERS FROM LITERATURE:
- Bullet Cluster: Clowe+ 2006, Bradač+ 2006, Paraficz+ 2016
- Abell 1689: Limousin+ 2007, Coe+ 2010
- Galaxy halos: Navarro+ 1997 (NFW), Kravtsov+ 2018 (concentration-mass)
- Dwarf spheroidals: Walker+ 2009, Strigari+ 2008
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
from scipy.ndimage import gaussian_filter

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
G = 6.67430e-11       # m³/kg/s²
c = 2.99792458e8      # m/s
M_sun = 1.98892e30    # kg
kpc = 3.0856775814913673e19  # m
Mpc = 3.0856775814913673e22  # m

# MOND/Graviton model parameters
a0 = 1.2e-10          # m/s²
A_CLUSTER = 8.45      # Cluster amplitude
A_GALAXY = 1.0        # Galaxy amplitude

# Cosmological parameters (Planck 2018)
h = 0.7               # H0 = 100h km/s/Mpc
Omega_m = 0.315
Omega_b = 0.0493
rho_crit = 1.36e11    # M_sun/Mpc³ (critical density at z=0)


# =============================================================================
# DARK MATTER HALO MODELS
# =============================================================================

@dataclass
class NFWHalo:
    """
    Navarro-Frenk-White dark matter halo profile.
    
    ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
    
    Parameters from concentration-mass relation (Dutton & Macciò 2014):
    log10(c) = 0.905 - 0.101 × log10(M_200 / 10¹² h⁻¹ M☉)
    """
    M_200: float        # Virial mass in M_sun (mass within r_200)
    concentration: float  # c = r_200 / r_s
    x_center: float = 0  # kpc
    y_center: float = 0  # kpc
    
    @property
    def r_200(self) -> float:
        """Virial radius in kpc."""
        # r_200 defined where mean density = 200 × ρ_crit
        # M_200 = (4π/3) × 200 × ρ_crit × r_200³
        rho_200 = 200 * rho_crit * 1e-9  # M_sun/kpc³
        r_200_kpc = (3 * self.M_200 / (4 * np.pi * rho_200))**(1/3)
        return r_200_kpc
    
    @property
    def r_s(self) -> float:
        """Scale radius in kpc."""
        return self.r_200 / self.concentration
    
    @property
    def rho_s(self) -> float:
        """Characteristic density in M_sun/kpc³."""
        f_c = np.log(1 + self.concentration) - self.concentration / (1 + self.concentration)
        return self.M_200 / (4 * np.pi * self.r_s**3 * f_c)
    
    def density_3d(self, r_kpc: np.ndarray) -> np.ndarray:
        """3D density in M_sun/kpc³."""
        r_kpc = np.maximum(r_kpc, 1e-6)
        x = r_kpc / self.r_s
        return self.rho_s / (x * (1 + x)**2)
    
    def enclosed_mass(self, r_kpc: np.ndarray) -> np.ndarray:
        """Enclosed mass M(<r) in M_sun."""
        r_kpc = np.maximum(r_kpc, 1e-6)
        x = r_kpc / self.r_s
        f_x = np.log(1 + x) - x / (1 + x)
        f_c = np.log(1 + self.concentration) - self.concentration / (1 + self.concentration)
        return self.M_200 * f_x / f_c
    
    def surface_density_2d(self, R_kpc: np.ndarray) -> np.ndarray:
        """
        Projected surface density Σ(R) in M_sun/kpc².
        
        Analytic formula from Bartelmann 1996.
        """
        R_kpc = np.maximum(R_kpc, 1e-6)
        x = R_kpc / self.r_s
        
        # NFW surface density formula
        Sigma_s = self.rho_s * self.r_s
        
        # Different expressions for x<1, x=1, x>1
        result = np.zeros_like(x)
        
        mask_lt = x < 1
        mask_eq = np.abs(x - 1) < 0.01
        mask_gt = x > 1
        
        # x < 1
        if np.any(mask_lt):
            x_lt = x[mask_lt]
            result[mask_lt] = (1 - 2/np.sqrt(1-x_lt**2) * np.arctanh(np.sqrt((1-x_lt)/(1+x_lt)))) / (x_lt**2 - 1)
        
        # x ≈ 1
        if np.any(mask_eq):
            result[mask_eq] = 1/3
        
        # x > 1
        if np.any(mask_gt):
            x_gt = x[mask_gt]
            result[mask_gt] = (1 - 2/np.sqrt(x_gt**2-1) * np.arctan(np.sqrt((x_gt-1)/(1+x_gt)))) / (x_gt**2 - 1)
        
        return 2 * Sigma_s * result


def concentration_from_mass(M_200: float, z: float = 0) -> float:
    """
    Concentration-mass relation from Dutton & Macciò 2014.
    
    log10(c) = a + b × log10(M_200 / 10¹² h⁻¹ M☉)
    
    For z=0: a = 0.905, b = -0.101
    """
    a = 0.905 - 0.101 * z
    b = -0.101 + 0.026 * z
    log_c = a + b * np.log10(M_200 * h / 1e12)
    return 10**log_c


# =============================================================================
# PUBLISHED DARK MATTER PARAMETERS
# =============================================================================

# Literature values for dark matter halos
DM_LITERATURE = {
    'bullet_cluster': {
        'source': 'Clowe+ 2006, Bradač+ 2006, Paraficz+ 2016',
        'main_cluster': {
            'M_200': 1.5e15,      # M_sun (from weak lensing)
            'concentration': 4.0,  # Typical for massive clusters
            'x_center': -200,      # Coincident with galaxies, not gas
            'y_center': 0,
        },
        'subcluster': {
            'M_200': 0.7e15,
            'concentration': 5.0,
            'x_center': 300,       # Coincident with galaxies
            'y_center': 0,
        },
        'total_dm_mass': 2.2e15,  # Total DM within lensing aperture
        'total_lensing_mass': 5.5e14,  # Observed lensing mass
    },
    
    'abell_1689': {
        'source': 'Limousin+ 2007, Coe+ 2010, Umetsu+ 2015',
        'halo': {
            'M_200': 1.8e15,
            'concentration': 6.0,  # Higher than average - well-relaxed cluster
            'x_center': 0,
            'y_center': 0,
        },
        'total_lensing_mass_1Mpc': 1.5e15,
    },
    
    'abell_520': {
        'source': 'Jee+ 2014, Clowe+ 2012',
        'note': 'Complex merger with "dark core" - DM not following galaxies',
        'main_halo': {
            'M_200': 0.8e15,
            'concentration': 4.0,
            'x_center': 0,  # Dark core at center
            'y_center': 0,
        },
        'subhalo_1': {
            'M_200': 0.3e15,
            'concentration': 5.0,
            'x_center': -150,
            'y_center': 100,
        },
    },
    
    'milky_way': {
        'source': 'McMillan 2017, Cautun+ 2020',
        'halo': {
            'M_200': 1.3e12,       # M_sun
            'concentration': 10.0, # From rotation curve fits
            'x_center': 0,
            'y_center': 0,
        },
        'M_star': 6.5e10,
    },
    
    'fornax_dsph': {
        'source': 'Walker+ 2009, Strigari+ 2008',
        'halo': {
            'M_200': 1e9,          # Highly uncertain
            'concentration': 20,   # dSphs have high concentrations
            'x_center': 0,
            'y_center': 0,
        },
        'M_star': 2e7,
        'note': 'M/L ~ 50-100 from dynamics',
    },
    
    'ngc1052_df2': {
        'source': 'van Dokkum+ 2018',
        'note': 'Appears to have NO dark matter halo',
        'halo': {
            'M_200': 0,  # No DM!
            'concentration': 0,
            'x_center': 0,
            'y_center': 0,
        },
        'M_star': 2e8,
        'upper_limit_M_halo': 1e9,  # Upper limit from dynamics
    },
    
    'dragonfly44': {
        'source': 'van Dokkum+ 2016',
        'note': 'Appears extremely DM dominated',
        'halo': {
            'M_200': 1e12,         # Similar to MW despite 100× less stars
            'concentration': 10,
            'x_center': 0,
            'y_center': 0,
        },
        'M_star': 3e8,
    },
}


# =============================================================================
# BARYONIC MASS MODELS (same as before)
# =============================================================================

@dataclass
class BaryonicComponent:
    """A baryonic mass component."""
    name: str
    M_total: float      # M_sun
    x_center: float     # kpc
    y_center: float     # kpc
    profile: str        # 'beta', 'plummer', 'exponential'
    scale_radius: float # kpc
    component_type: str = 'gas'  # 'gas' or 'stars'
    beta: float = 0.67
    
    def surface_density_2d(self, R_kpc: np.ndarray) -> np.ndarray:
        """Projected surface density in M_sun/kpc²."""
        R_kpc = np.maximum(R_kpc, 1e-6)
        r_s = self.scale_radius
        M = self.M_total
        
        if self.profile == 'plummer':
            return M / (np.pi * r_s**2) * (1 + (R_kpc/r_s)**2)**(-2)
        elif self.profile == 'exponential':
            return M / (2 * np.pi * r_s**2) * np.exp(-R_kpc / r_s)
        elif self.profile == 'beta':
            x = R_kpc / r_s
            Sigma_0 = M / (np.pi * r_s**2 * 4)
            return Sigma_0 / (1 + x**2)**(self.beta)
        else:
            raise ValueError(f"Unknown profile: {self.profile}")
    
    def enclosed_mass(self, r_kpc: np.ndarray) -> np.ndarray:
        """Enclosed mass M(<r) in M_sun."""
        r_kpc = np.maximum(r_kpc, 1e-6)
        r_s = self.scale_radius
        M = self.M_total
        
        if self.profile == 'plummer':
            x = r_kpc / r_s
            return M * x**3 / (1 + x**2)**1.5
        elif self.profile == 'exponential':
            x = r_kpc / r_s
            return M * (1 - (1 + x + 0.5*x**2) * np.exp(-x))
        elif self.profile == 'beta':
            x = r_kpc / r_s
            return M * (1 - 1/np.sqrt(1 + x**2))
        else:
            raise ValueError(f"Unknown profile: {self.profile}")


# =============================================================================
# GRAVITON MODEL FUNCTIONS
# =============================================================================

def graviton_enhancement(g_N: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """Enhancement factor from graviton model."""
    g_N = np.atleast_1d(np.asarray(g_N, dtype=float))
    g_N = np.maximum(g_N, 1e-20)
    f_coh = a0 / (a0 + g_N)
    boost_ratio = amplitude * np.sqrt(a0 / g_N) * f_coh
    return 1 + boost_ratio


def mond_interpolation(g_N: np.ndarray) -> np.ndarray:
    """MOND interpolation function."""
    g_N = np.atleast_1d(np.asarray(g_N, dtype=float))
    g_N = np.maximum(g_N, 1e-20)
    x = g_N / a0
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))


# =============================================================================
# COMBINED SIMULATION
# =============================================================================

@dataclass
class ScenarioResult:
    """Results for a single scenario."""
    name: str
    
    # Masses
    M_baryonic: float
    M_dm_literature: float
    M_total_lcdm: float
    M_eff_graviton: float
    M_eff_mond: float
    M_observed: float
    
    # Ratios
    ratio_lcdm: float
    ratio_graviton: float
    ratio_mond: float
    ratio_observed: float
    
    # Peak locations
    peak_baryonic: Tuple[float, float]
    peak_dm: Tuple[float, float]
    peak_graviton: Tuple[float, float]
    
    # Notes
    notes: str


def run_scenario(scenario_name: str, grid_size: int = 150) -> ScenarioResult:
    """
    Run a complete comparison for a scenario.
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name.upper()}")
    print(f"{'='*70}")
    
    # Get literature DM parameters
    dm_params = DM_LITERATURE.get(scenario_name, {})
    
    # Set up scenario-specific components
    if scenario_name == 'bullet_cluster':
        # Baryonic components
        baryons = [
            BaryonicComponent("Main gas", 1.5e14, -50, 0, 'beta', 250, 'gas'),
            BaryonicComponent("Main stars", 0.3e14, -200, 0, 'plummer', 100, 'stars'),
            BaryonicComponent("Sub gas", 0.6e14, 100, 0, 'beta', 150, 'gas'),
            BaryonicComponent("Sub stars", 0.2e14, 300, 0, 'plummer', 60, 'stars'),
        ]
        
        # DM halos
        dm_halos = [
            NFWHalo(dm_params['main_cluster']['M_200'], 
                   dm_params['main_cluster']['concentration'],
                   dm_params['main_cluster']['x_center'],
                   dm_params['main_cluster']['y_center']),
            NFWHalo(dm_params['subcluster']['M_200'],
                   dm_params['subcluster']['concentration'],
                   dm_params['subcluster']['x_center'],
                   dm_params['subcluster']['y_center']),
        ]
        
        fov = 2000
        amplitude = A_CLUSTER
        M_observed = 5.5e14
        
    elif scenario_name == 'abell_1689':
        baryons = [
            BaryonicComponent("ICM gas", 1.2e14, 0, 0, 'beta', 200, 'gas'),
            BaryonicComponent("Central galaxies", 0.15e14, 0, 0, 'plummer', 50, 'stars'),
        ]
        dm_halos = [
            NFWHalo(dm_params['halo']['M_200'],
                   dm_params['halo']['concentration'], 0, 0),
        ]
        fov = 1500
        amplitude = A_CLUSTER
        M_observed = 1.5e15
        
    elif scenario_name == 'milky_way':
        baryons = [
            BaryonicComponent("Stellar disk", 5e10, 0, 0, 'exponential', 3.0, 'stars'),
            BaryonicComponent("Bulge", 1e10, 0, 0, 'plummer', 0.5, 'stars'),
            BaryonicComponent("Gas", 1e10, 0, 0, 'exponential', 6.0, 'gas'),
        ]
        dm_halos = [
            NFWHalo(dm_params['halo']['M_200'],
                   dm_params['halo']['concentration'], 0, 0),
        ]
        fov = 50
        amplitude = A_GALAXY
        M_observed = 1.3e12  # Total dynamical mass
        
    elif scenario_name == 'fornax_dsph':
        baryons = [
            BaryonicComponent("Stars", 2e7, 0, 0, 'plummer', 0.7, 'stars'),
        ]
        dm_halos = [
            NFWHalo(dm_params['halo']['M_200'],
                   dm_params['halo']['concentration'], 0, 0),
        ]
        fov = 5
        amplitude = A_GALAXY
        M_observed = 1.5e8  # Dynamical mass within r_half
        
    elif scenario_name == 'ngc1052_df2':
        baryons = [
            BaryonicComponent("Stars", 2e8, 0, 0, 'plummer', 2.2, 'stars'),
        ]
        dm_halos = []  # No DM!
        fov = 20
        amplitude = A_GALAXY
        M_observed = 2e8  # Appears to have no DM
        
    elif scenario_name == 'dragonfly44':
        baryons = [
            BaryonicComponent("Stars", 3e8, 0, 0, 'plummer', 4.6, 'stars'),
        ]
        dm_halos = [
            NFWHalo(dm_params['halo']['M_200'],
                   dm_params['halo']['concentration'], 0, 0),
        ]
        fov = 40
        amplitude = A_GALAXY
        M_observed = 1e12  # Dynamical mass
        
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    # Create grid
    x = np.linspace(-fov/2, fov/2, grid_size)
    y = np.linspace(-fov/2, fov/2, grid_size)
    X, Y = np.meshgrid(x, y)
    pixel_area = (fov/grid_size)**2
    
    # Compute surface densities
    sigma_baryonic = np.zeros_like(X)
    for comp in baryons:
        dx = X - comp.x_center
        dy = Y - comp.y_center
        R = np.sqrt(dx**2 + dy**2)
        sigma_baryonic += comp.surface_density_2d(R)
    
    sigma_dm = np.zeros_like(X)
    for halo in dm_halos:
        dx = X - halo.x_center
        dy = Y - halo.y_center
        R = np.sqrt(dx**2 + dy**2)
        sigma_dm += halo.surface_density_2d(R)
    
    # Total ΛCDM surface density
    sigma_lcdm = sigma_baryonic + sigma_dm
    
    # Compute gravitational field for graviton model
    g_map = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            g_total = 0
            for comp in baryons:
                dx = x[j] - comp.x_center
                dy = y[i] - comp.y_center
                R_kpc = np.sqrt(dx**2 + dy**2)
                R_m = R_kpc * kpc
                M_enc = comp.enclosed_mass(R_kpc)
                if R_m > 1e10:
                    g_total += G * M_enc * M_sun / R_m**2
            g_map[i, j] = g_total
    
    # Graviton effective surface density
    enhancement = graviton_enhancement(g_map, amplitude)
    sigma_graviton = sigma_baryonic * enhancement
    
    # MOND effective surface density
    nu_mond = mond_interpolation(g_map)
    sigma_mond = sigma_baryonic * nu_mond
    
    # Compute total masses
    M_bar = np.sum(sigma_baryonic) * pixel_area
    M_dm = np.sum(sigma_dm) * pixel_area
    M_lcdm = np.sum(sigma_lcdm) * pixel_area
    M_grav = np.sum(sigma_graviton) * pixel_area
    M_mond = np.sum(sigma_mond) * pixel_area
    
    # Find peaks
    def find_peak(sigma_map):
        smoothed = gaussian_filter(sigma_map, sigma=2)
        idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        return (float(x[idx[1]]), float(y[idx[0]]))
    
    peak_bar = find_peak(sigma_baryonic)
    peak_dm = find_peak(sigma_dm) if dm_halos else (0, 0)
    peak_grav = find_peak(sigma_graviton)
    
    # Print results
    print(f"\nMass Comparison:")
    print(f"  Baryonic:         {M_bar:.2e} M☉")
    print(f"  Dark Matter:      {M_dm:.2e} M☉")
    print(f"  Total (ΛCDM):     {M_lcdm:.2e} M☉ ({M_lcdm/M_bar:.2f}× baryonic)")
    print(f"  Graviton eff:     {M_grav:.2e} M☉ ({M_grav/M_bar:.2f}× baryonic)")
    print(f"  MOND eff:         {M_mond:.2e} M☉ ({M_mond/M_bar:.2f}× baryonic)")
    print(f"  Observed:         {M_observed:.2e} M☉ ({M_observed/M_bar:.2f}× baryonic)")
    
    print(f"\nPeak Locations:")
    print(f"  Baryonic peak:    x={peak_bar[0]:.0f} kpc")
    print(f"  DM peak:          x={peak_dm[0]:.0f} kpc")
    print(f"  Graviton peak:    x={peak_grav[0]:.0f} kpc")
    
    # Notes
    notes = dm_params.get('note', '')
    if notes:
        print(f"\nNote: {notes}")
    
    return ScenarioResult(
        name=scenario_name,
        M_baryonic=M_bar,
        M_dm_literature=M_dm,
        M_total_lcdm=M_lcdm,
        M_eff_graviton=M_grav,
        M_eff_mond=M_mond,
        M_observed=M_observed,
        ratio_lcdm=M_lcdm/M_bar,
        ratio_graviton=M_grav/M_bar,
        ratio_mond=M_mond/M_bar,
        ratio_observed=M_observed/M_bar,
        peak_baryonic=peak_bar,
        peak_dm=peak_dm,
        peak_graviton=peak_grav,
        notes=notes
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║     GRAVITATIONAL LENSING: GRAVITON MODEL vs ΛCDM DARK MATTER                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    scenarios = ['bullet_cluster', 'abell_1689', 'milky_way', 
                 'fornax_dsph', 'ngc1052_df2', 'dragonfly44']
    
    results = {}
    for scenario in scenarios:
        try:
            results[scenario] = run_scenario(scenario)
        except Exception as e:
            print(f"Error in {scenario}: {e}")
    
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON")
    print("=" * 100)
    print(f"\n{'Scenario':<18} {'M_bar':<12} {'ΛCDM':<8} {'Graviton':<10} {'MOND':<8} {'Observed':<10} {'Winner':<10}")
    print("-" * 90)
    
    for name, r in results.items():
        # Determine which model is closest to observed
        errors = {
            'ΛCDM': abs(r.ratio_lcdm - r.ratio_observed),
            'Graviton': abs(r.ratio_graviton - r.ratio_observed),
            'MOND': abs(r.ratio_mond - r.ratio_observed),
        }
        winner = min(errors, key=errors.get)
        
        print(f"{name:<18} {r.M_baryonic:<12.1e} {r.ratio_lcdm:<8.2f} "
              f"{r.ratio_graviton:<10.2f} {r.ratio_mond:<8.2f} "
              f"{r.ratio_observed:<10.2f} {winner:<10}")
    
    # Bullet Cluster special analysis
    print("\n" + "=" * 100)
    print("BULLET CLUSTER: THE CRITICAL TEST")
    print("=" * 100)
    
    bc = results.get('bullet_cluster')
    if bc:
        print(f"""
The Bullet Cluster is the critical test because:
1. Gas dominates baryonic mass (80%)
2. But lensing peaks follow GALAXIES, not gas
3. The offset is ~150-200 kpc

Component locations:
  Main gas:    x = -50 kpc
  Main stars:  x = -200 kpc
  Sub gas:     x = 100 kpc  
  Sub stars:   x = 300 kpc

Peak locations:
  Baryonic surface density peak: x = {bc.peak_baryonic[0]:.0f} kpc (near sub-stars due to concentration)
  Dark matter peak (ΛCDM):       x = {bc.peak_dm[0]:.0f} kpc  ← Placed at galaxy locations
  Graviton effective peak:       x = {bc.peak_graviton[0]:.0f} kpc

KEY INSIGHT:
- ΛCDM: DM halos are PLACED at galaxy locations (input assumption)
- Graviton: Enhancement naturally follows concentrated mass (stars)
- MOND: Would follow total baryons (gas-dominated)

Mass ratios (comparing effective lensing mass to baryonic):
  ΛCDM:     {bc.ratio_lcdm:.2f}× 
  Graviton: {bc.ratio_graviton:.2f}× 
  MOND:     {bc.ratio_mond:.2f}×
  Observed: {bc.ratio_observed:.2f}×

Note: The "observed" ratio of 1.48× is M_lensing/M_baryonic from Clowe+ 2006.
ΛCDM predicts more mass because DM halos extend beyond the lensing aperture.
""")
    
    # DF2 special analysis
    print("\n" + "=" * 100)
    print("NGC1052-DF2: THE 'NO DARK MATTER' GALAXY")
    print("=" * 100)
    
    df2 = results.get('ngc1052_df2')
    if df2:
        print(f"""
NGC1052-DF2 appears to have NO dark matter halo!
  σ_obs = 8.5 km/s (very low for its size)
  
ΛCDM: Requires special explanation (tidal stripping? formation scenario?)
Graviton: Predicts {df2.ratio_graviton:.1f}× enhancement (σ ~ {8.5*np.sqrt(df2.ratio_graviton):.0f} km/s)
MOND: Predicts {df2.ratio_mond:.1f}× enhancement (σ ~ {8.5*np.sqrt(df2.ratio_mond):.0f} km/s)

Both Graviton and MOND OVERPREDICT the velocity dispersion!
This is a challenge for modified gravity theories.

Possible resolution: External Field Effect from NGC1052 host
""")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'model_comparison': 'Graviton vs ΛCDM vs MOND',
        'scenarios': {
            name: {
                'M_baryonic': r.M_baryonic,
                'M_dm_literature': r.M_dm_literature,
                'ratio_lcdm': r.ratio_lcdm,
                'ratio_graviton': r.ratio_graviton,
                'ratio_mond': r.ratio_mond,
                'ratio_observed': r.ratio_observed,
                'peak_baryonic': list(r.peak_baryonic),
                'peak_dm': list(r.peak_dm),
                'peak_graviton': list(r.peak_graviton),
                'notes': r.notes
            }
            for name, r in results.items()
        }
    }
    
    output_file = Path(__file__).parent / "lensing_dm_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

