"""
OBSERVATIONAL BENCHMARKS: GOLD STANDARD DATA
=============================================

This file documents the definitive observational constraints that any
gravity theory must satisfy. All values are from peer-reviewed literature.

This serves as the SINGLE SOURCE OF TRUTH for observational comparisons.
"""

import numpy as np

# =============================================================================
# PHYSICAL CONSTANTS (CODATA 2018)
# =============================================================================
CONSTANTS = {
    'c': 2.99792458e8,       # Speed of light [m/s] (exact)
    'G': 6.67430e-11,        # Gravitational constant [m³/kg/s²] (±0.00015)
    'hbar': 1.054571817e-34, # Reduced Planck [J·s]
    'M_sun': 1.98892e30,     # Solar mass [kg]
    'AU': 1.495978707e11,    # Astronomical unit [m] (exact)
    'pc': 3.0856775814913673e16,  # Parsec [m]
    'kpc': 3.0856775814913673e19, # Kiloparsec [m]
    'H0': 70.0,              # Hubble constant [km/s/Mpc] (±2)
    'H0_SI': 2.268e-18,      # Hubble constant [1/s]
}

# =============================================================================
# MOND/MODIFIED GRAVITY SCALE
# =============================================================================
MOND_SCALE = {
    'a0': 1.2e-10,           # MOND acceleration [m/s²]
    'a0_uncertainty': 0.2e-10,  # ±0.2×10⁻¹⁰
    'source': 'McGaugh 2016, Lelli+ 2017',
    'note': 'Empirical from RAR fits; a0 ≈ cH0/(2π) suggests cosmological origin'
}

# =============================================================================
# SOLAR SYSTEM CONSTRAINTS
# =============================================================================
SOLAR_SYSTEM = {
    'cassini_ppn_gamma': {
        'value': 1.0,
        'uncertainty': 2.3e-5,
        'source': 'Bertotti+ 2003, Nature 425, 374',
        'note': 'γ-1 = (2.1±2.3)×10⁻⁵ from Cassini radio tracking'
    },
    'lunar_laser_ranging': {
        'nordtvedt_eta': 0.0,
        'uncertainty': 4.4e-4,
        'source': 'Williams+ 2004, PRL 93, 261101',
        'note': 'Tests Strong Equivalence Principle'
    },
    'perihelion_precession': {
        'mercury_excess_arcsec_century': 42.98,
        'uncertainty': 0.04,
        'GR_prediction': 42.98,
        'source': 'Park+ 2017, AJ 153, 121'
    },
    'planetary_ephemerides': {
        'extra_acceleration_bound': 1e-15,  # m/s² at Saturn
        'source': 'Pitjeva & Pitjev 2013',
        'note': 'No anomalous acceleration detected'
    }
}

# =============================================================================
# SPARC GALAXY ROTATION CURVES
# =============================================================================
SPARC = {
    'source': 'Lelli, McGaugh & Schombert 2016, AJ 152, 157',
    'url': 'http://astroweb.cwru.edu/SPARC/',
    'n_galaxies': 175,
    'n_quality': 171,  # After quality cuts
    'quality_cuts': '≥5 points, valid V_bar at all radii',
    
    'radial_acceleration_relation': {
        'formula': 'g_obs = g_bar × ν(g_bar/a0)',
        'scatter': 0.13,  # dex
        'scatter_intrinsic': 0.057,  # dex (after removing observational errors)
        'source': 'McGaugh+ 2016, PRL 117, 201101'
    },
    
    'mass_to_light': {
        'disk_3.6um': 0.5,   # M☉/L☉
        'bulge_3.6um': 0.7,  # M☉/L☉
        'uncertainty': 0.1,
        'source': 'Lelli+ 2016'
    },
    
    'mond_rms': 17.15,  # km/s (with standard a0)
    'mond_win_rate': 0.53,  # vs NFW fits
}

# =============================================================================
# GALAXY CLUSTERS
# =============================================================================
CLUSTERS = {
    'fox2022': {
        'source': 'Fox+ 2022, ApJ 928, 87',
        'n_total': 94,
        'n_quality': 42,  # spec_z + M500 > 2×10¹⁴
        'quality_cuts': 'Spectroscopic z, M500 > 2×10¹⁴ M☉',
        
        'baryon_fraction': {
            'cosmic': 0.157,  # Ω_b/Ω_m from Planck
            'cluster_gas': 0.12,  # Gas fraction in clusters
            'cluster_stars': 0.02,
            'source': 'Planck 2018'
        },
        
        'mond_mass_discrepancy': 3.0,  # Factor by which MOND underpredicts
        'lcdm_success': True,  # NFW fits work well
    },
    
    'bullet_cluster': {
        'name': '1E 0657-56',
        'source': 'Clowe+ 2006, ApJ 648, L109',
        
        'gas_mass': 2.1e14,  # M☉ (from X-ray)
        'stellar_mass': 0.5e14,  # M☉
        'total_lensing_mass': 5.5e14,  # M☉ (from weak lensing)
        
        'key_observation': 'Lensing peaks offset from gas, coincident with galaxies',
        'mond_challenge': 'Gas dominates baryons but lensing follows stars',
        
        'separation_kpc': 720,  # Between main cluster and subcluster
        'collision_velocity_kms': 4700,  # Inferred from shock
    },
    
    'abell_1689': {
        'source': 'Limousin+ 2007',
        'M_lensing_1Mpc': 1.5e15,  # M☉
        'M_baryonic_1Mpc': 0.15e15,  # M☉
        'mass_discrepancy': 10,
    }
}

# =============================================================================
# MILKY WAY
# =============================================================================
MILKY_WAY = {
    'sun_position': {
        'R_gal_kpc': 8.178,  # ±0.013 (stat) ±0.022 (sys)
        'source': 'GRAVITY Collaboration 2019'
    },
    
    'rotation_curve': {
        'V_sun_kms': 220,  # ±10 (older estimates)
        'V_sun_kms_new': 233,  # ±3 (Eilers+ 2019)
        'source': 'Eilers+ 2019, ApJ 871, 120',
        
        'flat_to_kpc': 25,  # Rotation curve flat to ~25 kpc
        'V_flat_kms': 220,  # ±20
    },
    
    'baryonic_mass': {
        'stellar_disk': 4.6e10,  # M☉ (McMillan 2017)
        'stellar_bulge': 0.9e10,
        'gas': 1.0e10,
        'total': 6.5e10,
        'uncertainty_factor': 1.3,  # Could be 30% higher or lower
        'source': 'McMillan 2017, MNRAS 465, 76'
    },
    
    'escape_velocity': {
        'at_sun': 550,  # km/s (±50)
        'source': 'Piffl+ 2014'
    }
}

# =============================================================================
# WIDE BINARIES
# =============================================================================
WIDE_BINARIES = {
    'chae_2023': {
        'source': 'Chae 2023, ApJ 952, 128',
        'finding': 'Velocity boost ~30-40% at separations > 2000 AU',
        'n_pairs': 26500,  # Gaia DR3
        
        'acceleration_threshold': 1e-10,  # m/s² where effect appears
        'boost_factor': 1.35,  # Average at wide separations
        'uncertainty': 0.1,
        
        'controversy': 'Banik+ 2024 disputes; ongoing debate'
    },
    
    'hernandez_2023': {
        'source': 'Hernandez+ 2023',
        'finding': 'Similar to Chae - excess velocity at wide separations',
    },
    
    'pittordis_2023': {
        'source': 'Pittordis & Sutherland 2023',
        'finding': 'No significant deviation from Newton',
        'note': 'Different sample selection'
    }
}

# =============================================================================
# DWARF SPHEROIDAL GALAXIES
# =============================================================================
DWARF_SPHEROIDALS = {
    'source': 'Walker+ 2009, ApJ 704, 1274; McConnachie 2012',
    
    'fornax': {
        'M_star': 2.0e7,  # M☉
        'sigma_los': 10.7,  # km/s (±0.5)
        'r_half_kpc': 0.71,
        'M_dyn_half': 1.5e8,  # M☉ (dynamical within r_half)
        'M_L_ratio': 7.5,  # Very high - "DM dominated"
    },
    
    'draco': {
        'M_star': 2.9e5,
        'sigma_los': 9.1,  # km/s (±1.2)
        'r_half_kpc': 0.22,
        'M_L_ratio': 330,  # Extremely high
    },
    
    'sculptor': {
        'M_star': 2.3e6,
        'sigma_los': 9.2,  # km/s (±0.6)
        'r_half_kpc': 0.28,
        'M_L_ratio': 160,
    },
    
    'carina': {
        'M_star': 3.8e5,
        'sigma_los': 6.6,  # km/s (±1.2)
        'r_half_kpc': 0.25,
        'M_L_ratio': 40,
    },
    
    'mond_prediction': 'Generally works for isolated dSphs',
    'mond_challenge': 'Some dSphs may have EFE from MW'
}

# =============================================================================
# ULTRA-DIFFUSE GALAXIES
# =============================================================================
ULTRA_DIFFUSE_GALAXIES = {
    'df2': {
        'name': 'NGC1052-DF2',
        'source': 'van Dokkum+ 2018, Nature 555, 629',
        
        'M_star': 2e8,  # M☉
        'sigma_los': 8.5,  # km/s (±2.3) - VERY LOW
        'r_eff_kpc': 2.2,
        
        'key_observation': 'Appears to lack dark matter',
        'mond_challenge': 'MOND predicts σ ~ 20 km/s',
        'mond_resolution': 'External field from NGC1052 (EFE)',
        
        'distance_Mpc': 20,  # Disputed - could be 13 Mpc
        'host': 'NGC1052 group'
    },
    
    'df4': {
        'name': 'NGC1052-DF4',
        'source': 'van Dokkum+ 2019',
        'sigma_los': 4.2,  # km/s - even lower!
        'note': 'Similar to DF2, same group'
    },
    
    'dragonfly44': {
        'name': 'Dragonfly 44',
        'source': 'van Dokkum+ 2016, ApJ 828, L6',
        
        'M_star': 3e8,  # M☉
        'sigma_los': 47,  # km/s (±8) - VERY HIGH
        'r_eff_kpc': 4.6,
        
        'key_observation': 'Appears very DM dominated',
        'M_dyn': 1e12,  # M☉
        'note': 'Opposite extreme from DF2'
    }
}

# =============================================================================
# TULLY-FISHER RELATION
# =============================================================================
TULLY_FISHER = {
    'baryonic_tf': {
        'source': 'McGaugh 2012, AJ 143, 40',
        
        'formula': 'M_bar = A × V_flat^b',
        'slope_b': 3.98,  # ±0.06
        'normalization_A': 47,  # M☉/(km/s)^b
        
        'scatter': 0.10,  # dex (intrinsic)
        'note': 'Slope of 4 is natural prediction of MOND'
    },
    
    'stellar_tf': {
        'slope': 3.5,  # Steeper than baryonic
        'source': 'Various'
    }
}

# =============================================================================
# GRAVITATIONAL WAVES
# =============================================================================
GRAVITATIONAL_WAVES = {
    'gw170817': {
        'source': 'Abbott+ 2017, PRL 119, 161101',
        
        'c_gw_constraint': {
            'delta_c_over_c': 1e-15,  # |c_GW - c|/c
            'source': 'GW170817 + GRB170817A timing'
        },
        
        'distance_Mpc': 40,
        'time_delay_s': 1.7,  # GRB arrived 1.7s after GW
        
        'implications': 'Rules out many modified gravity theories'
    },
    
    'ligo_detections': {
        'n_bbh': 90,  # Binary black hole (as of O3)
        'n_bns': 2,   # Binary neutron star
        'n_nsbh': 4,  # Neutron star - black hole
        'source': 'GWTC-3, 2021'
    }
}

# =============================================================================
# COSMIC MICROWAVE BACKGROUND
# =============================================================================
CMB = {
    'planck_2018': {
        'source': 'Planck Collaboration 2020, A&A 641, A6',
        
        'Omega_b': 0.0493,   # Baryon density
        'Omega_c': 0.265,    # Cold dark matter density
        'Omega_m': 0.315,    # Total matter
        'Omega_Lambda': 0.685,
        
        'H0': 67.4,  # km/s/Mpc (±0.5) - TENSION with local
        'sigma8': 0.811,  # ±0.006
        
        'acoustic_peaks': {
            'first_peak_l': 220,
            'first_peak_height': 5700,  # μK²
            'odd_even_ratio': 2.0,  # Sensitive to Ω_b
        },
        
        'mond_challenge': 'CMB requires DM at z~1100; MOND alone fails'
    }
}

# =============================================================================
# STRUCTURE FORMATION
# =============================================================================
STRUCTURE_FORMATION = {
    'sigma8': {
        'planck': 0.811,
        'weak_lensing': 0.76,  # Lower - "S8 tension"
        'source': 'Various'
    },
    
    'galaxy_power_spectrum': {
        'source': 'SDSS, 2dF',
        'bao_scale_Mpc': 150,  # Baryon acoustic oscillation
    },
    
    'lyman_alpha_forest': {
        'note': 'Constrains small-scale power',
        'source': 'Various'
    }
}

# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_benchmarks():
    """Print all observational benchmarks."""
    print("=" * 80)
    print("OBSERVATIONAL BENCHMARKS - GOLD STANDARD DATA")
    print("=" * 80)
    
    print("\n1. MOND SCALE")
    print(f"   a₀ = {MOND_SCALE['a0']:.1e} ± {MOND_SCALE['a0_uncertainty']:.1e} m/s²")
    print(f"   Source: {MOND_SCALE['source']}")
    
    print("\n2. SOLAR SYSTEM")
    print(f"   Cassini γ-1 = (0 ± {SOLAR_SYSTEM['cassini_ppn_gamma']['uncertainty']:.1e})")
    print(f"   Source: {SOLAR_SYSTEM['cassini_ppn_gamma']['source']}")
    
    print("\n3. SPARC GALAXIES")
    print(f"   N = {SPARC['n_quality']} galaxies")
    print(f"   RAR scatter = {SPARC['radial_acceleration_relation']['scatter']:.2f} dex")
    print(f"   MOND RMS = {SPARC['mond_rms']:.2f} km/s")
    
    print("\n4. GALAXY CLUSTERS")
    print(f"   Fox+ 2022: N = {CLUSTERS['fox2022']['n_quality']} clusters")
    print(f"   MOND mass discrepancy: {CLUSTERS['fox2022']['mond_mass_discrepancy']}×")
    
    print("\n5. BULLET CLUSTER")
    print(f"   Gas mass: {CLUSTERS['bullet_cluster']['gas_mass']:.1e} M☉")
    print(f"   Stellar mass: {CLUSTERS['bullet_cluster']['stellar_mass']:.1e} M☉")
    print(f"   Lensing mass: {CLUSTERS['bullet_cluster']['total_lensing_mass']:.1e} M☉")
    print(f"   Challenge: {CLUSTERS['bullet_cluster']['mond_challenge']}")
    
    print("\n6. MILKY WAY")
    print(f"   V_sun = {MILKY_WAY['rotation_curve']['V_sun_kms_new']} ± 3 km/s")
    print(f"   R_sun = {MILKY_WAY['sun_position']['R_gal_kpc']:.3f} kpc")
    print(f"   M_baryonic = {MILKY_WAY['baryonic_mass']['total']:.1e} M☉")
    
    print("\n7. WIDE BINARIES")
    print(f"   Chae 2023: {WIDE_BINARIES['chae_2023']['boost_factor']:.0%} boost at >2000 AU")
    print(f"   Status: {WIDE_BINARIES['chae_2023']['controversy']}")
    
    print("\n8. DWARF SPHEROIDALS")
    print(f"   Fornax: σ = {DWARF_SPHEROIDALS['fornax']['sigma_los']:.1f} km/s, M/L = {DWARF_SPHEROIDALS['fornax']['M_L_ratio']}")
    print(f"   Draco: σ = {DWARF_SPHEROIDALS['draco']['sigma_los']:.1f} km/s, M/L = {DWARF_SPHEROIDALS['draco']['M_L_ratio']}")
    
    print("\n9. ULTRA-DIFFUSE GALAXIES")
    print(f"   DF2: σ = {ULTRA_DIFFUSE_GALAXIES['df2']['sigma_los']:.1f} km/s (LOW - 'no DM')")
    print(f"   Dragonfly44: σ = {ULTRA_DIFFUSE_GALAXIES['dragonfly44']['sigma_los']:.0f} km/s (HIGH - 'lots of DM')")
    
    print("\n10. TULLY-FISHER")
    print(f"   BTFR slope = {TULLY_FISHER['baryonic_tf']['slope_b']:.2f} (MOND predicts 4)")
    print(f"   Scatter = {TULLY_FISHER['baryonic_tf']['scatter']:.2f} dex")
    
    print("\n11. GRAVITATIONAL WAVES")
    print(f"   GW170817: |c_GW - c|/c < {GRAVITATIONAL_WAVES['gw170817']['c_gw_constraint']['delta_c_over_c']:.0e}")
    
    print("\n12. CMB")
    print(f"   Ω_b = {CMB['planck_2018']['Omega_b']:.4f}")
    print(f"   Ω_c = {CMB['planck_2018']['Omega_c']:.3f} (CDM)")
    print(f"   Challenge: {CMB['planck_2018']['mond_challenge']}")


if __name__ == "__main__":
    print_benchmarks()

