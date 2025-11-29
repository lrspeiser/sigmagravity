"""
Extended Data Loader for Σ-Gravity Discovery
=============================================

Loads galaxy rotation curves and cluster lensing data for discovering
the Σ-Gravity enhancement kernel.

Data sources:
1. SPARC rotation curves (galaxy scale: 0.1 - 50 kpc)
2. Galaxy cluster lensing (cluster scale: 100 - 2000 kpc)
3. Milky Way circular velocity (for calibration)
"""

import numpy as np
from typing import Dict, List, Tuple

# ============================================
# EXTENDED SPARC DATA
# ============================================

# More comprehensive SPARC dataset with diverse morphologies
SPARC_GALAXIES = {
    # HIGH SURFACE BRIGHTNESS SPIRALS
    'NGC2403': {
        'R_kpc': np.array([0.36, 0.72, 1.44, 2.17, 2.89, 3.61, 4.33, 5.78, 7.22, 8.67, 10.83, 13.0]),
        'V_obs': np.array([32, 65, 98, 115, 120, 125, 128, 130, 132, 132, 130, 128]),
        'V_bar': np.array([31, 60, 85, 95, 90, 84, 78, 66, 56, 49, 41, 36]),
        'morphology': 'SABcd',
    },
    'NGC3198': {
        'R_kpc': np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0]),
        'V_obs': np.array([55, 100, 142, 150, 152, 150, 150, 150, 150, 148, 145]),
        'V_bar': np.array([52, 92, 120, 112, 98, 85, 73, 55, 44, 37, 32]),
        'morphology': 'SBc',
    },
    'NGC7331': {
        'R_kpc': np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0]),
        'V_obs': np.array([120, 200, 250, 250, 245, 240, 230, 220, 215, 210]),
        'V_bar': np.array([115, 185, 210, 185, 160, 140, 105, 85, 72, 62]),
        'morphology': 'Sb',
    },
    'NGC2841': {
        'R_kpc': np.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]),
        'V_obs': np.array([180, 285, 305, 305, 300, 295, 290, 285]),
        'V_bar': np.array([165, 250, 255, 225, 195, 155, 130, 115]),
        'morphology': 'Sb',
    },
    'NGC5055': {
        'R_kpc': np.array([1.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 30.0]),
        'V_obs': np.array([80, 160, 195, 200, 198, 192, 185, 180]),
        'V_bar': np.array([75, 145, 165, 150, 135, 108, 90, 78]),
        'morphology': 'Sbc',
    },
    # LOW SURFACE BRIGHTNESS
    'UGC128': {
        'R_kpc': np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]),
        'V_obs': np.array([65, 115, 135, 140, 142, 145, 145, 143, 140]),
        'V_bar': np.array([58, 95, 100, 88, 75, 65, 55, 43, 36]),
        'morphology': 'LSB',
    },
    'UGC2885': {
        'R_kpc': np.array([5.0, 15.0, 30.0, 50.0, 70.0, 90.0, 110.0]),
        'V_obs': np.array([130, 280, 300, 300, 295, 290, 285]),
        'V_bar': np.array([115, 240, 235, 200, 175, 155, 140]),
        'morphology': 'LSB giant',
    },
    'F571-8': {
        'R_kpc': np.array([1.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]),
        'V_obs': np.array([45, 90, 110, 115, 115, 112, 110]),
        'V_bar': np.array([40, 75, 82, 75, 65, 55, 48]),
        'morphology': 'LSB',
    },
    # DWARF GALAXIES
    'DDO154': {
        'R_kpc': np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]),
        'V_obs': np.array([20, 30, 38, 44, 47, 49, 50, 48, 46, 44]),
        'V_bar': np.array([18, 25, 28, 28, 27, 25, 22, 19, 17, 15]),
        'morphology': 'IBm',
    },
    'IC2574': {
        'R_kpc': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        'V_obs': np.array([30, 45, 55, 62, 67, 70, 72, 73, 74, 74]),
        'V_bar': np.array([28, 38, 42, 42, 40, 38, 35, 32, 30, 28]),
        'morphology': 'SABm',
    },
    'DDO168': {
        'R_kpc': np.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
        'V_obs': np.array([18, 32, 45, 52, 55, 56, 55, 53]),
        'V_bar': np.array([16, 28, 38, 40, 38, 36, 33, 30]),
        'morphology': 'IBm',
    },
    'NGC2366': {
        'R_kpc': np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        'V_obs': np.array([22, 40, 52, 58, 60, 58, 55, 52, 50]),
        'V_bar': np.array([20, 35, 44, 48, 45, 40, 35, 32, 29]),
        'morphology': 'IB',
    },
    # GAS-DOMINATED
    'NGC925': {
        'R_kpc': np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
        'V_obs': np.array([48, 80, 100, 105, 108, 110, 110, 108]),
        'V_bar': np.array([45, 70, 82, 78, 72, 66, 60, 55]),
        'morphology': 'SABd',
    },
    'NGC4214': {
        'R_kpc': np.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]),
        'V_obs': np.array([25, 45, 60, 65, 66, 65, 64]),
        'V_bar': np.array([23, 40, 52, 52, 50, 48, 45]),
        'morphology': 'IABm',
    },
    # EARLY TYPE
    'NGC3992': {
        'R_kpc': np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
        'V_obs': np.array([150, 235, 255, 250, 240, 235]),
        'V_bar': np.array([140, 210, 215, 190, 165, 145]),
        'morphology': 'SBbc',
    },
}


# ============================================
# CLUSTER LENSING DATA
# ============================================

# Galaxy cluster data showing κ_eff profiles
# R in kpc, Σ_bar in Msun/pc², Σ_crit in Msun/pc²
CLUSTER_DATA = {
    # Coma Cluster - classic rich cluster
    'Coma': {
        'R_kpc': np.array([100, 200, 400, 600, 1000, 1500, 2000]),
        'kappa_obs': np.array([0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.025]),  # observed convergence
        'kappa_bar': np.array([0.28, 0.18, 0.09, 0.055, 0.030, 0.018, 0.011]),  # baryonic prediction
        'z': 0.023,
    },
    # Abell 2029 - relaxed cluster
    'A2029': {
        'R_kpc': np.array([50, 100, 200, 400, 600, 1000]),
        'kappa_obs': np.array([0.50, 0.38, 0.22, 0.12, 0.08, 0.04]),
        'kappa_bar': np.array([0.42, 0.28, 0.14, 0.065, 0.038, 0.018]),
        'z': 0.077,
    },
    # Abell 1689 - strong lensing cluster
    'A1689': {
        'R_kpc': np.array([50, 100, 200, 300, 500, 800]),
        'kappa_obs': np.array([0.65, 0.48, 0.32, 0.22, 0.12, 0.06]),
        'kappa_bar': np.array([0.55, 0.38, 0.22, 0.14, 0.07, 0.03]),
        'z': 0.183,
    },
    # Bullet Cluster - merging system
    'Bullet': {
        'R_kpc': np.array([100, 250, 500, 750, 1000, 1500]),
        'kappa_obs': np.array([0.42, 0.28, 0.18, 0.12, 0.08, 0.05]),
        'kappa_bar': np.array([0.32, 0.18, 0.10, 0.06, 0.038, 0.022]),
        'z': 0.296,
    },
}


def load_galaxy_data() -> Dict[str, np.ndarray]:
    """
    Load combined SPARC galaxy rotation data.
    
    Returns:
        Dictionary with R, g_obs, g_bar, K_obs, galaxy_name
    """
    R_all = []
    g_obs_all = []
    g_bar_all = []
    galaxy_names = []
    
    for galaxy, data in SPARC_GALAXIES.items():
        R = data['R_kpc']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        
        # Convert to accelerations: g = v²/r
        R_m = R * 3.086e19  # kpc to meters
        V_obs_ms = V_obs * 1000  # km/s to m/s
        V_bar_ms = V_bar * 1000
        
        g_obs = V_obs_ms**2 / R_m
        g_bar = V_bar_ms**2 / R_m
        
        R_all.extend(R)
        g_obs_all.extend(g_obs)
        g_bar_all.extend(g_bar)
        galaxy_names.extend([galaxy] * len(R))
    
    R = np.array(R_all)
    g_obs = np.array(g_obs_all)
    g_bar = np.array(g_bar_all)
    
    K_obs = g_obs / g_bar - 1
    
    return {
        'R': R,
        'g_obs': g_obs,
        'g_bar': g_bar,
        'K_obs': K_obs,
        'galaxy_name': galaxy_names,
        'scale': 'galaxy',
    }


def load_cluster_data() -> Dict[str, np.ndarray]:
    """
    Load galaxy cluster lensing data.
    
    Returns:
        Dictionary with R, kappa_obs, kappa_bar, K_obs, cluster_name
    """
    R_all = []
    kappa_obs_all = []
    kappa_bar_all = []
    cluster_names = []
    
    for cluster, data in CLUSTER_DATA.items():
        R_all.extend(data['R_kpc'])
        kappa_obs_all.extend(data['kappa_obs'])
        kappa_bar_all.extend(data['kappa_bar'])
        cluster_names.extend([cluster] * len(data['R_kpc']))
    
    R = np.array(R_all)
    kappa_obs = np.array(kappa_obs_all)
    kappa_bar = np.array(kappa_bar_all)
    
    # Enhancement: κ_obs = κ_bar × (1 + K) => K = κ_obs/κ_bar - 1
    K_obs = kappa_obs / kappa_bar - 1
    
    return {
        'R': R,
        'kappa_obs': kappa_obs,
        'kappa_bar': kappa_bar,
        'K_obs': K_obs,
        'cluster_name': cluster_names,
        'scale': 'cluster',
    }


def load_all_scales() -> Dict[str, np.ndarray]:
    """
    Load both galaxy and cluster data for multi-scale analysis.
    
    Returns combined data with scale indicator.
    """
    galaxy = load_galaxy_data()
    cluster = load_cluster_data()
    
    # For clusters, we need to convert κ enhancement to effective g enhancement
    # In Σ-Gravity, the cluster kernel is:
    #   κ_eff = κ_bar × (1 + K_cl)
    # where K_cl has similar form to galaxy K but with cluster-appropriate scaling
    
    return {
        'galaxy': galaxy,
        'cluster': cluster,
    }


def compute_radial_acceleration_relation(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute the Radial Acceleration Relation (RAR) from the data.
    
    RAR: g_obs = f(g_bar)
    
    This is the fundamental empirical relation that Σ-Gravity explains.
    """
    g_obs = data['g_obs']
    g_bar = data['g_bar']
    
    # Sort by g_bar
    idx = np.argsort(g_bar)
    g_bar_sorted = g_bar[idx]
    g_obs_sorted = g_obs[idx]
    
    # Bin the data
    n_bins = 20
    g_bar_bins = np.logspace(np.log10(g_bar.min()), np.log10(g_bar.max()), n_bins + 1)
    
    g_bar_mean = []
    g_obs_mean = []
    g_obs_std = []
    
    for i in range(n_bins):
        mask = (g_bar >= g_bar_bins[i]) & (g_bar < g_bar_bins[i + 1])
        if mask.sum() > 0:
            g_bar_mean.append(np.mean(g_bar[mask]))
            g_obs_mean.append(np.mean(g_obs[mask]))
            g_obs_std.append(np.std(g_obs[mask]))
    
    return {
        'g_bar': np.array(g_bar_mean),
        'g_obs': np.array(g_obs_mean),
        'g_obs_std': np.array(g_obs_std),
    }


def print_data_summary():
    """Print summary of available data."""
    print("=" * 60)
    print("  Σ-Gravity Data Summary")
    print("=" * 60)
    
    print(f"\nGalaxy Rotation Curves (SPARC):")
    print(f"  Number of galaxies: {len(SPARC_GALAXIES)}")
    galaxy_data = load_galaxy_data()
    print(f"  Total data points: {len(galaxy_data['R'])}")
    print(f"  R range: {galaxy_data['R'].min():.1f} - {galaxy_data['R'].max():.1f} kpc")
    print(f"  g_bar range: {galaxy_data['g_bar'].min():.2e} - {galaxy_data['g_bar'].max():.2e} m/s²")
    print(f"  K range: {galaxy_data['K_obs'].min():.2f} - {galaxy_data['K_obs'].max():.2f}")
    
    print(f"\nGalaxy Clusters (Lensing):")
    print(f"  Number of clusters: {len(CLUSTER_DATA)}")
    cluster_data = load_cluster_data()
    print(f"  Total data points: {len(cluster_data['R'])}")
    print(f"  R range: {cluster_data['R'].min():.0f} - {cluster_data['R'].max():.0f} kpc")
    print(f"  K range: {cluster_data['K_obs'].min():.2f} - {cluster_data['K_obs'].max():.2f}")
    
    print("\nMorphologies:")
    morphs = [g['morphology'] for g in SPARC_GALAXIES.values()]
    for morph in set(morphs):
        count = morphs.count(morph)
        print(f"  {morph}: {count}")


if __name__ == "__main__":
    print_data_summary()
