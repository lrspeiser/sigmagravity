"""
Test time-coherence kernel on galaxy clusters (lensing).
"""

from __future__ import annotations

import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import cluster loader
sys.path.insert(0, str(Path(__file__).parent.parent))

from coherence_time_kernel import compute_coherence_kernel

try:
    from many_path_model.cluster_data_loader import ClusterDataLoader
    CLUSTER_LOADER_AVAILABLE = True
except ImportError:
    CLUSTER_LOADER_AVAILABLE = False
    print("Warning: cluster_data_loader not found, using placeholder")


def load_cluster_profile(cluster_name: str):
    """
    Load cluster baryonic profile.
    
    Returns R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, M_required_Msun
    """
    if CLUSTER_LOADER_AVAILABLE:
        try:
            loader = ClusterDataLoader()
            data = loader.load_cluster(cluster_name, validate=False)
            
            R_kpc = data.r_kpc
            rho_total = data.rho_total  # Msun/kpc³
            
            # Convert to Msun/pc³
            rho_bar_msun_pc3 = rho_total * 1e-9
            
            # Compute g_bar from density
            # g_bar = GM(<R) / R², M(<R) = ∫ 4πr²ρ(r) dr
            G_msun_kpc_km2_s2 = 4.302e-6
            M_enclosed = np.zeros_like(R_kpc)
            for i, r in enumerate(R_kpc):
                if i == 0:
                    M_enclosed[i] = (4 * np.pi / 3) * rho_total[i] * r**3
                else:
                    # Integrate from 0 to R
                    r_grid = np.linspace(0, r, 100)
                    rho_interp = np.interp(r_grid, R_kpc[: i + 1], rho_total[: i + 1])
                    M_enclosed[i] = np.trapezoid(4 * np.pi * r_grid**2 * rho_interp, r_grid)
            
            g_bar_kms2 = G_msun_kpc_km2_s2 * M_enclosed / (R_kpc**2)
            g_bar_kms2 = np.clip(g_bar_kms2, 1e-8, None)
            
            # Get Einstein radius from gold standard
            gold_standard_file = Path("data/frontier/gold_standard/gold_standard_clusters.json")
            R_Einstein_kpc = None
            M_required_Msun = None
            
            if gold_standard_file.exists():
                gold_data = json.loads(gold_standard_file.read_text())
                name_mapping = {
                    "MACSJ0416": "macs0416",
                    "MACSJ0717": "macs0717",
                    "ABELL_1689": "a1689",
                    "ABELL_0370": "a370",
                }
                key = name_mapping.get(cluster_name, cluster_name.lower())
                if key in gold_data:
                    theta_E_arcsec = gold_data[key]["accepted"]["theta_E_arcsec"]
                    z_lens = data.z_lens
                    # Convert arcsec to kpc
                    from astropy.cosmology import FlatLambdaCDM
                    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
                    d_ang = cosmo.angular_diameter_distance(z_lens).value  # Mpc
                    R_Einstein_kpc = theta_E_arcsec * d_ang * 1000 / 206265  # kpc
                    
                    # Estimate required mass (rough)
                    M_required_Msun = 1e14 * (R_Einstein_kpc / 200) ** 2
            
            if R_Einstein_kpc is None:
                R_Einstein_kpc = 200.0  # Default
                M_required_Msun = 1e14  # Default
            
            return R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, M_required_Msun
            
        except Exception as e:
            print(f"Error loading cluster {cluster_name}: {e}")
            # Fall through to placeholder
    
    # Placeholder fallback
    R_kpc = np.logspace(0, 3, 1000)  # 1 to 1000 kpc
    g_bar_kms2 = 1e-4 * (R_kpc / 100) ** (-1.5)  # km/s²
    rho_bar_msun_pc3 = 1e-3 * (R_kpc / 100) ** (-2.0)  # Msun/pc³
    R_Einstein_kpc = 200.0  # kpc
    M_required_Msun = 1e14  # Msun
    
    return R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, M_required_Msun


def compute_enclosed_mass(R_kpc: np.ndarray, g_bar_kms2: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute enclosed mass profile with enhancement.
    
    M(<R) = ∫ g_eff(R') · R'² dR' / G
    """
    G_msun_kpc_km2_s2 = 4.302e-6
    g_eff = g_bar_kms2 * (1.0 + K)
    
    # Integrate: M(<R) = (1/G) ∫ g_eff(R') R'² dR'
    integrand = g_eff * R_kpc**2
    M_enclosed = np.trapezoid(integrand, R_kpc) / G_msun_kpc_km2_s2
    
    return M_enclosed


def test_cluster(cluster_name: str, sigma_v_kms: float = 1000.0, params: dict = None):
    """Test coherence kernel on a cluster."""
    print(f"\nTesting cluster: {cluster_name}")
    
    R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_E_kpc, M_req = load_cluster_profile(cluster_name)
    
    # Test different turbulence assumptions
    results = []
    
    for v_turb_kms in [100, 300, 500, 1000]:
        for L_turb_kpc in [50, 100, 200, 500]:
            for A_global in [1.0, 2.0, 5.0, 10.0]:
                # Use params if provided, otherwise defaults
                if params is None:
                    params = {"alpha_length": 0.037, "beta_sigma": 1.5, "backreaction_cap": 10.0,
                             "A_global": 1.0, "p": 0.757, "n_coh": 0.5}
                
                K = compute_coherence_kernel(
                    R_kpc=R_kpc,
                    g_bar_kms2=g_bar_kms2,
                    sigma_v_kms=sigma_v_kms,
                    A_global=params.get("A_global", A_global),
                    p=params.get("p", 0.757),
                    n_coh=params.get("n_coh", 0.5),
                    method="cluster",
                    v_turb_kms=v_turb_kms,
                    L_turb_kpc=L_turb_kpc,
                    rho_bar_msun_pc3=rho_bar_msun_pc3,
                    alpha_length=params.get("alpha_length", 0.037),
                    beta_sigma=params.get("beta_sigma", 1.5),
                    backreaction_cap=params.get("backreaction_cap"),
                )
                
                # Find K at Einstein radius
                idx_E = np.argmin(np.abs(R_kpc - R_E_kpc))
                K_E = K[idx_E]
                
                # Compute mass boost
                M_enclosed = compute_enclosed_mass(R_kpc, g_bar_kms2, K)
                M_baryon = compute_enclosed_mass(R_kpc, g_bar_kms2, np.zeros_like(K))
                
                mass_boost = M_enclosed / M_baryon if M_baryon > 0 else 1.0
                
                # Check if sufficient
                M_total = M_baryon * mass_boost
                sufficient = M_total >= M_req
                
                results.append({
                    "v_turb_kms": v_turb_kms,
                    "L_turb_kpc": L_turb_kpc,
                    "A_global": A_global,
                    "K_Einstein": float(K_E),
                    "mass_boost": float(mass_boost),
                    "M_total_Msun": float(M_total),
                    "M_required_Msun": float(M_req),
                    "sufficient": bool(sufficient),  # Convert numpy bool to Python bool
                })
                
                if sufficient:
                    print(
                        f"  [OK] v_turb={v_turb_kms}, L_turb={L_turb_kpc}, A={A_global}: "
                        f"K_E={K_E:.4f}, boost={mass_boost:.2f}x"
                    )
    
    return results


def main():
    print("Testing time-coherence kernel on galaxy clusters...")
    
    # Test on example clusters (use correct names)
    clusters = ["ABELL_1689", "ABELL_0370", "MACSJ0416", "MACSJ0717"]
    
    all_results = {}
    for cluster in clusters:
        results = test_cluster(cluster, sigma_v_kms=1000.0)
        all_results[cluster] = results
    
    # Save results
    output = {
        "clusters": all_results,
    }
    
    Path("time-coherence/cluster_coherence_test.json").write_text(
        json.dumps(output, indent=2)
    )
    print("\nResults saved to time-coherence/cluster_coherence_test.json")


if __name__ == "__main__":
    main()

