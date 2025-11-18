"""
Cluster time delay analysis: test "extra time in the field" picture for lensing.

Computes exposure factors Xi(R) and estimates extra Shapiro time delays
for rays passing near Einstein radii.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from coherence_time_kernel import (
    compute_coherence_kernel,
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
    compute_coherence_length,
    compute_exposure_factor,
)

C_LIGHT_KMS = 299792.458


def load_cluster_profile(cluster_name: str):
    """
    Load cluster profile data.
    Returns: R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, M_required_Msun
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from many_path_model.cluster_data_loader import ClusterDataLoader
        CLUSTER_LOADER_AVAILABLE = True
    except ImportError:
        CLUSTER_LOADER_AVAILABLE = False
    
    if CLUSTER_LOADER_AVAILABLE:
        try:
            loader = ClusterDataLoader()
            data = loader.load_cluster(cluster_name, validate=False)
            
            R_kpc = data.r_kpc
            rho_total = data.rho_total  # Msun/kpc³
            
            # Convert to Msun/pc³
            rho_bar_msun_pc3 = rho_total * 1e-9
            
            # Compute g_bar from density
            G_msun_kpc_km2_s2 = 4.302e-6
            M_enclosed = np.zeros_like(R_kpc)
            for i, r in enumerate(R_kpc):
                if i == 0:
                    M_enclosed[i] = (4 * np.pi / 3) * rho_total[i] * r**3
                else:
                    r_grid = np.linspace(0, r, 100)
                    rho_interp = np.interp(r_grid, R_kpc[: i + 1], rho_total[: i + 1])
                    M_enclosed[i] = np.trapezoid(4 * np.pi * r_grid**2 * rho_interp, r_grid)
            
            g_bar_kms2 = G_msun_kpc_km2_s2 * M_enclosed / (R_kpc**2)
            g_bar_kms2 = np.clip(g_bar_kms2, 1e-8, None)
            
            # Get Einstein radius from gold standard
            import json
            gold_standard_file = Path("data/frontier/gold_standard/gold_standard_clusters.json")
            R_Einstein_kpc = None
            M_required_Msun = None
            
            if gold_standard_file.exists():
                with open(gold_standard_file, "r") as f:
                    gold_data = json.load(f)
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
                    M_required_Msun = 1e14 * (R_Einstein_kpc / 200) ** 2
            
            if R_Einstein_kpc is None:
                R_Einstein_kpc = 200.0
                M_required_Msun = 1e14
            
            return R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, M_required_Msun
        except Exception as e:
            raise ValueError(f"Error loading cluster {cluster_name}: {e}")
    
    # Fallback placeholder
    R_kpc = np.logspace(0, 3, 1000)
    g_bar_kms2 = 1e-4 * (R_kpc / 100) ** (-1.5)
    rho_bar_msun_pc3 = 1e-3 * (R_kpc / 100) ** (-2.0)
    R_Einstein_kpc = 200.0
    M_required_Msun = 1e14
    return R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, M_required_Msun


def analyze_cluster(cluster_name: str, sigma_v_default: float = 1000.0):
    """
    Analyze a single cluster: compute K(R), Xi(R), and time delays.
    """
    params_path = Path("time-coherence/time_coherence_fiducial.json")
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {
            "A_global": 1.0,
            "p": 0.757,
            "n_coh": 0.5,
            "alpha_length": 0.037,
            "beta_sigma": 1.5,
            "alpha_geom": 1.0,
            "backreaction_cap": 10.0,
            "tau_geom_method": "tidal",
        }

    try:
        (
            R_kpc,
            g_bar_kms2,
            rho_bar_msun_pc3,
            R_Einstein_kpc,
            M_required_Msun,
        ) = load_cluster_profile(cluster_name)
    except Exception as e:
        print(f"Error loading {cluster_name}: {e}")
        return

    sigma_v_kms = sigma_v_default  # can be refined with real data later

    K = compute_coherence_kernel(
        R_kpc=R_kpc,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v_kms,
        A_global=params["A_global"],
        p=params["p"],
        n_coh=params["n_coh"],
        method="cluster",
        rho_bar_msun_pc3=rho_bar_msun_pc3,
        tau_geom_method=params.get("tau_geom_method", "tidal"),
        alpha_length=params["alpha_length"],
        beta_sigma=params["beta_sigma"],
        alpha_geom=params.get("alpha_geom", 1.0),
        backreaction_cap=params.get("backreaction_cap", 10.0),
    )

    tau_geom = compute_tau_geom(
        R_kpc,
        g_bar_kms2,
        rho_bar_msun_pc3,
        method=params.get("tau_geom_method", "tidal"),
        alpha_geom=params.get("alpha_geom", 1.0),
    )
    tau_noise = compute_tau_noise(
        R_kpc,
        sigma_v_kms,
        method="cluster",
        beta_sigma=params["beta_sigma"],
    )
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    ell_coh_kpc = compute_coherence_length(tau_coh, alpha=params["alpha_length"])
    Xi = compute_exposure_factor(R_kpc, g_bar_kms2, tau_coh)

    # Locate Einstein radius index
    idx_E = int(np.argmin(np.abs(R_kpc - R_Einstein_kpc)))

    # Very rough extra Shapiro delay estimate:
    # Δt_extra ≈ ∫ K(R) Φ(R) dl / c^3  ~ K * Φ * (path length)/c^3
    # For diagnostics we just compute dimensionless ratio of delays.
    phi_gr = g_bar_kms2 * R_kpc * 1e3  # ≈ v_circ^2 (km^2/s^2)
    # Path length ~ few × R_E; take ~2 R_E for simplicity
    path_kpc = 2.0 * R_Einstein_kpc
    path_km = path_kpc * 3.086e16

    phi_E = phi_gr[idx_E]
    K_E = K[idx_E]
    dt_gr_sec = phi_E * path_km / (C_LIGHT_KMS**3 * 1e3)
    dt_extra_sec = phi_E * K_E * path_km / (C_LIGHT_KMS**3 * 1e3)

    result = {
        "cluster": cluster_name,
        "R_kpc": R_kpc.tolist(),
        "K": K.tolist(),
        "Xi": Xi.tolist(),
        "ell_coh_kpc": ell_coh_kpc.tolist(),
        "R_Einstein_kpc": float(R_Einstein_kpc),
        "K_E": float(K_E),
        "Xi_E": float(Xi[idx_E]),
        "ell_coh_E_kpc": float(ell_coh_kpc[idx_E]),
        "dt_shapiro_gr_yr": float(dt_gr_sec / (365.25 * 86400)),
        "dt_shapiro_extra_yr": float(dt_extra_sec / (365.25 * 86400)),
        "dt_ratio": float(dt_extra_sec / dt_gr_sec) if dt_gr_sec > 0 else 0.0,
    }

    outpath = Path(f"time-coherence/cluster_time_delay_{cluster_name}.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"{cluster_name}:")
    print(f"  R_E = {R_Einstein_kpc:.1f} kpc")
    print(f"  K_E = {K_E:.3f}")
    print(f"  Xi_E = {Xi[idx_E]:.3e}")
    print(f"  ell_coh_E = {ell_coh_kpc[idx_E]:.1f} kpc")
    print(f"  dt_shapiro_GR = {result['dt_shapiro_gr_yr']:.3e} yr")
    print(f"  dt_shapiro_extra = {result['dt_shapiro_extra_yr']:.3e} yr")
    print(f"  dt_ratio = {result['dt_ratio']:.3f}")
    print(f"  Saved → {outpath}")


def main():
    print("=" * 80)
    print("CLUSTER TIME DELAY ANALYSIS")
    print("=" * 80)
    print()
    
    clusters = ["MACSJ0416", "MACSJ0717", "ABELL_1689"]
    
    for name in clusters:
        try:
            analyze_cluster(name, sigma_v_default=1000.0)
            print()
        except Exception as e:
            print(f"Error processing {name}: {e}")
            print()


if __name__ == "__main__":
    main()

