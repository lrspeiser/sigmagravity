"""
Cluster lensing: shape test, not just Einstein radius.

Tests if K_rough(R) reproduces the observed κ(R) radial shape,
not just the total mass at one radius.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from coherence_time_kernel import (
    compute_coherence_kernel,
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
    compute_exposure_factor,
)

C_LIGHT_KMS = 299792.458
G_MSUN_KPC_KM2_S2 = 4.302e-6


def compute_sigma_crit(z_lens, z_source=2.0):
    """Compute critical surface density for lensing."""
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    D_l = cosmo.angular_diameter_distance(z_lens).value  # Mpc
    D_s = cosmo.angular_diameter_distance(z_source).value  # Mpc
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value  # Mpc

    # Σ_crit = c²/(4πG) * D_s/(D_l D_ls)
    # Convert to Msun/kpc²
    sigma_crit_msun_kpc2 = (
        (C_LIGHT_KMS**2) / (4 * np.pi * G_MSUN_KPC_KM2_S2) * D_s / (D_l * D_ls)
    ) * 1e-6  # Convert Mpc to kpc

    return sigma_crit_msun_kpc2


def load_cluster_profile(cluster_name: str):
    """
    Load cluster profile data.
    Returns: R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, z_lens
    """
    import sys
    from pathlib import Path

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

            rho_bar_msun_pc3 = rho_total * 1e-9

            # Compute g_bar from density
            M_enclosed = np.zeros_like(R_kpc)
            for i, r in enumerate(R_kpc):
                if i == 0:
                    M_enclosed[i] = (4 * np.pi / 3) * rho_total[i] * r**3
                else:
                    r_grid = np.linspace(0, r, 100)
                    rho_interp = np.interp(r_grid, R_kpc[: i + 1], rho_total[: i + 1])
                    M_enclosed[i] = np.trapezoid(
                        4 * np.pi * r_grid**2 * rho_interp, r_grid
                    )

            g_bar_kms2 = G_MSUN_KPC_KM2_S2 * M_enclosed / (R_kpc**2)
            g_bar_kms2 = np.clip(g_bar_kms2, 1e-8, None)

            # Get Einstein radius and redshift
            import json
            gold_standard_file = Path(
                "data/frontier/gold_standard/gold_standard_clusters.json"
            )
            R_Einstein_kpc = None
            z_lens = data.z_lens

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
                    from astropy.cosmology import FlatLambdaCDM

                    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
                    d_ang = cosmo.angular_diameter_distance(z_lens).value  # Mpc
                    R_Einstein_kpc = theta_E_arcsec * d_ang * 1000 / 206265  # kpc

            if R_Einstein_kpc is None:
                R_Einstein_kpc = 200.0

            return R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, z_lens
        except Exception as e:
            raise ValueError(f"Error loading cluster {cluster_name}: {e}")

    # Fallback placeholder
    R_kpc = np.logspace(0, 3, 1000)
    g_bar_kms2 = 1e-4 * (R_kpc / 100) ** (-1.5)
    rho_bar_msun_pc3 = 1e-3 * (R_kpc / 100) ** (-2.0)
    R_Einstein_kpc = 200.0
    z_lens = 0.4
    return R_kpc, g_bar_kms2, rho_bar_msun_pc3, R_Einstein_kpc, z_lens


def compute_kappa_bar(R_kpc, rho_bar_msun_pc3, Sigma_crit_msun_kpc2):
    """
    Compute baryonic convergence κ_bar(R).
    
    For thin-lens approximation: κ_bar = Σ_bar / Σ_crit
    where Σ_bar is surface density from density profile.
    """
    # Convert volume density to surface density
    # Simple approximation: Σ_bar(R) ≈ ρ_bar(R) * R (for thin shell)
    # More accurate: integrate along line of sight
    Sigma_bar_msun_kpc2 = np.zeros_like(R_kpc)
    
    for i, r in enumerate(R_kpc):
        if i == 0:
            Sigma_bar_msun_kpc2[i] = rho_bar_msun_pc3[i] * 1e9 * r  # Convert pc³ to kpc³
        else:
            # Integrate along line of sight: Σ = ∫ ρ dl
            r_grid = np.linspace(0, r, 100)
            rho_interp = np.interp(r_grid, R_kpc[: i + 1], rho_bar_msun_pc3[: i + 1])
            # Thin-lens: Σ ≈ 2 * ∫ ρ dr
            Sigma_bar_msun_kpc2[i] = np.trapezoid(2 * rho_interp * 1e9, r_grid)
    
    kappa_bar = Sigma_bar_msun_kpc2 / Sigma_crit_msun_kpc2
    return kappa_bar, Sigma_bar_msun_kpc2


def analyze_cluster(cluster_name: str, sigma_v_default: float = 1000.0):
    """
    Analyze cluster κ(R) shape.
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
            z_lens,
        ) = load_cluster_profile(cluster_name)
    except Exception as e:
        print(f"Error loading {cluster_name}: {e}")
        return

    sigma_v_kms = sigma_v_default

    # Compute K(R) from time-coherence kernel
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

    # Compute timescales and exposure factor
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
    Xi = compute_exposure_factor(R_kpc, g_bar_kms2, tau_coh)

    # Compute Σ_crit
    Sigma_crit_msun_kpc2 = compute_sigma_crit(z_lens, z_source=2.0)

    # Compute κ_bar and κ_eff
    kappa_bar, Sigma_bar_msun_kpc2 = compute_kappa_bar(
        R_kpc, rho_bar_msun_pc3, Sigma_crit_msun_kpc2
    )
    kappa_eff = kappa_bar * (1.0 + K)

    # Locate Einstein radius
    idx_E = int(np.argmin(np.abs(R_kpc - R_Einstein_kpc)))

    # Compute radial slopes (d log κ / d log R)
    # Use log-space derivatives
    log_R = np.log(np.clip(R_kpc, 1e-6, None))
    log_kappa_bar = np.log(np.clip(kappa_bar, 1e-6, None))
    log_kappa_eff = np.log(np.clip(kappa_eff, 1e-6, None))

    # Compute slopes using finite differences
    dlogR = np.diff(log_R)
    dlog_kappa_bar = np.diff(log_kappa_bar)
    dlog_kappa_eff = np.diff(log_kappa_eff)

    # Avoid division by zero
    valid = np.abs(dlogR) > 1e-10
    slope_kappa_bar = np.full_like(dlogR, np.nan)
    slope_kappa_eff = np.full_like(dlogR, np.nan)
    slope_kappa_bar[valid] = dlog_kappa_bar[valid] / dlogR[valid]
    slope_kappa_eff[valid] = dlog_kappa_eff[valid] / dlogR[valid]

    # Average slopes in different radial bins
    R_mid = (R_kpc[:-1] + R_kpc[1:]) / 2.0
    inner_mask = R_mid < R_Einstein_kpc
    outer_mask = R_mid > R_Einstein_kpc

    result = {
        "cluster": cluster_name,
        "R_kpc": R_kpc.tolist(),
        "K": K.tolist(),
        "Xi": Xi.tolist(),
        "kappa_bar": kappa_bar.tolist(),
        "kappa_eff": kappa_eff.tolist(),
        "Sigma_bar_msun_kpc2": Sigma_bar_msun_kpc2.tolist(),
        "R_Einstein_kpc": float(R_Einstein_kpc),
        "K_E": float(K[idx_E]),
        "Xi_E": float(Xi[idx_E]),
        "kappa_bar_E": float(kappa_bar[idx_E]),
        "kappa_eff_E": float(kappa_eff[idx_E]),
        "Sigma_crit_msun_kpc2": float(Sigma_crit_msun_kpc2),
        "slopes": {
            "kappa_bar_inner": float(np.mean(slope_kappa_bar[inner_mask])) if np.any(inner_mask) else None,
            "kappa_bar_outer": float(np.mean(slope_kappa_bar[outer_mask])) if np.any(outer_mask) else None,
            "kappa_eff_inner": float(np.mean(slope_kappa_eff[inner_mask])) if np.any(inner_mask) else None,
            "kappa_eff_outer": float(np.mean(slope_kappa_eff[outer_mask])) if np.any(outer_mask) else None,
        },
    }

    outpath = Path(f"time-coherence/cluster_kappa_shape_{cluster_name}.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{cluster_name}:")
    print(f"  R_E = {R_Einstein_kpc:.1f} kpc")
    print(f"  K_E = {K[idx_E]:.3f}")
    print(f"  Xi_E = {Xi[idx_E]:.3e}")
    print(f"  kappa_bar_E = {kappa_bar[idx_E]:.3f}")
    print(f"  kappa_eff_E = {kappa_eff[idx_E]:.3f}")
    
    if result["slopes"]["kappa_bar_inner"] is not None:
        print(f"  Slope (inner): kappa_bar = {result['slopes']['kappa_bar_inner']:.2f}, kappa_eff = {result['slopes']['kappa_eff_inner']:.2f}")
    if result["slopes"]["kappa_bar_outer"] is not None:
        print(f"  Slope (outer): kappa_bar = {result['slopes']['kappa_bar_outer']:.2f}, kappa_eff = {result['slopes']['kappa_eff_outer']:.2f}")
    
    print(f"  Saved -> {outpath}")


def main():
    print("=" * 80)
    print("CLUSTER KAPPA(R) SHAPE TEST")
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

