"""
Debug script to understand why SPARC correlations are 0.0.
Check if K_rough actually varies with R within galaxies.
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
)
from test_sparc_coherence import load_rotmod

G_MSUN_KPC_KM2_S2 = 4.302e-6


def debug_one_galaxy(rotmod_path, params, sigma_v=25.0):
    """Debug a single galaxy to see K_rough variation."""
    df = load_rotmod(str(rotmod_path))
    
    R = df["R_kpc"].to_numpy(dtype=float)
    V_bar = np.sqrt(
        np.clip(
            df["V_gas"].to_numpy() ** 2
            + df["V_disk"].to_numpy() ** 2
            + df["V_bul"].to_numpy() ** 2,
            0.0,
            None,
        )
    )
    V_obs = df["V_obs"].to_numpy()
    mask = (V_bar > 1e-6) & np.isfinite(V_obs) & (V_obs > 0)
    
    if np.sum(mask) < 4:
        return None
    
    R_good = R[mask]
    g_bar_kms2 = (V_bar[mask]**2) / (R_good * 1e3)
    rho_bar_msun_pc3 = g_bar_kms2 / (G_MSUN_KPC_KM2_S2 * R_good * 1e3) * 1e-9
    
    # Compute all at once for efficiency and correctness
    tau_geom = compute_tau_geom(
        R_good,
        g_bar_kms2,
        rho_bar_msun_pc3,
        method=params.get("tau_geom_method", "tidal"),
        alpha_geom=params.get("alpha_geom", 1.0),
    )
    
    tau_noise = compute_tau_noise(
        R_good,
        sigma_v,
        method="galaxy",
        beta_sigma=params.get("beta_sigma", 1.5),
    )
    
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    
    # Compute coherence length
    from coherence_time_kernel import compute_coherence_length
    ell_coh = compute_coherence_length(tau_coh, alpha=params["alpha_length"])
    
    # Compute kernel for all radii at once
    K_rough = compute_coherence_kernel(
        R_kpc=R_good,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v,
        A_global=params["A_global"],
        p=params["p"],
        n_coh=params["n_coh"],
        method="galaxy",
        rho_bar_msun_pc3=rho_bar_msun_pc3,
        tau_geom_method=params.get("tau_geom_method", "tidal"),
        alpha_length=params["alpha_length"],
        beta_sigma=params["beta_sigma"],
        alpha_geom=params.get("alpha_geom", 1.0),
        backreaction_cap=params.get("backreaction_cap", 10.0),
    )
    
    K_req = (V_obs[mask]**2 / V_bar[mask]**2) - 1.0
    
    return {
        "galaxy": Path(rotmod_path).stem.replace("_rotmod", ""),
        "R": R_good,
        "K_rough": K_rough,
        "K_req": K_req,
        "ell_coh": ell_coh,
        "K_rough_std": float(np.std(K_rough)),
        "K_rough_range": float(np.max(K_rough) - np.min(K_rough)),
        "ell_coh_mean": float(np.mean(ell_coh)),
        "R_range": float(np.max(R_good) - np.min(R_good)),
        "corr": float(np.corrcoef(K_req, K_rough)[0, 1]) if np.std(K_rough) > 1e-6 and np.std(K_req) > 1e-6 else 0.0,
    }


def main():
    # Load parameters
    fiducial_path = Path("time-coherence/time_coherence_fiducial.json")
    if fiducial_path.exists():
        with open(fiducial_path, "r") as f:
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
    
    # Test a few galaxies
    rotmod_dir = Path("data/Rotmod_LTG")
    test_galaxies = ["DDO154", "NGC2403", "NGC5055", "UGC11914"]
    
    print("=" * 80)
    print("DEBUGGING SPARC CORRELATIONS")
    print("=" * 80)
    print()
    
    for galaxy_name in test_galaxies:
        path = rotmod_dir / f"{galaxy_name}_rotmod.dat"
        if not path.exists():
            continue
        
        result = debug_one_galaxy(path, params, sigma_v=25.0)
        if result:
            print(f"{result['galaxy']}:")
            print(f"  R range: {result['R'].min():.2f} - {result['R'].max():.2f} kpc ({result['R_range']:.2f} kpc)")
            print(f"  K_rough range: {result['K_rough'].min():.6f} - {result['K_rough'].max():.6f}")
            print(f"  K_rough std: {result['K_rough_std']:.6f}")
            print(f"  K_rough variation: {result['K_rough_range']:.6f}")
            print(f"  ell_coh mean: {result['ell_coh_mean']:.3f} kpc")
            print(f"  Correlation: {result['corr']:.3f}")
            print(f"  K_req range: {result['K_req'].min():.3f} - {result['K_req'].max():.3f}")
            print()


if __name__ == "__main__":
    main()

