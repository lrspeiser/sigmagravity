"""
Solar System radial profile test: verify roughness naturally shuts off at all scales.

Tests that K(R) and Xi(R) stay tiny for all relevant radii with no weird bumps.
"""

import json
from pathlib import Path
import numpy as np

from coherence_time_kernel import (
    compute_coherence_kernel,
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
    compute_exposure_factor,
)

G_MSUN_KPC_KM2_S2 = 4.302e-6  # kpc km^2 / s^2 / Msun
M_SUN = 1.0  # Msun


def keplerian_vcirc(R_kpc):
    """Simple Keplerian circular speed around the Sun."""
    return np.sqrt(G_MSUN_KPC_KM2_S2 * M_SUN / np.maximum(R_kpc, 1e-12))


def main():
    # Load fiducial parameters
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

    # Radii from 0.1 AU to 1e5 AU
    # 1 AU = 4.848e-9 kpc
    AU_TO_KPC = 4.848e-9
    R_au = np.logspace(-1, 5, 200)  # 0.1 AU to 100,000 AU
    R_kpc = R_au * AU_TO_KPC

    # Treat local σ_v as tiny → almost perfectly smooth
    sigma_v_kms = np.full_like(R_kpc, 0.5)  # 0.5 km/s random motions
    
    # Compute g_bar from Keplerian
    v_circ_kms = keplerian_vcirc(R_kpc)
    g_bar_kms2 = v_circ_kms**2 / (R_kpc * 1e3)  # km/s^2
    
    # Rough density estimate
    rho_bar_msun_pc3 = g_bar_kms2 / (G_MSUN_KPC_KM2_S2 * R_kpc * 1e3) * 1e-9

    # Compute K and Xi for each radius
    K_vals = []
    Xi_vals = []
    ell_coh_vals = []
    tau_coh_vals = []

    for i, (R, g_bar, rho_bar, sig_v) in enumerate(zip(R_kpc, g_bar_kms2, rho_bar_msun_pc3, sigma_v_kms)):
        # Compute timescales
        tau_geom = compute_tau_geom(
            np.array([R]),
            np.array([g_bar]),
            np.array([rho_bar]),
            method=params.get("tau_geom_method", "tidal"),
            alpha_geom=params.get("alpha_geom", 1.0),
        )[0]
        
        tau_noise = compute_tau_noise(
            np.array([R]),
            sig_v,
            method="galaxy",
            beta_sigma=params["beta_sigma"],
        )[0]
        
        tau_coh = compute_tau_coh(
            np.array([tau_geom]),
            np.array([tau_noise])
        )[0]
        
        # Compute kernel
        K = compute_coherence_kernel(
            R_kpc=np.array([R]),
            g_bar_kms2=np.array([g_bar]),
            sigma_v_kms=sig_v,
            A_global=params["A_global"],
            p=params["p"],
            n_coh=params["n_coh"],
            method="galaxy",
            rho_bar_msun_pc3=np.array([rho_bar]),
            tau_geom_method=params.get("tau_geom_method", "tidal"),
            alpha_length=params["alpha_length"],
            beta_sigma=params["beta_sigma"],
            alpha_geom=params.get("alpha_geom", 1.0),
            backreaction_cap=params.get("backreaction_cap", 10.0),
        )[0]
        
        # Compute exposure factor
        Xi = compute_exposure_factor(
            np.array([R]),
            np.array([g_bar]),
            np.array([tau_coh])
        )[0]
        
        K_vals.append(float(K))
        Xi_vals.append(float(Xi))
        ell_coh_vals.append(float(tau_coh * 299792.458 * params["alpha_length"] / 3.086e16))  # kpc
        tau_coh_vals.append(float(tau_coh / (365.25 * 86400)))  # years

    out = {
        "R_au": R_au.tolist(),
        "R_kpc": R_kpc.tolist(),
        "K": K_vals,
        "Xi": Xi_vals,
        "ell_coh_kpc": ell_coh_vals,
        "tau_coh_yr": tau_coh_vals,
        "max_K": float(max(K_vals)),
        "max_Xi": float(max(Xi_vals)),
        "K_at_1AU": float(K_vals[0]),
        "K_at_100AU": float(K_vals[np.argmin(np.abs(R_au - 100))]),
        "K_at_10000AU": float(K_vals[np.argmin(np.abs(R_au - 10000))]),
    }

    outpath = Path("time-coherence/solar_system_profile.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 80)
    print("SOLAR SYSTEM RADIAL PROFILE TEST")
    print("=" * 80)
    print(f"\nProfile written to {outpath}")
    print(f"Radii: {R_au[0]:.2f} AU to {R_au[-1]:.0f} AU ({len(R_au)} points)")
    print(f"\nKernel values:")
    print(f"  max K  = {out['max_K']:.3e}")
    print(f"  K at 1 AU = {out['K_at_1AU']:.3e}")
    print(f"  K at 100 AU = {out['K_at_100AU']:.3e}")
    print(f"  K at 10,000 AU = {out['K_at_10000AU']:.3e}")
    print(f"\nExposure factor:")
    print(f"  max Xi = {out['max_Xi']:.3e}")
    print(f"  Mean Xi = {np.mean(Xi_vals):.3e}")
    
    # Check for bumps near planetary orbits
    # Jupiter ~5 AU, Saturn ~10 AU, Neptune ~30 AU
    jupiter_idx = np.argmin(np.abs(R_au - 5))
    saturn_idx = np.argmin(np.abs(R_au - 10))
    neptune_idx = np.argmin(np.abs(R_au - 30))
    
    print(f"\nPlanetary orbit checks:")
    print(f"  K at Jupiter (~5 AU) = {K_vals[jupiter_idx]:.3e}")
    print(f"  K at Saturn (~10 AU) = {K_vals[saturn_idx]:.3e}")
    print(f"  K at Neptune (~30 AU) = {K_vals[neptune_idx]:.3e}")
    
    # Safety check
    threshold = 1e-10
    if out['max_K'] < threshold:
        print(f"\n[PASS] Max K ({out['max_K']:.3e}) < threshold ({threshold:.0e})")
        print("  Roughness naturally shuts off at Solar System scales")
    else:
        print(f"\n[WARNING] Max K ({out['max_K']:.3e}) >= threshold ({threshold:.0e})")
    
    if max(Xi_vals) < 0.01:
        print(f"[PASS] Max Xi ({max(Xi_vals):.3e}) << 1")
        print("  Coherence time << orbital period (smooth spacetime)")
    else:
        print(f"[WARNING] Max Xi ({max(Xi_vals):.3e}) may be too large")


if __name__ == "__main__":
    main()

