"""
Solar System safety test for time-coherence kernel.
Tests that "roughness" naturally switches off in strong-field, high-coherence environments.
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
)

FIDUCIAL_PATH = Path("time-coherence/time_coherence_fiducial.json")

def load_fiducial():
    if FIDUCIAL_PATH.exists():
        with open(FIDUCIAL_PATH, "r") as f:
            return json.load(f)
    return {
        "A_global": 1.0,
        "p": 0.757,
        "n_coh": 0.5,
        "alpha_length": 0.037,
        "beta_sigma": 1.5,
        "alpha_geom": 1.0,
        "backreaction_cap": 10.0,
        "tau_geom_method": "tidal",
    }


def main():
    """
    Sample radii: 1 AU, 10 AU, 10^3 AU, 10^4 AU around a 1 Msun star.
    """
    params = load_fiducial()

    # Radii in AU
    R_au = np.array([1.0, 10.0, 1e3, 1e4])
    # 1 pc = 206265 AU; 1 kpc = 1e3 pc
    AU_TO_KPC = 1.0 / (206265.0 * 1e3)
    R_kpc = R_au * AU_TO_KPC

    # Simple 1 Msun point-mass g_bar
    G_msun_kpc_km2_s2 = 4.302e-6
    M_sun = 1.0
    g_bar_kms2 = G_msun_kpc_km2_s2 * M_sun / np.clip(R_kpc, 1e-12, None) ** 2

    # Rough density estimate via rho ~ g / (G R)
    rho_bar_msun_pc3 = (
        g_bar_kms2 / (G_msun_kpc_km2_s2 * np.clip(R_kpc, 1e-12, None) * 1e3) * 1e-9
    )

    # Take σ_v ~ 10 km/s as a high-end Solar neighborhood dispersion
    sigma_v_kms = 10.0

    K = compute_coherence_kernel(
        R_kpc=R_kpc,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v_kms,
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
        method="galaxy",
        beta_sigma=params["beta_sigma"],
    )
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    ell_coh_kpc = compute_coherence_length(tau_coh, alpha=params["alpha_length"])

    print("=" * 80)
    print("SOLAR-SYSTEM COHERENCE TEST (1 Msun)")
    print("=" * 80)
    print("R [AU]   R [kpc]        K          ell_coh [kpc]   tau_coh [yr]")
    for R_au_i, R_kpc_i, K_i, ell_i, t_i in zip(
        R_au,
        R_kpc,
        K,
        ell_coh_kpc,
        tau_coh / (365.25 * 86400),
    ):
        print(
            f"{R_au_i:7.1f}  {R_kpc_i:9.3e}  {K_i:12.4e}  "
            f"{ell_i:13.4e}  {t_i:11.3e}"
        )

    print(
        "\nExpectation: K should be << 10^-6 at all these radii so the Solar System "
        "passes existing constraints."
    )
    
    # Save results
    result = {
        "R_AU": R_au.tolist(),
        "R_kpc": R_kpc.tolist(),
        "K": K.tolist(),
        "ell_coh_kpc": ell_coh_kpc.tolist(),
        "tau_coh_yr": (tau_coh / (365.25 * 86400)).tolist(),
        "max_K": float(np.max(K)),
        "safety_check": {
            "max_K_acceptable": 1e-6,
            "passed": float(np.max(K)) < 1e-6,
        }
    }
    
    outpath = Path("time-coherence/solar_system_coherence_test.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to {outpath}")

if __name__ == "__main__":
    main()

