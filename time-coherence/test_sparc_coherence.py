"""
Test time-coherence kernel on SPARC galaxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import glob
import os
import json
import argparse
from pathlib import Path
from coherence_time_kernel import compute_coherence_kernel
from morphology_gate import apply_morphology_gate

# Load fiducial parameters
_fiducial_path = Path(__file__).parent / "time_coherence_fiducial.json"
if _fiducial_path.exists():
    with open(_fiducial_path, "r") as f:
        _fiducial = json.load(f)
else:
    _fiducial = {"alpha_length": 0.037, "beta_sigma": 1.5, "backreaction_cap": 10.0, "A_global": 1.0, "p": 0.757, "n_coh": 0.5}


def load_rotmod(path: str) -> pd.DataFrame:
    """Load SPARC rotmod file."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4, 5],
        names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
        engine="python",
    )
    v_gr = np.sqrt(
        np.clip(
            df["V_gas"].to_numpy() ** 2
            + df["V_disk"].to_numpy() ** 2
            + df["V_bul"].to_numpy() ** 2,
            0.0,
            None,
        )
    )
    df["V_gr"] = v_gr
    return df[["R_kpc", "V_obs", "V_gr", "V_gas", "V_disk", "V_bul"]]


def process_galaxy(
    path: str,
    sigma_v_kms: float,
    A_global: float = 1.0,
    p: float = 0.757,
    tau_geom_method: str = "tidal",
    galaxy_meta: dict = None,
) -> dict | None:
    """Process a single galaxy."""
    galaxy = Path(path).name.replace("_rotmod.dat", "")
    
    try:
        df = load_rotmod(path)
    except Exception:
        return None
    
    R = df["R_kpc"].to_numpy(float)
    V_obs = df["V_obs"].to_numpy(float)
    V_gr = df["V_gr"].to_numpy(float)
    
    if len(R) < 4:
        return None
    
    # Compute g_bar from V_gr
    g_bar_kms2 = (V_gr**2) / (R * 1e3)  # km/s²
    
    # Estimate density for tidal method
    # From g_bar: rho ~ g / (G R)
    G_msun_kpc_km2_s2 = 4.302e-6
    rho_bar_msun_pc3 = g_bar_kms2 / (G_msun_kpc_km2_s2 * R * 1e3) * 1e-9
    
    # Compute coherence kernel
    K_raw = compute_coherence_kernel(
        R_kpc=R,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v_kms,
        A_global=A_global,
        p=p,
        method="galaxy",
        rho_bar_msun_pc3=rho_bar_msun_pc3 if tau_geom_method == "tidal" else None,
        tau_geom_method=tau_geom_method,
        alpha_length=_fiducial.get("alpha_length", 0.037),
        beta_sigma=_fiducial.get("beta_sigma", 1.5),
        backreaction_cap=_fiducial.get("backreaction_cap"),
    )
    
    # Apply morphology gate if metadata available
    if galaxy_meta is not None:
        K = apply_morphology_gate(K_raw, galaxy_meta)
    else:
        K = K_raw
    
    # Compute coherence scales for diagnostics
    from coherence_time_kernel import (
        compute_tau_geom,
        compute_tau_noise,
        compute_tau_coh,
        compute_coherence_length,
    )
    
    tau_geom = compute_tau_geom(
        R, g_bar_kms2, rho_bar_msun_pc3 if tau_geom_method == "tidal" else None, method=tau_geom_method
    )
    tau_noise = compute_tau_noise(R, sigma_v_kms, method="galaxy", beta_sigma=1.5)
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    ell_coh = compute_coherence_length(tau_coh, alpha=0.037)
    
    # Apply enhancement
    f_enh = 1.0 + K
    V_model = V_gr * np.sqrt(np.clip(f_enh, 0.0, None))
    
    # Compute RMS
    rms_gr = float(np.sqrt(np.mean((V_obs - V_gr) ** 2)))
    rms_coherence = float(np.sqrt(np.mean((V_obs - V_model) ** 2)))
    
    return {
        "galaxy": galaxy,
        "sigma_v_kms": sigma_v_kms,
        "n_points": len(R),
        "rms_gr": rms_gr,
        "rms_coherence": rms_coherence,
        "delta_rms": rms_coherence - rms_gr,
        "K_mean": float(np.mean(K)),
        "K_max": float(np.max(K)),
        "ell_coh_mean_kpc": float(np.mean(ell_coh)),
        "tau_coh_mean_yr": float(np.mean(tau_coh) / (365.25 * 86400)),
    }


def main():
    parser = argparse.ArgumentParser(description='Test time-coherence kernel on SPARC galaxies')
    parser.add_argument('--params-json', type=str, 
                       default='time-coherence/time_coherence_fiducial.json',
                       help='Path to parameters JSON file')
    parser.add_argument('--out-csv', type=str, 
                       default='time-coherence/sparc_coherence_test.csv',
                       help='Path to output CSV file')
    args = parser.parse_args()
    
    # Load parameters
    params_path = Path(args.params_json)
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
    else:
        print(f"Warning: {params_path} not found, using defaults")
        params = _fiducial
    
    rotmod_dir = "data/Rotmod_LTG"
    summary_csv = "data/sparc/sparc_combined.csv"
    
    # Load sigma map
    summary = pd.read_csv(summary_csv)
    sigma_map = dict(
        zip(
            summary["galaxy_name"].astype(str),
            summary["sigma_velocity"].astype(float),
        )
    )
    
    rotmod_paths = sorted(glob.glob(os.path.join(rotmod_dir, "*_rotmod.dat")))
    print(f"Processing {len(rotmod_paths)} SPARC galaxies...")
    print(f"Using parameters from: {params_path}")
    
    # Use fiducial parameters
    A_global = params.get("A_global", 1.0)
    p = params.get("p", 0.757)
    tau_geom_method = params.get("tau_geom_method", "tidal")
    use_morphology_gate = params.get("use_morphology_gate", True)
    
    print(f"Parameters: A={A_global}, p={p}, tau_geom_method={tau_geom_method}")
    if use_morphology_gate:
        print("Morphology gates: ENABLED")
    print()
    
    # Create galaxy metadata map
    galaxy_meta_map = {}
    if use_morphology_gate:
        for _, row in summary.iterrows():
            galaxy_name = str(row["galaxy_name"]).strip()
            galaxy_meta_map[galaxy_name] = {
                "bar_flag": row.get("bar_flag", 0),
                "warp_flag": row.get("warp_flag", 0),
                "bulge_frac": row.get("bulge_frac", 0.0),
                "inclination": row.get("inclination", 90.0),
            }
    
    results = []
    for i, path in enumerate(rotmod_paths):
        galaxy = Path(path).name.replace("_rotmod.dat", "")
        sigma_v = sigma_map.get(galaxy, 25.0)
        galaxy_meta = galaxy_meta_map.get(galaxy) if use_morphology_gate else None
        
        result = process_galaxy(
            path, sigma_v, A_global=A_global, p=p, tau_geom_method=tau_geom_method,
            galaxy_meta=galaxy_meta
        )
        if result:
            results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(rotmod_paths)} galaxies...")
    
    df = pd.DataFrame(results)
    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults:")
    print(f"  Total galaxies: {len(df)}")
    print(f"  Mean delta_rms: {df['delta_rms'].mean():.3f} km/s")
    print(f"  Median delta_rms: {df['delta_rms'].median():.3f} km/s")
    print(f"  Improved: {(df['delta_rms'] < 0).sum()} / {len(df)} ({(df['delta_rms'] < 0).sum()/len(df)*100:.1f}%)")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

