"""
Map time-coherence kernel to Burr-XII empirical form.
Fits Burr-XII to K_theory(R) for sample galaxies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.optimize import curve_fit
from coherence_time_kernel import compute_coherence_kernel

def burr_xii_kernel(R_kpc: np.ndarray, ell_0_kpc: float, p: float, n: float, A: float) -> np.ndarray:
    """
    Burr-XII kernel form: K(R) = A * [1 - (1 + (R/ell_0)^p)^(-n)]
    """
    x = R_kpc / ell_0_kpc
    x = np.clip(x, 1e-6, None)  # Avoid division by zero
    K = A * (1.0 - (1.0 + x**p) ** (-n))
    return np.clip(K, 0.0, None)

def fit_burr_xii_to_theory(R_kpc: np.ndarray, K_theory: np.ndarray, 
                           p_fixed: float = None, n_fixed: float = None) -> dict:
    """
    Fit Burr-XII to theory kernel K_theory(R).
    
    Returns best-fit parameters: ell_0_kpc, p, n, A
    """
    # Initial guess
    K_max = np.max(K_theory)
    R_peak = R_kpc[np.argmax(K_theory)]
    
    # Fit function
    if p_fixed is not None and n_fixed is not None:
        # Only fit ell_0 and A
        def fit_func(R, ell_0, A):
            return burr_xii_kernel(R, ell_0, p_fixed, n_fixed, A)
        
        p0 = [R_peak, K_max]
        bounds = ([R_peak * 0.1, 0.0], [R_peak * 10.0, K_max * 2.0])
        popt, pcov = curve_fit(fit_func, R_kpc, K_theory, p0=p0, bounds=bounds, maxfev=10000)
        ell_0, A = popt
        p, n = p_fixed, n_fixed
    else:
        # Fit all parameters
        def fit_func(R, ell_0, p, n, A):
            return burr_xii_kernel(R, ell_0, p, n, A)
        
        p0 = [R_peak, 0.757, 0.5, K_max]
        bounds = ([R_peak * 0.1, 0.1, 0.1, 0.0], 
                  [R_peak * 10.0, 2.0, 2.0, K_max * 2.0])
        popt, pcov = curve_fit(fit_func, R_kpc, K_theory, p0=p0, bounds=bounds, maxfev=10000)
        ell_0, p, n, A = popt
    
    # Compute RMS between theory and fitted Burr-XII
    K_fitted = burr_xii_kernel(R_kpc, ell_0, p, n, A)
    rms = np.sqrt(np.mean((K_theory - K_fitted)**2))
    max_diff = np.max(np.abs(K_theory - K_fitted))
    
    return {
        "ell_0_kpc": float(ell_0),
        "p": float(p),
        "n": float(n),
        "A": float(A),
        "rms": float(rms),
        "max_diff": float(max_diff),
        "relative_rms": float(rms / (np.max(K_theory) + 1e-6)),
    }

def test_mw_mapping():
    """Test mapping for MW."""
    from test_mw_coherence import load_mw_profile
    
    print("Testing MW mapping...")
    R_kpc, g_bar_kms2, rho_bar_msun_pc3 = load_mw_profile(r_min=12.0, r_max=16.0)
    sigma_v_mw = 30.0
    
    # Load fiducial params
    fiducial_path = Path("time-coherence/time_coherence_fiducial.json")
    if fiducial_path.exists():
        with open(fiducial_path, "r") as f:
            fiducial = json.load(f)
    else:
        fiducial = {"alpha_length": 0.037, "beta_sigma": 1.5, "backreaction_cap": 10.0, 
                    "A_global": 1.0, "p": 0.757, "n_coh": 0.5}
    
    # Compute theory kernel
    K_theory = compute_coherence_kernel(
        R_kpc=R_kpc,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v_mw,
        A_global=fiducial["A_global"],
        p=fiducial["p"],
        n_coh=fiducial["n_coh"],
        method="galaxy",
        rho_bar_msun_pc3=rho_bar_msun_pc3,
        tau_geom_method="tidal",
        alpha_length=fiducial["alpha_length"],
        beta_sigma=fiducial["beta_sigma"],
        backreaction_cap=fiducial.get("backreaction_cap"),
    )
    
    # Fit Burr-XII
    result = fit_burr_xii_to_theory(R_kpc, K_theory)
    
    print(f"  MW (12-16 kpc):")
    print(f"    Theory: ell_coh ~ {np.mean(R_kpc[np.argmax(K_theory)]):.2f} kpc (peak)")
    print(f"    Burr-XII fit:")
    print(f"      ell_0 = {result['ell_0_kpc']:.2f} kpc")
    print(f"      p = {result['p']:.3f}")
    print(f"      n = {result['n']:.3f}")
    print(f"      A = {result['A']:.3f}")
    print(f"      RMS = {result['rms']:.4f} (relative: {result['relative_rms']:.2%})")
    
    return result

def test_sparc_mapping(galaxy_name: str = "NGC2403"):
    """Test mapping for a SPARC galaxy."""
    from test_sparc_coherence import load_rotmod
    
    print(f"\nTesting SPARC mapping: {galaxy_name}...")
    
    rotmod_path = Path(f"data/Rotmod_LTG/{galaxy_name}_rotmod.dat")
    if not rotmod_path.exists():
        print(f"  Galaxy {galaxy_name} not found, skipping")
        return None
    
    df = load_rotmod(str(rotmod_path))
    R = df["R_kpc"].values
    V_gr = df["V_gr"].values
    
    # Compute g_bar
    g_bar_kms2 = (V_gr**2) / (R * 1e3)
    G_msun_kpc_km2_s2 = 4.302e-6
    rho_bar_msun_pc3 = g_bar_kms2 / (G_msun_kpc_km2_s2 * R * 1e3) * 1e-9
    
    # Get sigma_v
    summary = pd.read_csv("data/sparc/sparc_combined.csv")
    galaxy_row = summary[summary["galaxy_name"] == galaxy_name]
    if len(galaxy_row) > 0:
        sigma_v = galaxy_row.iloc[0]["sigma_velocity"]
    else:
        sigma_v = 20.0
    
    # Load fiducial params
    fiducial_path = Path("time-coherence/time_coherence_fiducial.json")
    if fiducial_path.exists():
        with open(fiducial_path, "r") as f:
            fiducial = json.load(f)
    else:
        fiducial = {"alpha_length": 0.037, "beta_sigma": 1.5, "backreaction_cap": 10.0,
                    "A_global": 1.0, "p": 0.757, "n_coh": 0.5}
    
    # Compute theory kernel
    K_theory = compute_coherence_kernel(
        R_kpc=R,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v,
        A_global=fiducial["A_global"],
        p=fiducial["p"],
        n_coh=fiducial["n_coh"],
        method="galaxy",
        rho_bar_msun_pc3=rho_bar_msun_pc3,
        tau_geom_method="tidal",
        alpha_length=fiducial["alpha_length"],
        beta_sigma=fiducial["beta_sigma"],
        backreaction_cap=fiducial.get("backreaction_cap"),
    )
    
    # Fit Burr-XII
    result = fit_burr_xii_to_theory(R, K_theory)
    
    print(f"  {galaxy_name}:")
    print(f"    Theory: ell_coh ~ {np.mean(R[np.argmax(K_theory)]):.2f} kpc (peak)")
    print(f"    Burr-XII fit:")
    print(f"      ell_0 = {result['ell_0_kpc']:.2f} kpc")
    print(f"      p = {result['p']:.3f}")
    print(f"      n = {result['n']:.3f}")
    print(f"      A = {result['A']:.3f}")
    print(f"      RMS = {result['rms']:.4f} (relative: {result['relative_rms']:.2%})")
    
    return result

def main():
    """Run mapping analysis."""
    print("=" * 80)
    print("MAPPING TIME-COHERENCE KERNEL TO BURR-XII EMPIRICAL FORM")
    print("=" * 80)
    
    results = {}
    
    # Test MW
    results["MW"] = test_mw_mapping()
    
    # Test sample SPARC galaxies
    test_galaxies = ["NGC2403", "NGC5055", "DDO154"]
    for galaxy in test_galaxies:
        result = test_sparc_mapping(galaxy)
        if result:
            results[galaxy] = result
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nBurr-XII fits to theory kernel:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  ell_0 = {result['ell_0_kpc']:.2f} kpc")
        print(f"  p = {result['p']:.3f}")
        print(f"  n = {result['n']:.3f}")
        print(f"  Relative RMS = {result['relative_rms']:.2%}")
    
    # Save results
    output_path = Path("time-coherence/burr_xii_mapping_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

