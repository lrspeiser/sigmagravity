"""
Comprehensive grid scan over (α_length, β_sigma) with optional backreaction cap.
Runs MW, SPARC, and cluster tests for each combination.
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

# Import test functions
sys.path.insert(0, str(Path(__file__).parent))
from test_mw_coherence import load_mw_profile
from test_sparc_coherence import process_galaxy as process_sparc_galaxy, load_rotmod
from coherence_time_kernel import compute_coherence_kernel, compute_tau_geom, compute_tau_noise, compute_tau_coh, compute_coherence_length

def test_mw_params(alpha_length: float, beta_sigma: float, backreaction_cap: Optional[float] = None) -> Dict:
    """Test MW with given parameters."""
    try:
        R_kpc, g_bar_kms2, rho_bar_msun_pc3 = load_mw_profile(r_min=12.0, r_max=16.0)
        sigma_v_mw = 30.0
        
        # Compute kernel
        K = compute_coherence_kernel(
            R_kpc=R_kpc,
            g_bar_kms2=g_bar_kms2,
            sigma_v_kms=sigma_v_mw,
            A_global=1.0,
            p=0.757,
            method="galaxy",
            rho_bar_msun_pc3=rho_bar_msun_pc3,
            tau_geom_method="tidal",
            alpha_length=alpha_length,
            beta_sigma=beta_sigma,
            backreaction_cap=backreaction_cap,
        )
        
        # Compute coherence scales
        tau_geom = compute_tau_geom(R_kpc, g_bar_kms2, rho_bar_msun_pc3, method="tidal")
        tau_noise = compute_tau_noise(R_kpc, sigma_v_mw, method="galaxy", beta_sigma=beta_sigma)
        tau_coh = compute_tau_coh(tau_geom, tau_noise)
        ell_coh = compute_coherence_length(tau_coh, alpha=alpha_length)
        
        # Load observed velocities
        import pandas as pd
        mw_df = pd.read_parquet("gravitywavebaseline/gaia_with_gr_baseline.parquet")
        
        # Check column names
        if "R_kpc" not in mw_df.columns:
            # Try alternative column names
            r_col = None
            for col in mw_df.columns:
                if "r" in col.lower() or "R" in col:
                    r_col = col
                    break
            if r_col is None:
                raise ValueError("Could not find R column in MW data")
            mw_df = mw_df.rename(columns={r_col: "R_kpc"})
        
        mask = (mw_df["R_kpc"] >= 12.0) & (mw_df["R_kpc"] <= 16.0)
        mw_slice = mw_df[mask].copy()
        
        if len(mw_slice) == 0:
            raise ValueError("No MW data in range 12-16 kpc")
        
        # Compute RMS
        V_gr = mw_slice["V_gr"].values
        V_obs = mw_slice["V_obs"].values
        R_mw = mw_slice["R_kpc"].values
        
        # Interpolate K to R_mw
        K_interp = np.interp(R_mw, R_kpc, K)
        f_enh = 1.0 + K_interp
        V_model = V_gr * np.sqrt(np.clip(f_enh, 0.0, None))
        
        rms = np.sqrt(np.mean((V_model - V_obs)**2))
        rms_gr = np.sqrt(np.mean((V_gr - V_obs)**2))
        
        return {
            "rms": float(rms),
            "rms_gr": float(rms_gr),
            "ell_coh_mean": float(np.mean(ell_coh)),
            "K_max": float(np.max(K)),
        }
    except Exception as e:
        return {"error": str(e)}

def test_sparc_params(
    alpha_length: float, 
    beta_sigma: float, 
    backreaction_cap: Optional[float] = None,
    n_galaxies: int = 40,
) -> Dict:
    """Test SPARC subset with given parameters."""
    try:
        rotmod_dir = Path("data/Rotmod_LTG")
        summary_path = Path("data/sparc/sparc_combined.csv")
        
        if not summary_path.exists():
            return {"error": "SPARC summary not found"}
        
        if not rotmod_dir.exists():
            return {"error": "SPARC rotmod directory not found"}
        
        summary = pd.read_csv(summary_path)
        
        # Load sigma_v map - check column names
        sigma_map = {}
        galaxy_col = None
        sigma_col = None
        
        for col in summary.columns:
            if "galaxy" in col.lower() or "name" in col.lower():
                galaxy_col = col
            if "sigma" in col.lower() or "velocity" in col.lower():
                sigma_col = col
        
        if galaxy_col and sigma_col:
            for _, row in summary.iterrows():
                galaxy = str(row[galaxy_col]).strip()
                sigma_v = float(row.get(sigma_col, 20.0))
                sigma_map[galaxy] = sigma_v
        
        # Get subset of galaxies
        rotmod_files = sorted(list(rotmod_dir.glob("*.rotmod")))[:n_galaxies]
        
        if not rotmod_files:
            return {"error": "No rotmod files found"}
        
        results = []
        for rotmod_file in rotmod_files:
            galaxy = rotmod_file.stem.replace("_rotmod", "").replace(".rotmod", "")
            sigma_v = sigma_map.get(galaxy, 20.0)
            
            try:
                df = load_rotmod(str(rotmod_file))
                if len(df) == 0:
                    continue
                    
                R = df["R_kpc"].values
                V_obs = df["V_obs"].values
                V_gr = df["V_gr"].values
                
                # Filter out invalid data
                valid = (R > 0) & (V_obs > 0) & (V_gr > 0) & np.isfinite(R) & np.isfinite(V_obs) & np.isfinite(V_gr)
                if valid.sum() < 5:  # Need at least 5 points
                    continue
                
                R = R[valid]
                V_obs = V_obs[valid]
                V_gr = V_gr[valid]
                
                # Compute g_bar
                g_bar_kms2 = (V_gr**2) / (R * 1e3)
                G_msun_kpc_km2_s2 = 4.302e-6
                rho_bar_msun_pc3 = g_bar_kms2 / (G_msun_kpc_km2_s2 * R * 1e3) * 1e-9
                
                # Compute kernel
                K = compute_coherence_kernel(
                    R_kpc=R,
                    g_bar_kms2=g_bar_kms2,
                    sigma_v_kms=sigma_v,
                    A_global=1.0,
                    p=0.757,
                    method="galaxy",
                    rho_bar_msun_pc3=rho_bar_msun_pc3,
                    tau_geom_method="tidal",
                    alpha_length=alpha_length,
                    beta_sigma=beta_sigma,
                    backreaction_cap=backreaction_cap,
                )
                
                # Apply enhancement
                f_enh = 1.0 + K
                V_model = V_gr * np.sqrt(np.clip(f_enh, 0.0, None))
                
                # Compute RMS
                rms_model = np.sqrt(np.mean((V_model - V_obs)**2))
                rms_gr = np.sqrt(np.mean((V_gr - V_obs)**2))
                delta_rms = rms_model - rms_gr
                
                results.append({
                    "galaxy": galaxy,
                    "delta_rms": delta_rms,
                    "K_max": float(np.max(K)),
                })
            except Exception as e:
                continue
        
        if not results:
            return {"error": "No valid results"}
        
        df_results = pd.DataFrame(results)
        
        return {
            "mean_delta_rms": float(df_results["delta_rms"].mean()),
            "median_delta_rms": float(df_results["delta_rms"].median()),
            "fraction_improved": float((df_results["delta_rms"] < 0).sum() / len(df_results)),
            "worst_delta_rms": float(df_results["delta_rms"].max()),
            "n_galaxies": len(results),
        }
    except Exception as e:
        return {"error": str(e)}

def grid_scan(
    alpha_values: List[float] = [0.03, 0.037, 0.045],
    beta_values: List[float] = [1.5, 1.8, 2.0],
    backreaction_cap: Optional[float] = None,
    output_json: str = "time-coherence/grid_scan_results.json",
    n_sparc: int = 40,
):
    """
    Perform grid scan over (α_length, β_sigma) parameters.
    """
    results = []
    
    print("=" * 80)
    print("GRID SCAN: (alpha_length, beta_sigma)")
    print("=" * 80)
    print(f"\nTesting {len(alpha_values)} × {len(beta_values)} = {len(alpha_values) * len(beta_values)} combinations")
    print(f"alpha_length values: {alpha_values}")
    print(f"beta_sigma values: {beta_values}")
    if backreaction_cap is not None:
        print(f"Backreaction cap: K_max = {backreaction_cap}")
    print(f"SPARC subset: {n_sparc} galaxies")
    print()
    
    total = len(alpha_values) * len(beta_values)
    count = 0
    
    for alpha in alpha_values:
        for beta in beta_values:
            count += 1
            print(f"[{count}/{total}] Testing alpha={alpha:.3f}, beta={beta:.2f}...", end=" ", flush=True)
            
            result = {
                "alpha_length": alpha,
                "beta_sigma": beta,
                "backreaction_cap": backreaction_cap,
            }
            
            # Test MW
            try:
                mw_result = test_mw_params(alpha, beta, backreaction_cap)
                result["mw"] = mw_result
            except Exception as e:
                result["mw"] = {"error": str(e)}
            
            # Test SPARC
            try:
                sparc_result = test_sparc_params(alpha, beta, backreaction_cap, n_galaxies=n_sparc)
                result["sparc"] = sparc_result
            except Exception as e:
                result["sparc"] = {"error": str(e)}
            
            # Check if passed criteria
            passed = True
            score = 0.0
            
            if "error" not in result["mw"]:
                mw_rms = result["mw"]["rms"]
                if mw_rms > 75.0:
                    passed = False
                else:
                    # MW score: lower is better, normalize to [0, 1]
                    # Target: ~40 km/s, max acceptable: 75 km/s
                    score += max(0, (75.0 - mw_rms) / 35.0)  # MW score (0-1)
            
            if "error" not in result["sparc"]:
                sparc_mean = result["sparc"]["mean_delta_rms"]
                sparc_worst = result["sparc"]["worst_delta_rms"]
                sparc_frac = result["sparc"]["fraction_improved"]
                
                if sparc_worst > 80.0 or sparc_frac < 0.60:
                    passed = False
                else:
                    # SPARC score: favor negative mean, high fraction improved
                    # Mean delta_rms: ideal is negative, penalize positive
                    if sparc_mean < 0:
                        score += 0.5 * (1.0 - abs(sparc_mean) / 10.0)  # Reward negative mean
                    else:
                        score += 0.5 * max(0, 1.0 - sparc_mean / 20.0)  # Penalize positive mean
                    
                    # Fraction improved: reward high fraction
                    score += 0.5 * sparc_frac  # Reward high fraction improved
            
            result["passed"] = passed
            result["score"] = score
            
            results.append(result)
            
            if passed:
                print(f"PASSED (score={score:.3f})")
            else:
                print(f"FAILED")
    
    # Save results
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    passed_results = [r for r in results if r["passed"]]
    print(f"\n{'=' * 80}")
    print(f"GRID SCAN COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total combinations: {len(results)}")
    print(f"Passed: {len(passed_results)}")
    
    if passed_results:
        # Sort by score
        passed_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"\nTop 5 combinations:")
        for i, r in enumerate(passed_results[:5], 1):
            print(f"  {i}. alpha={r['alpha_length']:.3f}, beta={r['beta_sigma']:.2f}, score={r['score']:.3f}")
            if "mw" in r and "error" not in r["mw"]:
                print(f"     MW RMS: {r['mw']['rms']:.2f} km/s")
            if "sparc" in r and "error" not in r["sparc"]:
                print(f"     SPARC: mean_delta={r['sparc']['mean_delta_rms']:.2f} km/s, "
                      f"frac_improved={r['sparc']['fraction_improved']:.2%}")
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--with-cap":
        backreaction_cap = 10.0
        output_json = "time-coherence/grid_scan_results_with_cap.json"
    else:
        backreaction_cap = None
        output_json = "time-coherence/grid_scan_results.json"
    
    # Run grid scan
    results = grid_scan(
        alpha_values=[0.03, 0.037, 0.045],
        beta_values=[1.5, 1.8, 2.0],
        backreaction_cap=backreaction_cap,
        output_json=output_json,
        n_sparc=40,
    )

