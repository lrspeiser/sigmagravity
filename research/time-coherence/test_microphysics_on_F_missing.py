"""
Test microphysics models against F_missing instead of K_total.

This refits coherence models to explain F_missing, which is the
"missing factor" after accounting for roughness.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Try to import coherence models
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "coherence tests"))
    from coherence_models import (
        metric_resonance,
        graviton_pairing,
        path_interference,
        vacuum_condensation,
        entanglement,
    )
    COHERENCE_MODELS_AVAILABLE = True
except ImportError:
    COHERENCE_MODELS_AVAILABLE = False
    print("Warning: coherence_models not available, using placeholder")


def load_data():
    """Load SPARC data with F_missing."""
    roughness_df = pd.read_csv("time-coherence/sparc_roughness_amplitude.csv")
    summary_df = pd.read_csv("data/sparc/sparc_combined.csv")
    
    # Merge
    galaxy_col_rough = "galaxy"
    galaxy_col_summary = None
    for col in ["galaxy_name", "Galaxy", "name", "gal"]:
        if col in summary_df.columns:
            galaxy_col_summary = col
            break
    
    if galaxy_col_summary is None:
        raise ValueError("Could not find galaxy name column")
    
    merged = roughness_df.merge(
        summary_df,
        left_on=galaxy_col_rough,
        right_on=galaxy_col_summary,
        how="inner",
    )
    
    return merged


def test_model_on_F_missing(model_name, model_func, df, param_bounds):
    """
    Test a microphysics model against F_missing.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    model_func : callable
        Model function that takes (R, g_bar, lambda_gw, sigma_v, params)
    df : pd.DataFrame
        DataFrame with galaxy data
    param_bounds : list
        Bounds for parameters [(min, max), ...]
    """
    print(f"\nTesting {model_name}...")
    
    # For each galaxy, fit model to F_missing
    results = []
    
    # Sample a few galaxies for testing
    test_galaxies = df.head(20)  # Test on first 20 galaxies
    
    for idx, row in test_galaxies.iterrows():
        if pd.isna(row["F_missing"]) or row["F_missing"] <= 0:
            continue
        
        F_target = row["F_missing"]
        sigma_v = row["sigma_velocity"]
        R_d = row.get("R_disk", 20.0)
        
        # Create mock R, g_bar, lambda_gw for testing
        # In real implementation, would load actual rotation curve data
        R = np.logspace(0, 2, 50)  # 1-100 kpc
        g_bar = 1e-4 / R**1.5  # Mock acceleration profile
        lambda_gw = 2 * np.pi * R  # Mock orbital wavelength
        
        def objective(params):
            try:
                # Model should return enhancement factor
                K_model = model_func(R, g_bar, lambda_gw, sigma_v, params)
                # Convert to F_missing prediction
                # F_missing is system-level, so use mean
                F_pred = float(np.mean(K_model))
                
                # Objective: minimize difference from target
                error = (F_pred - F_target) ** 2
                return error
            except Exception:
                return 1e10
        
        try:
            result = differential_evolution(
                objective,
                bounds=param_bounds,
                maxiter=50,
                popsize=5,
                seed=42,
            )
            
            if result.success:
                F_pred = objective(result.x)
                results.append(
                    {
                        "galaxy": row["galaxy"],
                        "F_target": float(F_target),
                        "F_pred": float(F_pred),
                        "params": result.x.tolist(),
                        "error": float(result.fun),
                    }
                )
        except Exception as e:
            print(f"  Error fitting {row['galaxy']}: {e}")
            continue
    
    if not results:
        print(f"  No successful fits")
        return None
    
    # Aggregate statistics
    results_df = pd.DataFrame(results)
    mean_error = results_df["error"].mean()
    mean_corr = np.corrcoef(results_df["F_target"], results_df["F_pred"])[0, 1] if len(results_df) > 1 else 0.0
    
    print(f"  N galaxies: {len(results_df)}")
    print(f"  Mean error: {mean_error:.3f}")
    print(f"  Correlation: {mean_corr:.3f}")
    
    return {
        "model": model_name,
        "n_galaxies": len(results_df),
        "mean_error": float(mean_error),
        "correlation": float(mean_corr),
        "results": results,
    }


def main():
    print("=" * 80)
    print("TESTING MICROPHYSICS MODELS ON F_MISSING")
    print("=" * 80)
    
    if not COHERENCE_MODELS_AVAILABLE:
        print("\nWarning: coherence_models not available")
        print("This script requires coherence_models.py from coherence tests/")
        print("Skipping microphysics model testing")
        return
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} galaxies")
    
    # Filter valid
    valid = (
        df["F_missing"].notna()
        & (df["F_missing"] > 0)
        & df["sigma_velocity"].notna()
        & (df["sigma_velocity"] > 0)
    )
    df_valid = df[valid].copy()
    
    print(f"Valid galaxies: {len(df_valid)}")
    
    # Test each model
    all_results = {}
    
    # Define parameter bounds for each model
    # These are placeholders - adjust based on actual model signatures
    model_configs = {
        "metric_resonance": {
            "func": metric_resonance,
            "bounds": [(0.1, 10.0), (0.1, 5.0)],  # Example bounds
        },
        "graviton_pairing": {
            "func": graviton_pairing,
            "bounds": [(0.1, 10.0), (0.1, 5.0)],
        },
        # Add other models as needed
    }
    
    for model_name, config in model_configs.items():
        try:
            result = test_model_on_F_missing(
                model_name,
                config["func"],
                df_valid,
                config["bounds"],
            )
            if result:
                all_results[model_name] = result
        except Exception as e:
            print(f"\nError testing {model_name}: {e}")
            continue
    
    # Save results
    if all_results:
        outpath = Path("time-coherence/microphysics_F_missing_results.json")
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {outpath}")
        
        # Find best model
        best = min(all_results.items(), key=lambda x: x[1]["mean_error"])
        print(f"\nBest model: {best[0]}")
        print(f"  Mean error: {best[1]['mean_error']:.3f}")
        print(f"  Correlation: {best[1]['correlation']:.3f}")
    else:
        print("\nNo successful model fits")


if __name__ == "__main__":
    main()

