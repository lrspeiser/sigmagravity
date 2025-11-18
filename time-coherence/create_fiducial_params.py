"""
Create fiducial hyperparameter JSON from grid scan results.
"""

import json
from pathlib import Path

def create_fiducial_from_grid_scan(
    grid_scan_json: str = "time-coherence/grid_scan_results.json",
    output_json: str = "time-coherence/time_coherence_fiducial.json",
    backreaction_cap: float = 10.0,
):
    """
    Select best parameters from grid scan and create fiducial JSON.
    """
    with open(grid_scan_json, "r") as f:
        results = json.load(f)
    
    # Filter passed results
    passed = [r for r in results if r.get("passed", False)]
    
    if not passed:
        print("Warning: No passed results in grid scan. Using default parameters.")
        fiducial = {
            "A_global": 1.0,
            "p": 0.757,
            "n_coh": 0.5,
            "delta_R_kpc": 0.1,
            "tau_geom_method": "tidal",
            "alpha_length": 0.037,
            "beta_sigma": 1.5,
            "alpha_geom": 1.0,
            "backreaction_cap": backreaction_cap,
            "source": "default",
        }
    else:
        # Sort by score
        passed.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        best = passed[0]
        
        fiducial = {
            "A_global": 1.0,  # Default, can be tuned separately
            "p": 0.757,
            "n_coh": 0.5,
            "delta_R_kpc": 0.1,  # Default, can be tuned separately
            "tau_geom_method": "tidal",
            "alpha_length": best["alpha_length"],
            "beta_sigma": best["beta_sigma"],
            "alpha_geom": 1.0,
            "backreaction_cap": backreaction_cap,
            "source": "grid_scan",
            "grid_scan_score": best.get("score", 0.0),
            "mw_rms": best.get("mw", {}).get("rms", None),
            "sparc_mean_delta_rms": best.get("sparc", {}).get("mean_delta_rms", None),
            "sparc_fraction_improved": best.get("sparc", {}).get("fraction_improved", None),
        }
    
    # Save
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fiducial, f, indent=2)
    
    print("=" * 80)
    print("FIDUCIAL PARAMETERS CREATED")
    print("=" * 80)
    print(f"\nSaved to: {output_path}")
    print(f"\nParameters:")
    for key, value in fiducial.items():
        if key not in ["source", "grid_scan_score", "mw_rms", "sparc_mean_delta_rms", "sparc_fraction_improved"]:
            print(f"  {key}: {value}")
    
    if "grid_scan_score" in fiducial:
        print(f"\nPerformance:")
        print(f"  Grid scan score: {fiducial['grid_scan_score']:.3f}")
        if fiducial.get("mw_rms") is not None:
            print(f"  MW RMS: {fiducial['mw_rms']:.2f} km/s")
        if fiducial.get("sparc_mean_delta_rms") is not None:
            print(f"  SPARC mean delta_RMS: {fiducial['sparc_mean_delta_rms']:.2f} km/s")
        if fiducial.get("sparc_fraction_improved") is not None:
            print(f"  SPARC fraction improved: {fiducial['sparc_fraction_improved']:.2%}")
    
    return fiducial


if __name__ == "__main__":
    fiducial = create_fiducial_from_grid_scan(backreaction_cap=10.0)

