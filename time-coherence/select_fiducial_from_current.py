"""
Select fiducial parameters based on current best results.
Uses existing test results and adds backreaction cap testing.
"""

import json
from pathlib import Path

def create_fiducial_params():
    """
    Create fiducial parameters based on current best results.
    
    Current best:
    - alpha_length = 0.037 (brings scales to ~1-2 kpc)
    - beta_sigma = 1.5 (stronger σ_v suppression)
    - MW RMS = 66.40 km/s (good, target ~40-70 km/s)
    - SPARC mean delta_RMS = 5.906 km/s (needs improvement with cap)
    """
    
    fiducial = {
        "A_global": 1.0,
        "p": 0.757,
        "n_coh": 0.5,
        "delta_R_kpc": 0.1,
        "tau_geom_method": "tidal",
        "alpha_length": 0.037,
        "beta_sigma": 1.5,
        "alpha_geom": 1.0,
        "backreaction_cap": 10.0,
        "source": "current_best_with_cap",
        "notes": "Based on current test results with backreaction cap to stabilize SPARC outliers",
        "performance": {
            "mw_rms_target": "40-70 km/s",
            "sparc_mean_delta_rms_target": "<= 0 km/s",
            "sparc_fraction_improved_target": ">= 70%",
            "cluster_boost_range": "1.5× - 10×",
        }
    }
    
    # Save
    output_path = Path("time-coherence/time_coherence_fiducial.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fiducial, f, indent=2)
    
    print("=" * 80)
    print("FIDUCIAL PARAMETERS CREATED")
    print("=" * 80)
    print(f"\nSaved to: {output_path}")
    print(f"\nParameters:")
    for key, value in fiducial.items():
        if key not in ["source", "notes", "performance"]:
            print(f"  {key}: {value}")
    
    print(f"\n{fiducial.get('notes', '')}")
    print(f"\nPerformance targets:")
    for key, value in fiducial.get("performance", {}).items():
        print(f"  {key}: {value}")
    
    return fiducial


if __name__ == "__main__":
    fiducial = create_fiducial_params()

