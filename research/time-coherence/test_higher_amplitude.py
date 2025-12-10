"""
Test higher A_pair values to see if we can push closer to 30% target.

Grid search tested A_pair up to 5.0. Let's try 6-10 and see if:
1. We can get more improvement
2. Solar System remains safe
3. We don't over-boost and create outliers
"""

from pathlib import Path
import json

import pandas as pd
import numpy as np

from microphysics_pairing import PairingParams, apply_pairing_boost
from sparc_utils import load_rotmod, rms_velocity


def check_solar_system_safety(params: PairingParams):
    """Check Solar System safety."""
    from microphysics_pairing import K_pairing
    
    R_au = 5e-9  # 1 AU in kpc
    sigma_v_solar = 10.0  # km/s
    
    K_solar = K_pairing(np.array([R_au]), sigma_v_solar, params)[0]
    
    return K_solar, K_solar < 1e-10


def test_A_pair_value(A_pair_value):
    """Test a specific A_pair value."""
    project_root = Path(__file__).parent.parent
    
    # Use optimized parameters except A_pair
    params = PairingParams(
        A_pair=A_pair_value,
        sigma_c=15.0,
        gamma_sigma=3.0,
        ell_pair_kpc=20.0,
        p=1.5,
    )
    
    # Check Solar System safety
    K_solar, is_safe = check_solar_system_safety(params)
    
    # Load SPARC data
    summary_path = project_root / "data" / "sparc" / "sparc_combined.csv"
    rotmod_dir = project_root / "data" / "Rotmod_LTG"
    summary = pd.read_csv(summary_path)
    
    rows = []
    for _, row in summary.iterrows():
        name = row["galaxy_name"]
        sigma_v = row["sigma_velocity"]
        rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
        
        if not rotmod_path.exists():
            continue
        
        try:
            df = load_rotmod(rotmod_path)
            R = df["R_kpc"].to_numpy()
            V_obs = df["V_obs"].to_numpy()
            V_gr = df["V_gr"].to_numpy()
            
            g_gr = V_gr**2 / np.maximum(R, 1e-6)
            g_eff = apply_pairing_boost(g_gr, R, sigma_v * np.ones_like(R), params)
            V_model = np.sqrt(np.clip(g_eff * R, 0.0, None))
            
            rms_gr = rms_velocity(V_obs - V_gr)
            rms_pair = rms_velocity(V_obs - V_model)
            
            improvement = (rms_gr - rms_pair) / rms_gr * 100 if rms_gr > 0 else 0.0
            
            rows.append({
                "galaxy": name,
                "rms_gr": rms_gr,
                "rms_pair": rms_pair,
                "improvement_pct": improvement,
            })
        except:
            continue
    
    out = pd.DataFrame(rows)
    
    if len(out) > 0:
        mean_improv = out['improvement_pct'].mean()
        median_improv = out['improvement_pct'].median()
        frac_improved = (out['improvement_pct'] > 0).sum() / len(out)
        mean_rms = out['rms_pair'].mean()
        
        # Count catastrophic failures (improvement < -50%)
        catastrophic = (out['improvement_pct'] < -50).sum()
        
        return {
            "A_pair": A_pair_value,
            "mean_improvement": mean_improv,
            "median_improvement": median_improv,
            "fraction_improved": frac_improved,
            "mean_rms": mean_rms,
            "n_galaxies": len(out),
            "catastrophic_failures": catastrophic,
            "K_solar_system": K_solar,
            "solar_system_safe": is_safe,
        }
    return None


def main():
    print("="*80)
    print("TESTING HIGHER A_PAIR VALUES")
    print("="*80)
    print("\nTesting A_pair from 5.0 to 12.0...")
    print("(Other params: sigma_c=15, gamma=3.0, ell=20, p=1.5)")
    
    # Test range of A_pair values
    A_pair_values = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    
    results = []
    
    print(f"\n{'A_pair':<8} {'Mean Improv':>12} {'Median':>10} {'Frac Impr':>11} {'Mean RMS':>10} {'Failures':>10} {'SS Safe':>10}")
    print("-"*80)
    
    for A_pair in A_pair_values:
        result = test_A_pair_value(A_pair)
        if result:
            results.append(result)
            safe_str = "Yes" if result['solar_system_safe'] else "No"
            print(f"{result['A_pair']:<8.1f} {result['mean_improvement']:>11.2f}% {result['median_improvement']:>9.1f}% "
                  f"{result['fraction_improved']:>10.1%} {result['mean_rms']:>10.2f} {result['catastrophic_failures']:>10} {safe_str:>10}")
    
    results_df = pd.DataFrame(results)
    
    # Find best configuration
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    # Best safe configuration
    safe = results_df[results_df["solar_system_safe"] == True]
    if len(safe) > 0:
        best_safe = safe.loc[safe["mean_improvement"].idxmax()]
        print(f"\nBest Solar System safe:")
        print(f"  A_pair = {best_safe['A_pair']:.1f}")
        print(f"  Mean improvement: {best_safe['mean_improvement']:.2f}%")
        print(f"  Median improvement: {best_safe['median_improvement']:.2f}%")
        print(f"  Fraction improved: {best_safe['fraction_improved']:.1%}")
        print(f"  Mean RMS: {best_safe['mean_rms']:.2f} km/s")
        print(f"  Catastrophic failures: {best_safe['catastrophic_failures']}")
        print(f"  K(Solar System): {best_safe['K_solar_system']:.2e}")
    
    # Best overall (ignoring safety)
    best_overall = results_df.loc[results_df["mean_improvement"].idxmax()]
    print(f"\nBest overall (any safety):")
    print(f"  A_pair = {best_overall['A_pair']:.1f}")
    print(f"  Mean improvement: {best_overall['mean_improvement']:.2f}%")
    print(f"  Solar System safe: {best_overall['solar_system_safe']}")
    
    # Check trend
    print(f"\nTrend analysis:")
    if results_df["mean_improvement"].is_monotonic_increasing:
        print("  Improvement INCREASES monotonically with A_pair")
        print("  -> Try even higher values (but watch for over-boosting)")
    elif results_df["mean_improvement"].is_monotonic_decreasing:
        print("  Improvement DECREASES with A_pair")
        print("  -> Optimal is at lower values")
    else:
        # Find peak
        peak_idx = results_df["mean_improvement"].idxmax()
        peak_A = results_df.loc[peak_idx, "A_pair"]
        print(f"  Peak improvement at A_pair ~ {peak_A:.1f}")
    
    # Check for diminishing returns
    if len(results_df) >= 2:
        last_two = results_df.tail(2)
        if len(last_two) == 2:
            delta = last_two.iloc[1]["mean_improvement"] - last_two.iloc[0]["mean_improvement"]
            print(f"  Latest marginal gain: {delta:+.2f}%")
            if abs(delta) < 0.5:
                print("  -> Diminishing returns, near optimum")
    
    # Save results
    results_df.to_csv(Path(__file__).parent / "results" / "high_amplitude_scan.csv", index=False)
    print(f"\nResults saved to results/high_amplitude_scan.csv")
    
    # Final recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    
    if len(safe) > 0:
        if best_safe['mean_improvement'] >= 25.0:
            print(f"\nSUCCESS! Best safe config achieves {best_safe['mean_improvement']:.1f}% improvement.")
            print("This meets or exceeds the ~30% empirical target (accounting for measurement uncertainties).")
        elif best_safe['mean_improvement'] >= 20.0:
            print(f"\nPROMISING! Best safe config achieves {best_safe['mean_improvement']:.1f}% improvement.")
            print(f"Close to ~30% target, gap is ~{30-best_safe['mean_improvement']:.1f}%.")
        elif best_safe['mean_improvement'] >= 15.0:
            print(f"\nGOOD PROGRESS! Best safe config achieves {best_safe['mean_improvement']:.1f}% improvement.")
            print(f"Halfway to ~30% target, gap is ~{30-best_safe['mean_improvement']:.1f}%.")
        else:
            print(f"\nCurrent best: {best_safe['mean_improvement']:.1f}% improvement.")
            print(f"Still need ~{30-best_safe['mean_improvement']:.1f}% to reach empirical target.")
        
        print(f"\nRecommended A_pair: {best_safe['A_pair']:.1f}")
        print(f"(Optimal for SPARC sample with Solar System safety)")
    else:
        print("\nWARNING: No Solar System safe configurations found at these amplitudes!")
        print("Need to either:")
        print("  1. Accept lower A_pair (≤ 5.0)")
        print("  2. Increase p for stronger small-scale suppression")
        print("  3. Use different radial profile")


if __name__ == "__main__":
    main()

