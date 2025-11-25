"""
RAR Scatter Comparison: With vs Without Winding
================================================

Runs the paper's official RAR scatter calculation using the validation suite,
comparing results with and without the spiral winding gate.

Author: Leonard Speiser
Date: 2025-11-25
"""

import sys
import os
from pathlib import Path

# Add spiral folder to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from validation_suite_winding import ValidationSuite
from path_spectrum_kernel_winding import PathSpectrumKernel, PathSpectrumHyperparams


def run_rar_test(use_winding: bool, N_crit: float = 10.0):
    """Run RAR test with specified winding settings."""
    
    print("=" * 80)
    if use_winding:
        print(f"RAR TEST: WITH SPIRAL WINDING (N_crit={N_crit})")
    else:
        print("RAR TEST: WITHOUT WINDING (original Σ-Gravity)")
    print("=" * 80)
    
    # Create validation suite
    output_dir = SCRIPT_DIR / "outputs" / ("winding" if use_winding else "no_winding")
    suite = ValidationSuite(output_dir, load_sparc=True)
    
    # Use paper's tuned hyperparameters (from paper_release/config/hyperparams_track2.json)
    hp = PathSpectrumHyperparams(
        L_0=4.993,
        beta_bulge=1.759,
        alpha_shear=0.149,
        gamma_bar=0.0,
        A_0=1.1,
        p=0.75,
        n_coh=0.5,
        g_dagger=1.2e-10,
        # Winding settings (optimized: gentler winding preserves RAR)
        use_winding=use_winding,
        N_crit=N_crit,
        t_age=10.0,
        wind_power=1.0  # Gentler than 2.0 (tuned for RAR)
    )
    
    print(f"\nParameters (paper tuned):")
    print(f"  L_0 = {hp.L_0} kpc, A_0 = {hp.A_0}, p = {hp.p}, n_coh = {hp.n_coh}")
    print(f"  beta_bulge = {hp.beta_bulge}, alpha_shear = {hp.alpha_shear}")
    print(f"  use_winding = {hp.use_winding}")
    if use_winding:
        print(f"  N_crit = {hp.N_crit}, t_age = {hp.t_age} Gyr, wind_power = {hp.wind_power}")
    
    # Get valid data with rotation curves
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    print(f"\nGalaxies with rotation curves: {len(df_valid)}")
    
    # Run RAR test using the correct API: compute_btfr_rar(df, hp_override)
    btfr_scatter, rar_scatter = suite.compute_btfr_rar(
        df_valid, 
        hp_override=hp
    )
    
    return rar_scatter, btfr_scatter


def main():
    print("\n" + "=" * 80)
    print("Σ-GRAVITY RAR COMPARISON: WITH vs WITHOUT SPIRAL WINDING")
    print("=" * 80)
    print("\nThis test compares RAR scatter using the paper's official methodology:")
    print("  RAR = g_bar / (1 - exp(-√(g_bar/g†)))")
    print("  Scatter = std(log₁₀(g_model) - log₁₀(g_RAR))")
    
    # Test WITHOUT winding
    rar_no_wind, btfr_no_wind = run_rar_test(use_winding=False)
    
    print("\n" + "-" * 80)
    
    # Test WITH winding (optimal N_crit=100)
    rar_wind, btfr_wind = run_rar_test(use_winding=True, N_crit=100.0)
    
    # N_crit sweep
    print("\n" + "=" * 80)
    print("N_crit SWEEP (RAR scatter, wind_power=1.0)")
    print("=" * 80)
    
    print(f"\n{'N_crit':<10} {'RAR scatter (dex)':<20}")
    print("-" * 35)
    
    best_N_crit = 100
    best_rar = rar_wind
    
    for N_crit in [30, 50, 75, 100, 150, 200]:
        rar, _ = run_rar_test(use_winding=True, N_crit=N_crit)
        marker = " ← BEST" if rar < best_rar else ""
        if rar < best_rar:
            best_rar = rar
            best_N_crit = N_crit
        print(f"{N_crit:<10} {rar:<20.4f}{marker}")
    
    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    print(f"""
┌────────────────────────────────────┬───────────────┬───────────────┐
│ Metric                             │ No Winding    │ With Winding  │
├────────────────────────────────────┼───────────────┼───────────────┤
│ RAR scatter (dex)                  │ {rar_no_wind:.4f}        │ {rar_wind:.4f}        │
│ BTFR scatter (dex)                 │ {btfr_no_wind:.4f}        │ {btfr_wind:.4f}        │
├────────────────────────────────────┼───────────────┼───────────────┤
│ Paper target                       │ 0.087 dex     │               │
│ MOND (literature)                  │ 0.10-0.13 dex │               │
└────────────────────────────────────┴───────────────┴───────────────┘
""")
    
    if rar_wind < rar_no_wind:
        improvement = (rar_no_wind - rar_wind) / rar_no_wind * 100
        print(f"✓ Winding REDUCES RAR scatter by {improvement:.1f}%")
        print(f"  {rar_no_wind:.4f} → {rar_wind:.4f} dex")
    else:
        print(f"✗ Winding does not reduce RAR scatter")
    
    if rar_wind < 0.087:
        print(f"\n✓ BEATS paper's 0.087 dex target!")
    elif rar_wind < 0.10:
        print(f"\n✓ BEATS MOND (0.10-0.13 dex)!")
    elif rar_wind < 0.15:
        print(f"\n⚠ Within acceptable range (< 0.15 dex)")
    else:
        print(f"\n✗ RAR scatter too high (> 0.15 dex)")
    
    print(f"\nOptimal N_crit = {best_N_crit} with RAR scatter = {best_rar:.4f} dex")


if __name__ == "__main__":
    main()
