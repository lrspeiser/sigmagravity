"""
Tune A_0 to compensate for winding suppression
==============================================

The winding gate suppresses the boost factor, so we need to increase A_0 to compensate.
This script finds the optimal A_0 for different N_crit values.
"""

import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from validation_suite_winding import ValidationSuite
from path_spectrum_kernel_winding import PathSpectrumKernel, PathSpectrumHyperparams


def test_rar(A_0: float, N_crit: float, use_winding: bool = True, verbose: bool = False):
    """Quick RAR test with specified parameters."""
    output_dir = SCRIPT_DIR / "outputs" / "tuning"
    suite = ValidationSuite(output_dir, load_sparc=True)
    
    hp = PathSpectrumHyperparams(
        L_0=4.993,
        beta_bulge=1.759,
        alpha_shear=0.149,
        gamma_bar=0.0,
        A_0=A_0,
        p=0.75,
        n_coh=0.5,
        g_dagger=1.2e-10,
        use_winding=use_winding,
        N_crit=N_crit,
        t_age=10.0,
        wind_power=2.0
    )
    
    df_valid = suite.sparc_data[suite.sparc_data['r_all'].notna()].copy()
    
    # Suppress output during tuning
    import io
    import contextlib
    
    if not verbose:
        with contextlib.redirect_stdout(io.StringIO()):
            btfr, rar = suite.compute_btfr_rar(df_valid, hp_override=hp)
    else:
        btfr, rar = suite.compute_btfr_rar(df_valid, hp_override=hp)
    
    return rar


def test_rar_full(A_0: float, N_crit: float, wind_power: float = 2.0, use_winding: bool = True):
    """Quick RAR test with all winding parameters."""
    output_dir = SCRIPT_DIR / "outputs" / "tuning"
    suite = ValidationSuite(output_dir, load_sparc=True)
    
    hp = PathSpectrumHyperparams(
        L_0=4.993,
        beta_bulge=1.759,
        alpha_shear=0.149,
        gamma_bar=0.0,
        A_0=A_0,
        p=0.75,
        n_coh=0.5,
        g_dagger=1.2e-10,
        use_winding=use_winding,
        N_crit=N_crit,
        t_age=10.0,
        wind_power=wind_power
    )
    
    df_valid = suite.sparc_data[suite.sparc_data['r_all'].notna()].copy()
    
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        btfr, rar = suite.compute_btfr_rar(df_valid, hp_override=hp)
    
    return rar


def main():
    print("=" * 80)
    print("TUNING WINDING PARAMETERS FOR OPTIMAL RAR")
    print("=" * 80)
    
    # Get baseline without winding
    print("\nBaseline (no winding, A_0=1.1):")
    baseline_rar = test_rar(A_0=1.1, N_crit=10, use_winding=False)
    print(f"  RAR scatter = {baseline_rar:.4f} dex")
    
    # Try gentler winding: higher N_crit, lower wind_power
    print("\n" + "=" * 60)
    print("TESTING GENTLER WINDING (lower wind_power, higher N_crit)")
    print("=" * 60)
    
    wind_powers = [1.0, 1.5, 2.0]
    N_crit_values = [30, 50, 100]
    
    results = {}
    
    print(f"\n{'wind_power':<12} {'N_crit':<10} {'A_0':<8} {'RAR scatter':<15} {'vs baseline':<15}")
    print("-" * 60)
    
    best_params = None
    best_rar = 999
    
    for wp in wind_powers:
        for N_crit in N_crit_values:
            # For gentler winding, less A_0 compensation needed
            rar = test_rar_full(A_0=1.1, N_crit=N_crit, wind_power=wp, use_winding=True)
            diff = rar - baseline_rar
            marker = " ← BEST" if rar < best_rar else ""
            if rar < best_rar:
                best_rar = rar
                best_params = (wp, N_crit, 1.1)
            print(f"{wp:<12.1f} {N_crit:<10} {1.1:<8.1f} {rar:<15.4f} {diff:+.4f}{marker}")
            results[(wp, N_crit)] = rar
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBaseline (no winding): {baseline_rar:.4f} dex")
    print(f"Paper target: 0.087 dex")
    print(f"MOND literature: 0.10-0.13 dex")
    
    if best_params:
        wp, nc, a0 = best_params
        print(f"\nBest with winding: wind_power={wp}, N_crit={nc}, A_0={a0}")
        print(f"  RAR scatter = {best_rar:.4f} dex")
        print(f"  vs baseline: {best_rar - baseline_rar:+.4f} dex")
        
        if best_rar < baseline_rar:
            print(f"\n✓ BEATS baseline!")
        else:
            print(f"\n✗ Still worse than baseline")


if __name__ == "__main__":
    main()
