"""
QUICK IMPROVEMENT TESTER

Test individual improvements one at a time to see their effect.
Much faster than full optimization - just evaluates with fixed parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
G_KPC = 4.498e-12

# ============================================================================
# ANALYTICAL COMPONENTS (copy from improved script)
# ============================================================================

def hernquist_bulge(R, M_bulge=1.5e10, a_bulge=0.7):
    """Analytical bulge contribution."""
    v_squared = G_KPC * M_bulge * R / (R + a_bulge)**2
    return np.sqrt(v_squared)

def exponential_gas(R, M_gas=1e10, R_gas=7.0):
    """Analytical gas disk contribution."""
    from scipy.special import i0, i1, k0, k1
    y = R / (2 * R_gas)
    v_squared = 4 * np.pi * G_KPC * M_gas * R**2 / (2 * R_gas)**2 * \
                (i0(y) * k0(y) - i1(y) * k1(y))
    return np.sqrt(np.maximum(v_squared, 0))

# ============================================================================
# QUICK TESTER
# ============================================================================

def quick_test_improvements(gaia_file='gravitywavebaseline/gaia_with_periods.parquet'):
    """
    Quick test of improvements without full optimization.
    
    Uses best parameters from first run (A=2.08, lambda_0=9.94, alpha=2.78)
    """
    
    print("="*80)
    print("QUICK IMPROVEMENT TESTER")
    print("="*80)
    print("\nThis quickly evaluates improvements using fixed parameters")
    print("(No optimization - just shows what each improvement contributes)")
    
    # Load data
    print("\nLoading data...")
    gaia = pd.read_parquet(gaia_file)
    print(f"  {len(gaia):,} stars loaded")
    
    # Setup test points
    R_test = np.linspace(4, 16, 25)
    v_target = np.ones_like(R_test) * 220.0  # Flat curve target
    
    # Best parameters from first run
    A_best = 2.08
    lambda_0_best = 9.94
    alpha_best = 2.78
    
    print(f"\nUsing best parameters from first run:")
    print(f"  A = {A_best:.2f}")
    print(f"  lambda_0 = {lambda_0_best:.2f} kpc")
    print(f"  alpha = {alpha_best:.2f}")
    
    # ========================================================================
    # TEST 1: Baseline (original result)
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 1: BASELINE (disk stars only, Jeans period)")
    print("="*80)
    
    v_baseline = compute_simple_model(
        gaia, R_test, 
        period_col='lambda_jeans',
        A=A_best, lambda_0=lambda_0_best, alpha=alpha_best,
        M_disk=5e10
    )
    
    rms_baseline = np.sqrt(np.mean((v_baseline - v_target)**2))
    print(f"\n  RMS: {rms_baseline:.1f} km/s")
    
    # ========================================================================
    # TEST 2: Add bulge
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 2: ADD BULGE")
    print("="*80)
    
    v_bulge = hernquist_bulge(R_test, M_bulge=1.5e10, a_bulge=0.7)
    v_with_bulge = np.sqrt(v_baseline**2 + v_bulge**2)
    
    rms_with_bulge = np.sqrt(np.mean((v_with_bulge - v_target)**2))
    improvement = rms_baseline - rms_with_bulge
    
    print(f"\n  Bulge contribution at R=8 kpc: {v_bulge[np.argmin(abs(R_test-8))]:.1f} km/s")
    print(f"  RMS: {rms_with_bulge:.1f} km/s")
    print(f"  Improvement: {improvement:.1f} km/s ({improvement/rms_baseline*100:.1f}%)")
    
    # ========================================================================
    # TEST 3: Add gas
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 3: ADD GAS DISK")
    print("="*80)
    
    v_gas = exponential_gas(R_test, M_gas=1e10, R_gas=7.0)
    v_with_gas = np.sqrt(v_with_bulge**2 + v_gas**2)
    
    rms_with_gas = np.sqrt(np.mean((v_with_gas - v_target)**2))
    improvement = rms_baseline - rms_with_gas
    
    print(f"\n  Gas contribution at R=8 kpc: {v_gas[np.argmin(abs(R_test-8))]:.1f} km/s")
    print(f"  RMS: {rms_with_gas:.1f} km/s")
    print(f"  Total improvement: {improvement:.1f} km/s ({improvement/rms_baseline*100:.1f}%)")
    
    # ========================================================================
    # TEST 4: Hybrid period (Jeans + Orbital)
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 4: HYBRID PERIOD (Jeans + Orbital)")
    print("="*80)
    
    # Create hybrid period
    lambda_hybrid = np.sqrt(gaia['lambda_jeans']**2 + gaia['lambda_orbital']**2)
    gaia['lambda_hybrid_test'] = lambda_hybrid
    
    v_hybrid_disk = compute_simple_model(
        gaia, R_test,
        period_col='lambda_hybrid_test',
        A=A_best, lambda_0=lambda_0_best, alpha=alpha_best,
        M_disk=5e10
    )
    
    v_hybrid_total = np.sqrt(v_hybrid_disk**2 + v_bulge**2 + v_gas**2)
    
    rms_hybrid = np.sqrt(np.mean((v_hybrid_total - v_target)**2))
    improvement = rms_baseline - rms_hybrid
    
    print(f"\n  Hybrid period median: {np.median(lambda_hybrid):.1f} kpc")
    print(f"  RMS: {rms_hybrid:.1f} km/s")
    print(f"  Total improvement: {improvement:.1f} km/s ({improvement/rms_baseline*100:.1f}%)")
    
    # ========================================================================
    # Summary plot
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOT")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Quick Improvement Test Results', fontsize=14, fontweight='bold')
    
    # Plot 1: Rotation curves
    ax = axes[0]
    ax.plot(R_test, v_target, 'k--', linewidth=2, label='Target (220 km/s)', alpha=0.7)
    ax.plot(R_test, v_baseline, 'b-', linewidth=2, label=f'Baseline (RMS={rms_baseline:.0f})')
    ax.plot(R_test, v_with_bulge, 'g-', linewidth=2, label=f'+Bulge (RMS={rms_with_bulge:.0f})')
    ax.plot(R_test, v_with_gas, 'orange', linewidth=2, label=f'+Gas (RMS={rms_with_gas:.0f})')
    ax.plot(R_test, v_hybrid_total, 'r-', linewidth=2, label=f'Hybrid period (RMS={rms_hybrid:.0f})')
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_circ [km/s]', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Rotation Curves')
    ax.set_ylim(150, 250)
    
    # Plot 2: RMS comparison
    ax = axes[1]
    
    labels = ['Baseline\n(disk only)', 'Add\nbulge', 'Add\ngas', 'Hybrid\nperiod']
    rms_values = [rms_baseline, rms_with_bulge, rms_with_gas, rms_hybrid]
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = ax.bar(labels, rms_values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(20, color='green', linestyle='--', linewidth=2, label='Target (<20 km/s)')
    ax.axhline(rms_baseline, color='blue', linestyle=':', alpha=0.5, label='Starting point')
    
    # Add value labels on bars
    for bar, val in zip(bars, rms_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{val:.1f}',
               ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('RMS [km/s]', fontsize=12)
    ax.set_title('RMS Comparison (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(rms_values) * 1.2)
    
    plt.tight_layout()
    plt.savefig('gravitywavebaseline/quick_improvement_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  [OK] Saved: gravitywavebaseline/quick_improvement_test.png")
    plt.close()
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nStarting point: {rms_baseline:.1f} km/s")
    print(f"With bulge:     {rms_with_bulge:.1f} km/s ({(rms_baseline-rms_with_bulge)/rms_baseline*100:.1f}% improvement)")
    print(f"With gas:       {rms_with_gas:.1f} km/s ({(rms_baseline-rms_with_gas)/rms_baseline*100:.1f}% improvement)")
    print(f"With hybrid:    {rms_hybrid:.1f} km/s ({(rms_baseline-rms_hybrid)/rms_baseline*100:.1f}% improvement)")
    
    print(f"\nTotal improvement possible: {rms_baseline - min(rms_values):.1f} km/s")
    
    if min(rms_values) < 20:
        print("\n[SUCCESS] Reached target RMS < 20 km/s!")
    elif min(rms_values) < 40:
        print("\n[OK] Significant improvement! Close to target.")
    else:
        print("\n[!] More work needed to reach RMS < 20 km/s")
    
    print("\nNext step: Run full optimization with improved_multiplier_calculation.py")
    print("           to fine-tune parameters for each configuration.")

def compute_simple_model(gaia, R_obs, period_col, A, lambda_0, alpha, M_disk,
                        n_sample=30000):
    """
    Simple model evaluation without full N×N calculation.
    Uses stratified sampling for speed.
    """
    
    # Sample stars for speed
    if len(gaia) > n_sample:
        sample_idx = np.random.choice(len(gaia), n_sample, replace=False)
        gaia_sample = gaia.iloc[sample_idx].copy()
    else:
        gaia_sample = gaia.copy()
    
    # Scale masses
    M_scale_factor = M_disk / gaia_sample['M_star'].sum()
    M_scaled = gaia_sample['M_star'].values * M_scale_factor
    
    # Star positions
    x_stars = gaia_sample['x'].values
    y_stars = gaia_sample['y'].values
    z_stars = gaia_sample['z'].values
    
    # Periods
    lambda_stars = gaia_sample[period_col].values
    
    # Calculate velocity at each radius
    v_model = np.zeros_like(R_obs)
    
    for i, R in enumerate(R_obs):
        # Observation point
        x_obs, y_obs, z_obs = R, 0, 0
        
        # Distances
        dx = x_stars - x_obs
        dy = y_stars - y_obs
        dz = z_stars - z_obs
        r = np.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)
        
        # Base gravity
        g_base = G_KPC * M_scaled / r**2
        
        # Power-law multiplier
        multiplier = 1.0 + A * (lambda_stars / lambda_0)**alpha
        g_enhanced = g_base * multiplier
        
        # Radial component
        cos_theta = dx / r
        g_radial = np.sum(g_enhanced * cos_theta)
        
        # Velocity
        v_model[i] = np.sqrt(max(R * g_radial, 0))
    
    return v_model

if __name__ == "__main__":
    quick_test_improvements()

