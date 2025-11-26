#!/usr/bin/env python3
"""
Test 3A: Morphology Split Analysis for Time-Dependent Coherence
================================================================

If gravitational coherence builds over dynamical timescales, older galaxies
should show MORE enhancement than younger ones.

Morphology as age proxy:
  - E/S0 (T=0): Old systems (~10 Gyr) → Full enhancement expected
  - Sa/Sb (T=1-4): Intermediate age → ~85% enhancement  
  - Sd/Irr (T=7-11): Young, chaotic (~2 Gyr) → ~50% enhancement

This test uses existing SPARC data to check if RAR residuals correlate
with morphological type.

PREDICTION (if time-dependent):
  - Early types (E/S0) should have smaller RAR residuals (better fit)
  - Late types (Sd/Irr) should have larger RAR residuals (less enhancement)

See README.md for data sources.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]


# Hubble type mapping
HUBBLE_TYPES = {
    0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
    6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'
}

# Age groupings
AGE_GROUPS = {
    'early': {'T_range': [0, 2], 'age_proxy': '~8-10 Gyr', 'color': 'red'},
    'intermediate': {'T_range': [3, 5], 'age_proxy': '~4-6 Gyr', 'color': 'orange'},
    'late': {'T_range': [6, 11], 'age_proxy': '~1-3 Gyr', 'color': 'blue'}
}


def load_sparc_mastersheet():
    """Load SPARC galaxy properties including Hubble types."""
    path = ROOT / "data" / "Rotmod_LTG" / "MasterSheet_SPARC.mrt"
    
    if not path.exists():
        print(f"ERROR: SPARC mastersheet not found at {path}")
        return None
    
    # Parse by splitting on whitespace - more robust
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Data starts after line 98 (index 98 onwards, which is line 99 1-based)
    for line in lines[98:]:
        line = line.strip()
        if not line:
            continue
            
        try:
            parts = line.split()
            if len(parts) < 14:
                continue
                
            # Column order from visual inspection:
            # 0: Galaxy, 1: T, 2: D, 3: e_D, 4: f_D, 5: Inc, 6: e_Inc, 
            # 7: L36, 8: e_L36, 9: Reff, 10: SBeff, 11: Rdisk, 12: SBdisk,
            # 13: MHI, 14: RHI, 15: Vflat, 16: e_Vflat, 17: Q, 18: Ref
            
            galaxy = parts[0]
            T = int(parts[1])
            D = float(parts[2])
            L36 = float(parts[7]) if len(parts) > 7 else np.nan
            
            # Vflat is at position 15 (0-indexed), might be '0.0' for no measurement
            Vflat = np.nan
            if len(parts) > 15:
                vflat_str = parts[15]
                if vflat_str not in ['0.0', '0']:
                    Vflat = float(vflat_str)
            
            Q = int(parts[17]) if len(parts) > 17 and parts[17].isdigit() else 3
            
            data.append({
                'Galaxy': galaxy,
                'T': T,
                'D_Mpc': D,
                'L36_1e9Lsun': L36,
                'Vflat_kms': Vflat,
                'Quality': Q
            })
        except (ValueError, IndexError) as e:
            continue
    
    df = pd.DataFrame(data)
    df = df[df['T'] >= 0]  # Valid Hubble types only
    
    print(f"Loaded {len(df)} SPARC galaxies with Hubble types")
    
    # Add age group classification
    def classify_age(T):
        for group, info in AGE_GROUPS.items():
            if info['T_range'][0] <= T <= info['T_range'][1]:
                return group
        return 'unknown'
    
    df['age_group'] = df['T'].apply(classify_age)
    
    return df


def load_rotation_curve_fits():
    """
    Load existing Σ-Gravity fits for SPARC galaxies.
    
    This reads from previous fit results if available.
    """
    # Try to find existing fits
    fit_paths = [
        ROOT / "outputs" / "sparc_fits.csv",
        ROOT / "outputs" / "sigma_gravity_fits.csv",
        ROOT / "coherence-field-theory" / "galaxies" / "sparc_results.csv",
    ]
    
    for path in fit_paths:
        if path.exists():
            print(f"Found fit results: {path}")
            return pd.read_csv(path)
    
    print("No existing fit results found - will compute RAR residuals from raw data")
    return None


def compute_rar_residuals_simple(sparc_df):
    """
    Compute simple RAR residuals using Vflat and luminosity.
    
    The RAR predicts: g_obs = g_bar / (1 - exp(-sqrt(g_bar/g†)))
    
    We use V² = g_obs × R, and estimate g_bar from luminosity.
    """
    G = 4.302e-6  # kpc km²/s²/Msun
    g_dagger = 1.2e-10  # m/s² = 3.7e-6 kpc km²/s²/Msun × Msun
    g_dagger_kpc = 3.7e-6  # kpc (km/s)² / Msun
    
    # Mass-to-light ratio assumption (solar units at 3.6μm)
    ML_36 = 0.5  # Msun/Lsun
    
    results = []
    
    for _, row in sparc_df.iterrows():
        if np.isnan(row['Vflat_kms']) or row['Vflat_kms'] <= 0:
            continue
        if np.isnan(row['L36_1e9Lsun']) or row['L36_1e9Lsun'] <= 0:
            continue
            
        V = row['Vflat_kms']
        L = row['L36_1e9Lsun'] * 1e9  # Convert to Lsun
        M_bar = ML_36 * L
        
        # Estimate effective radius (rough)
        R_eff = 3.0  # kpc typical
        
        # Baryonic acceleration
        g_bar = G * M_bar / R_eff**2
        
        # Observed acceleration from Vflat
        g_obs = V**2 / R_eff
        
        # RAR prediction (McGaugh 2016)
        g_rar = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger_kpc)))
        
        # Residual (log space)
        if g_rar > 0 and g_obs > 0:
            residual = np.log10(g_obs) - np.log10(g_rar)
        else:
            residual = np.nan
        
        results.append({
            'Galaxy': row['Galaxy'],
            'T': row['T'],
            'age_group': row['age_group'],
            'Vflat': V,
            'M_bar': M_bar,
            'g_bar': g_bar,
            'g_obs': g_obs,
            'g_rar': g_rar,
            'rar_residual': residual
        })
    
    return pd.DataFrame(results)


def analyze_morphology_split(df):
    """
    Analyze RAR residuals as function of morphological type.
    """
    print("\n" + "="*70)
    print("MORPHOLOGY SPLIT ANALYSIS")
    print("="*70)
    
    results = {}
    
    for group in ['early', 'intermediate', 'late']:
        subset = df[df['age_group'] == group]
        
        if len(subset) < 3:
            continue
            
        residuals = subset['rar_residual'].dropna()
        
        results[group] = {
            'n_galaxies': len(residuals),
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'median_residual': residuals.median(),
            'T_range': AGE_GROUPS[group]['T_range'],
            'age_proxy': AGE_GROUPS[group]['age_proxy']
        }
        
        print(f"\n{group.upper()} types (T={AGE_GROUPS[group]['T_range']}, {AGE_GROUPS[group]['age_proxy']}):")
        print(f"  N galaxies: {len(residuals)}")
        print(f"  Mean RAR residual: {residuals.mean():.4f} ± {residuals.std()/np.sqrt(len(residuals)):.4f} dex")
        print(f"  Median: {residuals.median():.4f} dex")
    
    # Statistical test: early vs late
    early = df[df['age_group'] == 'early']['rar_residual'].dropna()
    late = df[df['age_group'] == 'late']['rar_residual'].dropna()
    
    if len(early) > 3 and len(late) > 3:
        t_stat, p_value = stats.ttest_ind(early, late)
        mw_stat, mw_pvalue = stats.mannwhitneyu(early, late, alternative='two-sided')
        
        results['statistical_test'] = {
            'early_n': len(early),
            'late_n': len(late),
            'early_mean': early.mean(),
            'late_mean': late.mean(),
            't_statistic': t_stat,
            't_pvalue': p_value,
            'mannwhitney_pvalue': mw_pvalue
        }
        
        print("\n" + "-"*60)
        print("STATISTICAL COMPARISON: Early vs Late types")
        print("-"*60)
        print(f"  Early (old) mean:    {early.mean():.4f} ± {early.std():.4f} dex")
        print(f"  Late (young) mean:   {late.mean():.4f} ± {late.std():.4f} dex")
        print(f"  Difference:          {early.mean() - late.mean():.4f} dex")
        print(f"  t-test p-value:      {p_value:.4f}")
        print(f"  Mann-Whitney p:      {mw_pvalue:.4f}")
        
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        diff = early.mean() - late.mean()
        if diff < -0.05 and p_value < 0.05:
            print("  ✅ SUPPORTS time-dependent coherence:")
            print("     Early (old) types show LESS residual than late (young) types")
            print("     → Old systems more coherent = better RAR fit")
        elif diff > 0.05 and p_value < 0.05:
            print("  ❌ CONTRADICTS time-dependent coherence:")
            print("     Early types show MORE residual than late types")
            print("     → Young systems more coherent (unexpected)")
        else:
            print("  ❓ NO SIGNIFICANT DIFFERENCE detected")
            print("     → Coherence appears time-INDEPENDENT")
            print("     → Or morphology is a poor age proxy")
    
    return results


def plot_morphology_results(df, results, output_path=None):
    """Create visualization of morphology split results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Residual vs Hubble type
    ax = axes[0, 0]
    for group, info in AGE_GROUPS.items():
        subset = df[df['age_group'] == group]
        ax.scatter(subset['T'], subset['rar_residual'], 
                  c=info['color'], s=50, alpha=0.7, label=f"{group} ({info['age_proxy']})")
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Hubble Type T', fontsize=12)
    ax.set_ylabel('RAR Residual [dex]', fontsize=12)
    ax.set_title('RAR Residuals by Morphological Type', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Histogram by age group
    ax = axes[0, 1]
    for group, info in AGE_GROUPS.items():
        subset = df[df['age_group'] == group]['rar_residual'].dropna()
        if len(subset) > 0:
            ax.hist(subset, bins=15, alpha=0.5, color=info['color'], 
                   label=f"{group} (n={len(subset)})")
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('RAR Residual [dex]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Residual Distribution by Age Group', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Panel 3: Box plot
    ax = axes[1, 0]
    box_data = []
    box_labels = []
    box_colors = []
    for group in ['early', 'intermediate', 'late']:
        data = df[df['age_group'] == group]['rar_residual'].dropna()
        if len(data) > 0:
            box_data.append(data)
            box_labels.append(f"{group}\n({AGE_GROUPS[group]['age_proxy']})")
            box_colors.append(AGE_GROUPS[group]['color'])
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('RAR Residual [dex]', fontsize=12)
    ax.set_title('Residual Distribution Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    text = """
TIME-DEPENDENT COHERENCE TEST
=============================

Hypothesis: If coherence builds over dynamical
timescales, older systems should show MORE
gravitational enhancement.

Test: Compare RAR residuals across morphology:
  - Early types (S0/Sa/Sab): oldest, ~8-10 Gyr
  - Late types (Sd/Sm/Im): youngest, ~1-3 Gyr

PREDICTION (if time-dependent):
  Residual(early) < Residual(late)
  (old systems closer to RAR = more coherent)

"""
    
    if 'statistical_test' in results:
        st = results['statistical_test']
        text += f"""
RESULTS:
  Early types: {st['early_mean']:.4f} ± {results['early']['std_residual']:.4f} dex
  Late types:  {st['late_mean']:.4f} ± {results['late']['std_residual']:.4f} dex
  Difference:  {st['early_mean'] - st['late_mean']:.4f} dex
  p-value:     {st['t_pvalue']:.4f}

"""
        if st['t_pvalue'] < 0.05:
            if st['early_mean'] < st['late_mean']:
                text += "CONCLUSION: ✅ Supports time-dependence\n(older = more coherent)"
            else:
                text += "CONCLUSION: ❌ Contradicts time-dependence"
        else:
            text += "CONCLUSION: ❓ No significant difference\n(coherence time-independent?)"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot: {output_path}")
    
    plt.close()


def run_morphology_test():
    """Run the full morphology split test."""
    print("\n" + "="*70)
    print("TEST 3A: MORPHOLOGY SPLIT FOR TIME-DEPENDENT COHERENCE")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading SPARC galaxy properties...")
    sparc_df = load_sparc_mastersheet()
    if sparc_df is None:
        return None
    
    # Show distribution
    print("\nHubble type distribution:")
    for T in sorted(sparc_df['T'].unique()):
        n = (sparc_df['T'] == T).sum()
        print(f"  T={T:2d} ({HUBBLE_TYPES.get(T, '?'):4s}): {n:3d} galaxies")
    
    print("\nAge group distribution:")
    for group in ['early', 'intermediate', 'late']:
        n = (sparc_df['age_group'] == group).sum()
        print(f"  {group:12s}: {n:3d} galaxies ({AGE_GROUPS[group]['age_proxy']})")
    
    # Load or compute RAR residuals
    print("\n[2/4] Computing RAR residuals...")
    residual_df = compute_rar_residuals_simple(sparc_df)
    print(f"  Computed residuals for {len(residual_df)} galaxies")
    
    # Analyze
    print("\n[3/4] Analyzing morphology split...")
    results = analyze_morphology_split(residual_df)
    
    # Plot
    print("\n[4/4] Generating plots...")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plot_path = output_dir / "morphology_split_results.png"
    plot_morphology_results(residual_df, results, plot_path)
    
    # Save results
    results_path = output_dir / "morphology_split_results.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Saved results: {results_path}")
    
    return results


if __name__ == "__main__":
    run_morphology_test()
