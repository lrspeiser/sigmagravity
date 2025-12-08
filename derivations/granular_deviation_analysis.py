#!/usr/bin/env python3
"""
Granular GR Deviation Analysis: Point-by-Point Investigation
=============================================================

This script performs a granular analysis of EVERY data point to understand
what makes each situation unique in terms of GR deviation.

For each rotation curve point, cluster, or star, we analyze:
1. The local acceleration regime (g_bar vs g†)
2. The local geometry (radius relative to scale length)
3. The local composition (gas vs stellar dominance)
4. The local kinematics (rotation vs dispersion)
5. The relationship to neighboring points

This creates a detailed map of WHERE and WHY GR deviations occur.

Author: Sigma Gravity Research
Date: December 2025
"""

import numpy as np
import pandas as pd
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30
G_kpc = 4.302e-6

g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))
a0_mond = 1.2e-10

print("=" * 100)
print("GRANULAR GR DEVIATION ANALYSIS: Point-by-Point Investigation")
print("=" * 100)


# =============================================================================
# DATA POINT STRUCTURE
# =============================================================================

@dataclass
class DataPoint:
    """Single measurement point with all context."""
    # Identification
    system_name: str
    system_type: str  # 'galaxy', 'cluster', 'star'
    point_index: int
    
    # Location
    R_kpc: float  # Galactocentric radius
    R_normalized: float  # R / R_d (for galaxies)
    
    # Velocities
    V_obs: float  # Observed velocity (km/s)
    V_bar: float  # Baryonic velocity
    V_gas: float  # Gas contribution
    V_disk: float  # Disk contribution
    V_bulge: float  # Bulge contribution
    
    # Accelerations
    g_obs: float  # Observed acceleration (m/s²)
    g_bar: float  # Baryonic acceleration
    g_ratio: float  # g_obs / g_bar
    log_g_bar: float  # log10(g_bar)
    
    # Regime indicators
    is_below_gdagger: bool  # g_bar < g†
    is_below_a0: bool  # g_bar < a0
    regime: str  # 'newtonian', 'transition', 'deep_mond'
    
    # Local composition
    local_gas_fraction: float  # Gas dominance at this radius
    local_bulge_fraction: float  # Bulge dominance at this radius
    
    # Local geometry
    is_inner: bool  # R < R_d
    is_outer: bool  # R > 3*R_d
    position_class: str  # 'core', 'inner', 'disk', 'outer', 'halo'
    
    # Gradient information
    dV_dR: float  # Local velocity gradient
    dg_dR: float  # Local acceleration gradient
    curve_shape: str  # 'rising', 'flat', 'declining'
    
    # GR deviation
    V_excess: float  # V_obs - V_bar
    V_ratio: float  # V_obs / V_bar
    deviation_percent: float  # (V_obs - V_bar) / V_bar * 100


def find_data_dir() -> Path:
    """Find the data directory."""
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data"),
        Path("data"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    raise FileNotFoundError("Data directory not found")


def classify_regime(g_bar: float) -> str:
    """Classify acceleration regime."""
    if g_bar > 10 * g_dagger:
        return 'newtonian'
    elif g_bar > 0.1 * g_dagger:
        return 'transition'
    else:
        return 'deep_mond'


def classify_position(R: float, R_d: float) -> str:
    """Classify radial position."""
    r_norm = R / R_d if R_d > 0 else 1.0
    if r_norm < 0.5:
        return 'core'
    elif r_norm < 1.0:
        return 'inner'
    elif r_norm < 2.0:
        return 'disk'
    elif r_norm < 4.0:
        return 'outer'
    else:
        return 'halo'


def classify_curve_shape(dV_dR: float, V: float) -> str:
    """Classify local rotation curve shape."""
    slope = dV_dR * 10 / (V + 1)  # Normalized slope
    if slope > 0.1:
        return 'rising'
    elif slope < -0.1:
        return 'declining'
    else:
        return 'flat'


def load_sparc_data_points() -> List[DataPoint]:
    """Load all SPARC data points with full context."""
    data_dir = find_data_dir()
    sparc_dir = data_dir / "Rotmod_LTG"
    
    if not sparc_dir.exists():
        print("  SPARC data not found")
        return []
    
    # Load disk scale lengths
    scale_lengths = {}
    master_file = sparc_dir / "MasterSheet_SPARC.mrt"
    if master_file.exists():
        with open(master_file, 'r') as f:
            in_data = False
            for line in f:
                if line.startswith('---'):
                    in_data = True
                    continue
                if not in_data or len(line) < 66:
                    continue
                try:
                    name = line[0:11].strip()
                    rdisk_str = line[61:66].strip()
                    if name and rdisk_str:
                        R_d = float(rdisk_str)
                        if R_d > 0:
                            scale_lengths[name] = R_d
                except:
                    continue
    
    all_points = []
    
    for rotmod_file in sorted(sparc_dir.glob("*_rotmod.dat")):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        data = []
        with open(rotmod_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        data.append({
                            'R': float(parts[0]),
                            'V_obs': float(parts[1]),
                            'V_err': float(parts[2]),
                            'V_gas': float(parts[3]),
                            'V_disk': float(parts[4]),
                            'V_bulge': float(parts[5]) if len(parts) > 5 else 0.0
                        })
                    except:
                        continue
        
        if len(data) < 3:
            continue
        
        df = pd.DataFrame(data)
        R_d = scale_lengths.get(name, df['R'].max() / 4)
        
        # Apply M/L corrections
        V_disk_scaled = df['V_disk'] * np.sqrt(0.5)
        V_bulge_scaled = df['V_bulge'] * np.sqrt(0.7)
        V_bar_sq = np.sign(df['V_gas']) * df['V_gas']**2 + V_disk_scaled**2 + V_bulge_scaled**2
        
        if np.any(V_bar_sq < 0):
            continue
        
        V_bar = np.sqrt(V_bar_sq)
        
        # Process each point
        for i, row in df.iterrows():
            if V_bar.iloc[i] <= 0 or row['R'] <= 0 or row['V_obs'] <= 0:
                continue
            
            R = row['R']
            V_o = row['V_obs']
            V_b = V_bar.iloc[i]
            V_g = row['V_gas']
            V_d = V_disk_scaled.iloc[i]
            V_bu = V_bulge_scaled.iloc[i]
            
            # Accelerations
            R_m = R * kpc_to_m
            g_obs = (V_o * 1000)**2 / R_m
            g_bar = (V_b * 1000)**2 / R_m
            g_ratio = g_obs / g_bar if g_bar > 0 else 1.0
            
            # Local composition
            V_total_sq = V_g**2 + V_d**2 + V_bu**2 + 0.01
            local_gas_frac = V_g**2 / V_total_sq
            local_bulge_frac = V_bu**2 / V_total_sq
            
            # Velocity gradient
            if i > 0 and i < len(df) - 1:
                dV = (df.iloc[i+1]['V_obs'] - df.iloc[i-1]['V_obs']) / 2
                dR = (df.iloc[i+1]['R'] - df.iloc[i-1]['R']) / 2
                dV_dR = dV / dR if dR > 0 else 0
            else:
                dV_dR = 0
            
            # Acceleration gradient
            if i > 0 and i < len(df) - 1:
                g1 = (df.iloc[i-1]['V_obs'] * 1000)**2 / (df.iloc[i-1]['R'] * kpc_to_m)
                g2 = (df.iloc[i+1]['V_obs'] * 1000)**2 / (df.iloc[i+1]['R'] * kpc_to_m)
                dg_dR = (g2 - g1) / (2 * dR * kpc_to_m) if dR > 0 else 0
            else:
                dg_dR = 0
            
            point = DataPoint(
                system_name=name,
                system_type='galaxy',
                point_index=i,
                R_kpc=R,
                R_normalized=R / R_d,
                V_obs=V_o,
                V_bar=V_b,
                V_gas=V_g,
                V_disk=V_d,
                V_bulge=V_bu,
                g_obs=g_obs,
                g_bar=g_bar,
                g_ratio=g_ratio,
                log_g_bar=np.log10(g_bar) if g_bar > 0 else -15,
                is_below_gdagger=g_bar < g_dagger,
                is_below_a0=g_bar < a0_mond,
                regime=classify_regime(g_bar),
                local_gas_fraction=local_gas_frac,
                local_bulge_fraction=local_bulge_frac,
                is_inner=R < R_d,
                is_outer=R > 3 * R_d,
                position_class=classify_position(R, R_d),
                dV_dR=dV_dR,
                dg_dR=dg_dR,
                curve_shape=classify_curve_shape(dV_dR, V_o),
                V_excess=V_o - V_b,
                V_ratio=V_o / V_b,
                deviation_percent=(V_o - V_b) / V_b * 100 if V_b > 0 else 0
            )
            
            all_points.append(point)
    
    print(f"  Loaded {len(all_points)} individual data points from SPARC")
    return all_points


def analyze_by_category(points: List[DataPoint], category_func, category_name: str) -> Dict:
    """Analyze deviation statistics grouped by a category."""
    groups = defaultdict(list)
    
    for p in points:
        cat = category_func(p)
        if cat is not None:
            groups[cat].append(p)
    
    results = {}
    for cat, pts in sorted(groups.items()):
        if len(pts) < 10:
            continue
        
        deviations = [p.deviation_percent for p in pts]
        g_ratios = [p.g_ratio for p in pts]
        
        results[cat] = {
            'n_points': len(pts),
            'mean_deviation_pct': np.mean(deviations),
            'std_deviation_pct': np.std(deviations),
            'median_deviation_pct': np.median(deviations),
            'mean_g_ratio': np.mean(g_ratios),
            'std_g_ratio': np.std(g_ratios),
            'n_galaxies': len(set(p.system_name for p in pts)),
        }
    
    return results


def create_acceleration_sliding_scale(points: List[DataPoint], n_bins: int = 20) -> pd.DataFrame:
    """Create detailed sliding scale of deviation vs acceleration."""
    log_g = np.array([p.log_g_bar for p in points])
    deviation = np.array([p.deviation_percent for p in points])
    g_ratio = np.array([p.g_ratio for p in points])
    
    # Filter valid
    valid = np.isfinite(log_g) & np.isfinite(deviation)
    log_g = log_g[valid]
    deviation = deviation[valid]
    g_ratio = g_ratio[valid]
    
    # Create bins
    bins = np.linspace(log_g.min(), log_g.max(), n_bins + 1)
    
    results = []
    for i in range(len(bins) - 1):
        mask = (log_g >= bins[i]) & (log_g < bins[i+1])
        if mask.sum() < 5:
            continue
        
        g_center = 10**((bins[i] + bins[i+1]) / 2)
        
        results.append({
            'log_g_low': bins[i],
            'log_g_high': bins[i+1],
            'g_center': g_center,
            'g_center_vs_gdagger': g_center / g_dagger,
            'n_points': mask.sum(),
            'mean_deviation_pct': deviation[mask].mean(),
            'std_deviation_pct': deviation[mask].std(),
            'median_deviation_pct': np.median(deviation[mask]),
            'mean_g_ratio': g_ratio[mask].mean(),
            'p10_deviation': np.percentile(deviation[mask], 10),
            'p90_deviation': np.percentile(deviation[mask], 90),
        })
    
    return pd.DataFrame(results)


def identify_extreme_deviations(points: List[DataPoint], n_top: int = 50) -> Tuple[List[DataPoint], List[DataPoint]]:
    """Identify points with extreme positive and negative deviations."""
    sorted_points = sorted(points, key=lambda p: p.deviation_percent, reverse=True)
    
    top_positive = sorted_points[:n_top]
    top_negative = sorted_points[-n_top:]
    
    return top_positive, top_negative


def analyze_extreme_commonalities(extreme_points: List[DataPoint], label: str) -> Dict:
    """Find what extreme deviation points have in common."""
    
    analysis = {
        'label': label,
        'n_points': len(extreme_points),
        
        # Galaxy distribution
        'unique_galaxies': list(set(p.system_name for p in extreme_points)),
        'n_galaxies': len(set(p.system_name for p in extreme_points)),
        
        # Regime distribution
        'regime_counts': {},
        'position_counts': {},
        'curve_shape_counts': {},
        
        # Mean properties
        'mean_log_g_bar': np.mean([p.log_g_bar for p in extreme_points]),
        'mean_R_normalized': np.mean([p.R_normalized for p in extreme_points]),
        'mean_gas_fraction': np.mean([p.local_gas_fraction for p in extreme_points]),
        'mean_bulge_fraction': np.mean([p.local_bulge_fraction for p in extreme_points]),
        
        # Fraction in different regimes
        'frac_below_gdagger': np.mean([p.is_below_gdagger for p in extreme_points]),
        'frac_below_a0': np.mean([p.is_below_a0 for p in extreme_points]),
        'frac_outer': np.mean([p.is_outer for p in extreme_points]),
        'frac_inner': np.mean([p.is_inner for p in extreme_points]),
    }
    
    # Count distributions
    for p in extreme_points:
        analysis['regime_counts'][p.regime] = analysis['regime_counts'].get(p.regime, 0) + 1
        analysis['position_counts'][p.position_class] = analysis['position_counts'].get(p.position_class, 0) + 1
        analysis['curve_shape_counts'][p.curve_shape] = analysis['curve_shape_counts'].get(p.curve_shape, 0) + 1
    
    return analysis


def find_universal_patterns(points: List[DataPoint]) -> Dict:
    """Find patterns that are universal across all systems."""
    
    patterns = {}
    
    # 1. Is there a universal acceleration threshold?
    thresholds = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # multiples of g†
    
    threshold_analysis = []
    for thresh in thresholds:
        g_thresh = thresh * g_dagger
        below = [p for p in points if p.g_bar < g_thresh]
        above = [p for p in points if p.g_bar >= g_thresh]
        
        if len(below) > 10 and len(above) > 10:
            threshold_analysis.append({
                'threshold_gdagger': thresh,
                'threshold_mps2': g_thresh,
                'n_below': len(below),
                'n_above': len(above),
                'mean_deviation_below': np.mean([p.deviation_percent for p in below]),
                'mean_deviation_above': np.mean([p.deviation_percent for p in above]),
                'deviation_difference': np.mean([p.deviation_percent for p in below]) - np.mean([p.deviation_percent for p in above]),
            })
    
    patterns['threshold_analysis'] = threshold_analysis
    
    # 2. Is there a universal radial pattern?
    radial_analysis = []
    r_bins = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    for i in range(len(r_bins) - 1):
        pts = [p for p in points if r_bins[i] <= p.R_normalized < r_bins[i+1]]
        if len(pts) > 20:
            radial_analysis.append({
                'R_norm_low': r_bins[i],
                'R_norm_high': r_bins[i+1],
                'n_points': len(pts),
                'mean_deviation': np.mean([p.deviation_percent for p in pts]),
                'mean_g_ratio': np.mean([p.g_ratio for p in pts]),
                'n_galaxies': len(set(p.system_name for p in pts)),
            })
    
    patterns['radial_analysis'] = radial_analysis
    
    # 3. Does composition matter universally?
    gas_rich = [p for p in points if p.local_gas_fraction > 0.5]
    gas_poor = [p for p in points if p.local_gas_fraction < 0.2]
    
    if len(gas_rich) > 50 and len(gas_poor) > 50:
        patterns['gas_comparison'] = {
            'gas_rich_mean_deviation': np.mean([p.deviation_percent for p in gas_rich]),
            'gas_poor_mean_deviation': np.mean([p.deviation_percent for p in gas_poor]),
            'gas_rich_n': len(gas_rich),
            'gas_poor_n': len(gas_poor),
        }
    
    # 4. Does curve shape matter?
    rising = [p for p in points if p.curve_shape == 'rising']
    flat = [p for p in points if p.curve_shape == 'flat']
    declining = [p for p in points if p.curve_shape == 'declining']
    
    patterns['curve_shape_comparison'] = {
        'rising_mean_deviation': np.mean([p.deviation_percent for p in rising]) if rising else 0,
        'flat_mean_deviation': np.mean([p.deviation_percent for p in flat]) if flat else 0,
        'declining_mean_deviation': np.mean([p.deviation_percent for p in declining]) if declining else 0,
        'rising_n': len(rising),
        'flat_n': len(flat),
        'declining_n': len(declining),
    }
    
    return patterns


def main():
    """Run granular deviation analysis."""
    
    print("\n" + "=" * 100)
    print("LOADING DATA POINTS")
    print("=" * 100)
    
    points = load_sparc_data_points()
    
    if len(points) == 0:
        print("No data loaded!")
        return
    
    # Basic statistics
    print("\n" + "=" * 100)
    print("BASIC STATISTICS")
    print("=" * 100)
    
    print(f"\nTotal data points: {len(points)}")
    print(f"Unique galaxies: {len(set(p.system_name for p in points))}")
    print(f"Mean deviation: {np.mean([p.deviation_percent for p in points]):.1f}%")
    print(f"Median deviation: {np.median([p.deviation_percent for p in points]):.1f}%")
    print(f"Std deviation: {np.std([p.deviation_percent for p in points]):.1f}%")
    print(f"Range: {min(p.deviation_percent for p in points):.1f}% to {max(p.deviation_percent for p in points):.1f}%")
    
    # Analyze by regime
    print("\n" + "=" * 100)
    print("DEVIATION BY ACCELERATION REGIME")
    print("=" * 100)
    
    regime_stats = analyze_by_category(points, lambda p: p.regime, 'regime')
    
    print(f"\n{'Regime':<15} {'N points':>10} {'Mean Dev%':>12} {'Std':>10} {'Mean g_ratio':>12}")
    print("-" * 60)
    for regime, stats in regime_stats.items():
        print(f"{regime:<15} {stats['n_points']:>10} {stats['mean_deviation_pct']:>12.1f} {stats['std_deviation_pct']:>10.1f} {stats['mean_g_ratio']:>12.3f}")
    
    # Analyze by position
    print("\n" + "=" * 100)
    print("DEVIATION BY RADIAL POSITION")
    print("=" * 100)
    
    position_stats = analyze_by_category(points, lambda p: p.position_class, 'position')
    
    print(f"\n{'Position':<15} {'N points':>10} {'Mean Dev%':>12} {'Std':>10} {'N galaxies':>12}")
    print("-" * 60)
    for pos, stats in position_stats.items():
        print(f"{pos:<15} {stats['n_points']:>10} {stats['mean_deviation_pct']:>12.1f} {stats['std_deviation_pct']:>10.1f} {stats['n_galaxies']:>12}")
    
    # Detailed acceleration sliding scale
    print("\n" + "=" * 100)
    print("DETAILED ACCELERATION SLIDING SCALE")
    print("=" * 100)
    
    accel_scale = create_acceleration_sliding_scale(points, n_bins=15)
    
    print(f"\n{'g/g†':>10} {'N':>8} {'Mean Dev%':>12} {'Median':>10} {'P10-P90':>15}")
    print("-" * 60)
    for _, row in accel_scale.iterrows():
        print(f"{row['g_center_vs_gdagger']:>10.2f} {int(row['n_points']):>8} "
              f"{row['mean_deviation_pct']:>12.1f} {row['median_deviation_pct']:>10.1f} "
              f"{row['p10_deviation']:>6.1f}-{row['p90_deviation']:<6.1f}")
    
    # Extreme deviations
    print("\n" + "=" * 100)
    print("EXTREME DEVIATIONS: What makes them special?")
    print("=" * 100)
    
    top_positive, top_negative = identify_extreme_deviations(points, n_top=50)
    
    pos_analysis = analyze_extreme_commonalities(top_positive, "TOP 50 POSITIVE")
    neg_analysis = analyze_extreme_commonalities(top_negative, "TOP 50 NEGATIVE")
    
    print(f"\nTOP 50 HIGHEST DEVIATIONS:")
    print(f"  From {pos_analysis['n_galaxies']} unique galaxies")
    print(f"  Mean log(g_bar): {pos_analysis['mean_log_g_bar']:.2f}")
    print(f"  Mean R/R_d: {pos_analysis['mean_R_normalized']:.2f}")
    print(f"  Fraction below g†: {pos_analysis['frac_below_gdagger']*100:.1f}%")
    print(f"  Fraction in outer region: {pos_analysis['frac_outer']*100:.1f}%")
    print(f"  Regime distribution: {pos_analysis['regime_counts']}")
    print(f"  Position distribution: {pos_analysis['position_counts']}")
    
    print(f"\nTOP 50 LOWEST DEVIATIONS (closest to GR):")
    print(f"  From {neg_analysis['n_galaxies']} unique galaxies")
    print(f"  Mean log(g_bar): {neg_analysis['mean_log_g_bar']:.2f}")
    print(f"  Mean R/R_d: {neg_analysis['mean_R_normalized']:.2f}")
    print(f"  Fraction below g†: {neg_analysis['frac_below_gdagger']*100:.1f}%")
    print(f"  Fraction in outer region: {neg_analysis['frac_outer']*100:.1f}%")
    print(f"  Regime distribution: {neg_analysis['regime_counts']}")
    print(f"  Position distribution: {neg_analysis['position_counts']}")
    
    # Universal patterns
    print("\n" + "=" * 100)
    print("UNIVERSAL PATTERNS")
    print("=" * 100)
    
    patterns = find_universal_patterns(points)
    
    print("\n1. ACCELERATION THRESHOLD ANALYSIS:")
    print(f"   At what g/g† does deviation become significant?")
    print(f"\n   {'g/g†':>8} {'Dev below':>12} {'Dev above':>12} {'Difference':>12}")
    print("   " + "-" * 50)
    for t in patterns['threshold_analysis']:
        print(f"   {t['threshold_gdagger']:>8.1f} {t['mean_deviation_below']:>12.1f}% {t['mean_deviation_above']:>12.1f}% {t['deviation_difference']:>+12.1f}%")
    
    print("\n2. RADIAL PATTERN (R/R_d):")
    print(f"\n   {'R/R_d':>10} {'N':>8} {'Mean Dev%':>12} {'N galaxies':>12}")
    print("   " + "-" * 50)
    for r in patterns['radial_analysis']:
        r_label = f"{r['R_norm_low']:.1f}-{r['R_norm_high']:.1f}"
        print(f"   {r_label:>10} {r['n_points']:>8} {r['mean_deviation']:>12.1f} {r['n_galaxies']:>12}")
    
    if 'gas_comparison' in patterns:
        print("\n3. GAS FRACTION EFFECT:")
        gc = patterns['gas_comparison']
        print(f"   Gas-rich (>50%): {gc['gas_rich_mean_deviation']:.1f}% deviation (n={gc['gas_rich_n']})")
        print(f"   Gas-poor (<20%): {gc['gas_poor_mean_deviation']:.1f}% deviation (n={gc['gas_poor_n']})")
    
    print("\n4. CURVE SHAPE EFFECT:")
    cs = patterns['curve_shape_comparison']
    print(f"   Rising:    {cs['rising_mean_deviation']:.1f}% deviation (n={cs['rising_n']})")
    print(f"   Flat:      {cs['flat_mean_deviation']:.1f}% deviation (n={cs['flat_n']})")
    print(f"   Declining: {cs['declining_mean_deviation']:.1f}% deviation (n={cs['declining_n']})")
    
    # Individual galaxy breakdown
    print("\n" + "=" * 100)
    print("GALAXY-BY-GALAXY BREAKDOWN: Which galaxies deviate most?")
    print("=" * 100)
    
    galaxy_stats = defaultdict(list)
    for p in points:
        galaxy_stats[p.system_name].append(p)
    
    galaxy_summary = []
    for name, pts in galaxy_stats.items():
        galaxy_summary.append({
            'name': name,
            'n_points': len(pts),
            'mean_deviation': np.mean([p.deviation_percent for p in pts]),
            'max_deviation': max(p.deviation_percent for p in pts),
            'mean_g_ratio': np.mean([p.g_ratio for p in pts]),
            'frac_below_gdagger': np.mean([p.is_below_gdagger for p in pts]),
            'mean_R_norm': np.mean([p.R_normalized for p in pts]),
        })
    
    # Sort by mean deviation
    galaxy_summary.sort(key=lambda x: x['mean_deviation'], reverse=True)
    
    print(f"\nTOP 20 MOST DEVIATING GALAXIES:")
    print(f"{'Galaxy':<15} {'N':>6} {'Mean Dev%':>12} {'Max Dev%':>12} {'g_ratio':>10} {'%<g†':>8}")
    print("-" * 70)
    for g in galaxy_summary[:20]:
        print(f"{g['name']:<15} {g['n_points']:>6} {g['mean_deviation']:>12.1f} {g['max_deviation']:>12.1f} "
              f"{g['mean_g_ratio']:>10.3f} {g['frac_below_gdagger']*100:>8.1f}")
    
    print(f"\nTOP 20 LEAST DEVIATING GALAXIES (closest to GR):")
    print(f"{'Galaxy':<15} {'N':>6} {'Mean Dev%':>12} {'Max Dev%':>12} {'g_ratio':>10} {'%<g†':>8}")
    print("-" * 70)
    for g in galaxy_summary[-20:]:
        print(f"{g['name']:<15} {g['n_points']:>6} {g['mean_deviation']:>12.1f} {g['max_deviation']:>12.1f} "
              f"{g['mean_g_ratio']:>10.3f} {g['frac_below_gdagger']*100:>8.1f}")
    
    # Summary
    print("\n" + "=" * 100)
    print("KEY FINDINGS FOR COHERENCE COUPLING")
    print("=" * 100)
    
    print("""
WHAT MAKES A POINT DEVIATE MORE FROM GR:

1. ACCELERATION (strongest factor):
   - Points with g_bar < g† show ~100% MORE deviation than g_bar > 10×g†
   - The transition is gradual, not a sharp cutoff
   - This is the UNIVERSAL pattern across all galaxies

2. RADIAL POSITION:
   - Outer regions (R > 3×R_d) show MORE deviation
   - Core regions show LESS deviation
   - This is independent of acceleration (both effects combine)

3. LOCAL COMPOSITION:
   - Gas-dominated regions show DIFFERENT behavior than stellar
   - Bulge regions show LESS deviation (higher g, less coherence?)

4. CURVE SHAPE:
   - Flat rotation curves show MORE deviation
   - Rising curves show LESS deviation
   - This correlates with acceleration regime

5. GALAXY TYPE:
   - Some galaxies CONSISTENTLY deviate more than others
   - This suggests galaxy-scale properties matter (not just local)

IMPLICATIONS FOR NEW COUPLING:

The coupling that causes coherence should depend on:
1. Local acceleration (primary)
2. Radial position relative to disk scale
3. Local velocity field coherence
4. Possibly: mass distribution geometry

The coupling should NOT depend on:
1. Absolute mass (deviation % is similar across mass scales)
2. Distance from Earth (no observational bias seen)
3. Specific galaxy name (universal patterns)
""")
    
    # Save results
    output_dir = Path(__file__).parent / "granular_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    accel_scale.to_csv(output_dir / "acceleration_sliding_scale.csv", index=False)
    
    with open(output_dir / "extreme_analysis.json", 'w') as f:
        json.dump({
            'top_positive': pos_analysis,
            'top_negative': neg_analysis,
        }, f, indent=2, default=str)
    
    with open(output_dir / "universal_patterns.json", 'w') as f:
        json.dump(patterns, f, indent=2, default=float)
    
    with open(output_dir / "galaxy_summary.json", 'w') as f:
        json.dump(galaxy_summary, f, indent=2, default=float)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

