#!/usr/bin/env python3
"""
SPARC Galaxy Classifier and Diagnostic Report Generator

This script classifies SPARC galaxies by multiple criteria and generates
detailed reports to identify systematic issues in Σ-Gravity predictions.

Classifications:
1. Morphological type (from SPARC metadata)
2. Mass regime (dwarf, normal, massive)
3. Surface brightness (LSB, HSB)
4. Gas fraction (gas-rich, gas-poor)
5. Bulge fraction (disk-dominated, bulge-dominated)
6. Quality of fit (good, moderate, poor)

Reports generated:
- Summary statistics by category
- Worst-performing galaxies with diagnostics
- Systematic trends (residuals vs properties)
- Recommendations for model improvements

Usage:
    python sparc_galaxy_classifier.py [--output-dir <dir>]
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import math

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration
g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))

# Optimized parameters
R0 = 5.0  # kpc
A_COEFF = 1.6
B_COEFF = 109.0
G_GALAXY = 0.038
A_GALAXY = np.sqrt(A_COEFF + B_COEFF * G_GALAXY**2)


@dataclass
class GalaxyData:
    """Container for a single galaxy's data and results."""
    name: str
    R: np.ndarray
    V_obs: np.ndarray
    V_bar: np.ndarray
    V_gas: np.ndarray
    V_disk: np.ndarray
    V_bulge: np.ndarray
    V_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    V_mond: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Derived properties
    R_max: float = 0.0
    V_flat: float = 0.0
    M_bar: float = 0.0
    M_gas: float = 0.0
    M_disk: float = 0.0
    M_bulge: float = 0.0
    gas_fraction: float = 0.0
    bulge_fraction: float = 0.0
    R_d: float = 0.0  # Disk scale length estimate
    
    # Fit quality
    rms_sigma: float = 0.0
    rms_mond: float = 0.0
    chi2_sigma: float = 0.0
    chi2_mond: float = 0.0
    wins_sigma: bool = False
    
    # Classifications
    mass_class: str = ""
    sb_class: str = ""
    gas_class: str = ""
    bulge_class: str = ""
    fit_class: str = ""
    
    # Diagnostics
    mean_residual: float = 0.0
    inner_residual: float = 0.0
    outer_residual: float = 0.0
    residual_trend: float = 0.0  # Slope of residual vs R


def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g)."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def f_path(r: np.ndarray, r0: float) -> np.ndarray:
    """Path-length factor f(r)."""
    return r / (r + r0)


def predict_sigma_gravity(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """Predict rotation velocity using Σ-Gravity."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    f = f_path(R_kpc, R0)
    
    Sigma = 1 + A_GALAXY * f * h
    return V_bar * np.sqrt(Sigma)


def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """Predict rotation velocity using MOND."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    a0 = 1.2e-10
    x = g_bar / a0
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))
    return V_bar * np.power(nu, 0.25)


def load_sparc_galaxies(data_dir: Path) -> List[GalaxyData]:
    """Load all SPARC galaxies with full data."""
    sparc_dir = data_dir / "Rotmod_LTG"
    galaxy_files = sorted(sparc_dir.glob("*_rotmod.dat"))
    
    galaxies = []
    for gf in galaxy_files:
        data = []
        with open(gf) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    data.append({
                        'R': float(parts[0]),
                        'V_obs': float(parts[1]),
                        'V_err': float(parts[2]) if len(parts) > 2 else 5.0,
                        'V_gas': float(parts[3]),
                        'V_disk': float(parts[4]),
                        'V_bulge': float(parts[5]) if len(parts) > 5 else 0.0
                    })
        
        if len(data) < 5:
            continue
        
        df = pd.DataFrame(data)
        
        # Apply M/L corrections
        V_disk_scaled = df['V_disk'] * np.sqrt(0.5)
        V_bulge_scaled = df['V_bulge'] * np.sqrt(0.7)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    V_disk_scaled**2 + V_bulge_scaled**2)
        V_bar = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        # Filter valid points
        valid = (V_bar > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        if valid.sum() < 5:
            continue
        
        gal = GalaxyData(
            name=gf.stem.replace('_rotmod', ''),
            R=df.loc[valid, 'R'].values,
            V_obs=df.loc[valid, 'V_obs'].values,
            V_bar=V_bar[valid].values,
            V_gas=df.loc[valid, 'V_gas'].values,
            V_disk=V_disk_scaled[valid].values,
            V_bulge=V_bulge_scaled[valid].values
        )
        
        # Compute predictions
        gal.V_pred = predict_sigma_gravity(gal.R, gal.V_bar)
        gal.V_mond = predict_mond(gal.R, gal.V_bar)
        
        # Compute derived properties
        gal.R_max = gal.R.max()
        gal.V_flat = np.median(gal.V_obs[-3:]) if len(gal.V_obs) >= 3 else gal.V_obs[-1]
        
        # Estimate masses from V_flat and R_max
        G_kpc = 4.302e-6  # (km/s)^2 kpc / M_sun
        gal.M_bar = gal.V_flat**2 * gal.R_max / G_kpc
        
        # Component masses (rough estimates)
        V_gas_max = np.abs(gal.V_gas).max() if len(gal.V_gas) > 0 else 0
        V_disk_max = np.abs(gal.V_disk).max() if len(gal.V_disk) > 0 else 0
        V_bulge_max = np.abs(gal.V_bulge).max() if len(gal.V_bulge) > 0 else 0
        
        V_total_sq = V_gas_max**2 + V_disk_max**2 + V_bulge_max**2
        if V_total_sq > 0:
            gal.gas_fraction = V_gas_max**2 / V_total_sq
            gal.bulge_fraction = V_bulge_max**2 / V_total_sq
        
        # Estimate disk scale length (R where V_disk peaks or R_max/3)
        if len(gal.V_disk) > 0 and gal.V_disk.max() > 0:
            peak_idx = np.argmax(gal.V_disk)
            gal.R_d = gal.R[peak_idx] if peak_idx > 0 else gal.R_max / 3
        else:
            gal.R_d = gal.R_max / 3
        
        # Compute fit quality
        residuals = gal.V_obs - gal.V_pred
        residuals_mond = gal.V_obs - gal.V_mond
        
        gal.rms_sigma = np.sqrt((residuals**2).mean())
        gal.rms_mond = np.sqrt((residuals_mond**2).mean())
        gal.wins_sigma = gal.rms_sigma < gal.rms_mond
        
        # Diagnostics
        gal.mean_residual = residuals.mean()
        
        # Inner vs outer residuals
        R_mid = gal.R_max / 2
        inner_mask = gal.R < R_mid
        outer_mask = gal.R >= R_mid
        
        if inner_mask.sum() > 0:
            gal.inner_residual = residuals[inner_mask].mean()
        if outer_mask.sum() > 0:
            gal.outer_residual = residuals[outer_mask].mean()
        
        # Residual trend (slope)
        if len(gal.R) > 3:
            coef = np.polyfit(gal.R, residuals, 1)
            gal.residual_trend = coef[0]  # km/s per kpc
        
        galaxies.append(gal)
    
    return galaxies


def classify_galaxies(galaxies: List[GalaxyData]) -> None:
    """Classify galaxies by multiple criteria."""
    for gal in galaxies:
        # Mass classification
        if gal.V_flat < 80:
            gal.mass_class = "dwarf"
        elif gal.V_flat < 150:
            gal.mass_class = "normal"
        else:
            gal.mass_class = "massive"
        
        # Surface brightness (based on V_bar/R scaling)
        # LSB galaxies have lower V_bar for given R
        mean_vbar_r = (gal.V_bar / gal.R).mean() if (gal.R > 0).all() else 0
        if mean_vbar_r < 15:  # km/s/kpc
            gal.sb_class = "LSB"
        else:
            gal.sb_class = "HSB"
        
        # Gas fraction
        if gal.gas_fraction > 0.5:
            gal.gas_class = "gas-rich"
        elif gal.gas_fraction > 0.2:
            gal.gas_class = "mixed"
        else:
            gal.gas_class = "gas-poor"
        
        # Bulge fraction
        if gal.bulge_fraction > 0.3:
            gal.bulge_class = "bulge-dominated"
        elif gal.bulge_fraction > 0.1:
            gal.bulge_class = "intermediate"
        else:
            gal.bulge_class = "disk-dominated"
        
        # Fit quality
        if gal.rms_sigma < 10:
            gal.fit_class = "excellent"
        elif gal.rms_sigma < 20:
            gal.fit_class = "good"
        elif gal.rms_sigma < 30:
            gal.fit_class = "moderate"
        else:
            gal.fit_class = "poor"


def generate_category_report(galaxies: List[GalaxyData], 
                              category: str, 
                              get_class: callable) -> Dict:
    """Generate statistics for a given classification category."""
    classes = {}
    for gal in galaxies:
        cls = get_class(gal)
        if cls not in classes:
            classes[cls] = []
        classes[cls].append(gal)
    
    report = {
        'category': category,
        'classes': {}
    }
    
    for cls, gals in sorted(classes.items()):
        rms_vals = [g.rms_sigma for g in gals]
        rms_mond_vals = [g.rms_mond for g in gals]
        wins = sum(1 for g in gals if g.wins_sigma)
        
        mean_residuals = [g.mean_residual for g in gals]
        inner_residuals = [g.inner_residual for g in gals]
        outer_residuals = [g.outer_residual for g in gals]
        trends = [g.residual_trend for g in gals]
        
        report['classes'][cls] = {
            'count': len(gals),
            'rms_mean': np.mean(rms_vals),
            'rms_std': np.std(rms_vals),
            'rms_median': np.median(rms_vals),
            'mond_rms_mean': np.mean(rms_mond_vals),
            'wins': wins,
            'win_rate': wins / len(gals) if len(gals) > 0 else 0,
            'improvement_vs_mond': (np.mean(rms_mond_vals) - np.mean(rms_vals)) / np.mean(rms_mond_vals) * 100,
            'mean_residual': np.mean(mean_residuals),
            'inner_residual': np.mean(inner_residuals),
            'outer_residual': np.mean(outer_residuals),
            'residual_trend': np.mean(trends),
            'galaxies': [g.name for g in gals]
        }
    
    return report


def identify_problem_galaxies(galaxies: List[GalaxyData], 
                               n_worst: int = 20) -> List[Dict]:
    """Identify the worst-performing galaxies with diagnostics."""
    sorted_gals = sorted(galaxies, key=lambda g: g.rms_sigma, reverse=True)
    
    problems = []
    for gal in sorted_gals[:n_worst]:
        # Identify likely issue
        issues = []
        
        if gal.mean_residual > 10:
            issues.append("systematic_underprediction")
        elif gal.mean_residual < -10:
            issues.append("systematic_overprediction")
        
        if gal.inner_residual > 15:
            issues.append("inner_underprediction")
        elif gal.inner_residual < -15:
            issues.append("inner_overprediction")
        
        if gal.outer_residual > 15:
            issues.append("outer_underprediction")
        elif gal.outer_residual < -15:
            issues.append("outer_overprediction")
        
        if abs(gal.residual_trend) > 2:
            if gal.residual_trend > 0:
                issues.append("rising_residual_with_R")
            else:
                issues.append("falling_residual_with_R")
        
        if gal.gas_fraction > 0.7:
            issues.append("very_gas_rich")
        
        if gal.bulge_fraction > 0.4:
            issues.append("strong_bulge")
        
        if gal.R_max < 5:
            issues.append("small_extent")
        
        if not issues:
            issues.append("unknown")
        
        problems.append({
            'name': gal.name,
            'rms_sigma': gal.rms_sigma,
            'rms_mond': gal.rms_mond,
            'wins_sigma': gal.wins_sigma,
            'V_flat': gal.V_flat,
            'R_max': gal.R_max,
            'gas_fraction': gal.gas_fraction,
            'bulge_fraction': gal.bulge_fraction,
            'mass_class': gal.mass_class,
            'sb_class': gal.sb_class,
            'mean_residual': gal.mean_residual,
            'inner_residual': gal.inner_residual,
            'outer_residual': gal.outer_residual,
            'residual_trend': gal.residual_trend,
            'likely_issues': issues
        })
    
    return problems


def analyze_systematic_trends(galaxies: List[GalaxyData]) -> Dict:
    """Analyze systematic trends in residuals vs galaxy properties."""
    # Collect data
    V_flat = np.array([g.V_flat for g in galaxies])
    R_max = np.array([g.R_max for g in galaxies])
    gas_frac = np.array([g.gas_fraction for g in galaxies])
    bulge_frac = np.array([g.bulge_fraction for g in galaxies])
    mean_resid = np.array([g.mean_residual for g in galaxies])
    rms = np.array([g.rms_sigma for g in galaxies])
    
    # Compute correlations
    def safe_corr(x, y):
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            return 0.0
        return np.corrcoef(x[valid], y[valid])[0, 1]
    
    trends = {
        'rms_vs_V_flat': {
            'correlation': safe_corr(V_flat, rms),
            'interpretation': 'positive = worse fits for massive galaxies'
        },
        'rms_vs_R_max': {
            'correlation': safe_corr(R_max, rms),
            'interpretation': 'positive = worse fits for extended galaxies'
        },
        'rms_vs_gas_fraction': {
            'correlation': safe_corr(gas_frac, rms),
            'interpretation': 'positive = worse fits for gas-rich galaxies'
        },
        'rms_vs_bulge_fraction': {
            'correlation': safe_corr(bulge_frac, rms),
            'interpretation': 'positive = worse fits for bulge-dominated galaxies'
        },
        'residual_vs_V_flat': {
            'correlation': safe_corr(V_flat, mean_resid),
            'interpretation': 'positive = underprediction for massive galaxies'
        },
        'residual_vs_gas_fraction': {
            'correlation': safe_corr(gas_frac, mean_resid),
            'interpretation': 'positive = underprediction for gas-rich galaxies'
        }
    }
    
    # Identify strongest trends
    strongest = max(trends.items(), key=lambda x: abs(x[1]['correlation']))
    trends['strongest_trend'] = {
        'name': strongest[0],
        'correlation': strongest[1]['correlation'],
        'interpretation': strongest[1]['interpretation']
    }
    
    return trends


def generate_recommendations(category_reports: List[Dict], 
                              problems: List[Dict],
                              trends: Dict) -> List[str]:
    """Generate recommendations for model improvements."""
    recommendations = []
    
    # Check for mass-dependent issues
    mass_report = next((r for r in category_reports if r['category'] == 'Mass'), None)
    if mass_report:
        dwarf = mass_report['classes'].get('dwarf', {})
        massive = mass_report['classes'].get('massive', {})
        
        if dwarf.get('rms_mean', 0) > massive.get('rms_mean', 0) * 1.3:
            recommendations.append(
                "DWARF GALAXY ISSUE: Dwarf galaxies have significantly worse fits. "
                "Consider: (1) Different G_galaxy for dwarfs, (2) Modified f(r) for small R, "
                "(3) Different amplitude scaling for low-mass systems."
            )
        
        if massive.get('mean_residual', 0) > 10:
            recommendations.append(
                "MASSIVE GALAXY UNDERPREDICTION: Systematic underprediction for massive galaxies. "
                "Consider: (1) Increasing A_galaxy, (2) Reducing r0, (3) Mass-dependent amplitude."
            )
    
    # Check for surface brightness issues
    sb_report = next((r for r in category_reports if r['category'] == 'Surface Brightness'), None)
    if sb_report:
        lsb = sb_report['classes'].get('LSB', {})
        hsb = sb_report['classes'].get('HSB', {})
        
        if lsb.get('rms_mean', 0) > hsb.get('rms_mean', 0) * 1.3:
            recommendations.append(
                "LSB GALAXY ISSUE: Low surface brightness galaxies have worse fits. "
                "Consider: (1) Surface brightness modulation of amplitude, "
                "(2) Different coherence scale for diffuse systems."
            )
    
    # Check for gas fraction issues
    gas_report = next((r for r in category_reports if r['category'] == 'Gas Fraction'), None)
    if gas_report:
        gas_rich = gas_report['classes'].get('gas-rich', {})
        gas_poor = gas_report['classes'].get('gas-poor', {})
        
        if gas_rich.get('rms_mean', 0) > gas_poor.get('rms_mean', 0) * 1.3:
            recommendations.append(
                "GAS-RICH GALAXY ISSUE: Gas-rich galaxies have worse fits. "
                "Consider: (1) Different M/L for gas, (2) Gas contribution to coherence, "
                "(3) Separate treatment of cold gas vs stellar disk."
            )
    
    # Check for bulge issues
    bulge_report = next((r for r in category_reports if r['category'] == 'Bulge Fraction'), None)
    if bulge_report:
        bulge_dom = bulge_report['classes'].get('bulge-dominated', {})
        disk_dom = bulge_report['classes'].get('disk-dominated', {})
        
        if bulge_dom.get('rms_mean', 0) > disk_dom.get('rms_mean', 0) * 1.3:
            recommendations.append(
                "BULGE-DOMINATED GALAXY ISSUE: Bulge-dominated galaxies have worse fits. "
                "Consider: (1) Different G for bulge-dominated systems (G → 0.1-0.2), "
                "(2) Two-component model with separate G for bulge and disk."
            )
    
    # Check for radial issues
    common_issues = {}
    for p in problems:
        for issue in p['likely_issues']:
            common_issues[issue] = common_issues.get(issue, 0) + 1
    
    if common_issues.get('inner_underprediction', 0) > 5:
        recommendations.append(
            "INNER REGION UNDERPREDICTION: Multiple galaxies show underprediction in inner regions. "
            "Consider: (1) Reducing r0 to allow earlier coherence buildup, "
            "(2) Modified f(r) with faster inner rise."
        )
    
    if common_issues.get('outer_underprediction', 0) > 5:
        recommendations.append(
            "OUTER REGION UNDERPREDICTION: Multiple galaxies show underprediction in outer regions. "
            "Consider: (1) Increasing A_galaxy, (2) f(r) that doesn't saturate as quickly."
        )
    
    if common_issues.get('rising_residual_with_R', 0) > 5:
        recommendations.append(
            "RISING RESIDUALS WITH RADIUS: Enhancement grows too slowly with radius. "
            "Consider: (1) Reducing r0, (2) Steeper f(r) function."
        )
    
    # Check correlation trends
    if abs(trends.get('rms_vs_V_flat', {}).get('correlation', 0)) > 0.3:
        corr = trends['rms_vs_V_flat']['correlation']
        if corr > 0:
            recommendations.append(
                f"MASS-DEPENDENT FIT QUALITY: RMS correlates with V_flat (r={corr:.2f}). "
                "Massive galaxies fit worse. Consider mass-dependent amplitude scaling."
            )
        else:
            recommendations.append(
                f"MASS-DEPENDENT FIT QUALITY: RMS anti-correlates with V_flat (r={corr:.2f}). "
                "Dwarf galaxies fit worse. Consider different treatment for low-mass systems."
            )
    
    if not recommendations:
        recommendations.append(
            "NO MAJOR SYSTEMATIC ISSUES IDENTIFIED. Current model parameters appear well-optimized. "
            "Remaining scatter likely due to: (1) Observational uncertainties, "
            "(2) Individual galaxy peculiarities, (3) Intrinsic scatter in coherence properties."
        )
    
    return recommendations


def print_report(galaxies: List[GalaxyData], output_dir: Optional[Path] = None) -> None:
    """Generate and print comprehensive report."""
    
    print("=" * 100)
    print("SPARC GALAXY CLASSIFICATION AND DIAGNOSTIC REPORT")
    print("=" * 100)
    
    # Overall statistics
    print(f"\n{'='*100}")
    print("1. OVERALL STATISTICS")
    print("=" * 100)
    
    rms_vals = [g.rms_sigma for g in galaxies]
    rms_mond = [g.rms_mond for g in galaxies]
    wins = sum(1 for g in galaxies if g.wins_sigma)
    
    print(f"\nTotal galaxies: {len(galaxies)}")
    print(f"Σ-Gravity Mean RMS: {np.mean(rms_vals):.2f} km/s (median: {np.median(rms_vals):.2f})")
    print(f"MOND Mean RMS: {np.mean(rms_mond):.2f} km/s (median: {np.median(rms_mond):.2f})")
    print(f"Σ-Gravity wins: {wins}/{len(galaxies)} ({100*wins/len(galaxies):.1f}%)")
    print(f"Improvement vs MOND: {(np.mean(rms_mond) - np.mean(rms_vals))/np.mean(rms_mond)*100:.1f}%")
    
    # Category reports
    categories = [
        ('Mass', lambda g: g.mass_class),
        ('Surface Brightness', lambda g: g.sb_class),
        ('Gas Fraction', lambda g: g.gas_class),
        ('Bulge Fraction', lambda g: g.bulge_class),
        ('Fit Quality', lambda g: g.fit_class)
    ]
    
    category_reports = []
    for cat_name, get_class in categories:
        report = generate_category_report(galaxies, cat_name, get_class)
        category_reports.append(report)
        
        print(f"\n{'='*100}")
        print(f"2. {cat_name.upper()} BREAKDOWN")
        print("=" * 100)
        
        print(f"\n{'Class':<20} {'N':>5} {'RMS (Σ)':>10} {'RMS (M)':>10} {'Wins':>8} {'Win%':>8} {'Improv':>8} {'<Resid>':>10}")
        print("-" * 100)
        
        for cls, stats in sorted(report['classes'].items()):
            print(f"{cls:<20} {stats['count']:>5} "
                  f"{stats['rms_mean']:>10.2f} {stats['mond_rms_mean']:>10.2f} "
                  f"{stats['wins']:>8} {stats['win_rate']*100:>7.1f}% "
                  f"{stats['improvement_vs_mond']:>7.1f}% "
                  f"{stats['mean_residual']:>10.2f}")
    
    # Radial breakdown
    print(f"\n{'='*100}")
    print("3. RADIAL RESIDUAL ANALYSIS")
    print("=" * 100)
    
    for cat_name, get_class in categories[:2]:  # Just mass and SB
        report = next(r for r in category_reports if r['category'] == cat_name)
        print(f"\n{cat_name}:")
        print(f"{'Class':<20} {'Inner Resid':>12} {'Outer Resid':>12} {'Trend (km/s/kpc)':>18}")
        print("-" * 70)
        for cls, stats in sorted(report['classes'].items()):
            print(f"{cls:<20} {stats['inner_residual']:>12.2f} {stats['outer_residual']:>12.2f} "
                  f"{stats['residual_trend']:>18.3f}")
    
    # Problem galaxies
    print(f"\n{'='*100}")
    print("4. WORST-PERFORMING GALAXIES (TOP 20)")
    print("=" * 100)
    
    problems = identify_problem_galaxies(galaxies, n_worst=20)
    
    print(f"\n{'Galaxy':<15} {'RMS(Σ)':>8} {'RMS(M)':>8} {'Win':>5} {'V_flat':>8} {'f_gas':>6} {'f_bulge':>7} {'Issues':<40}")
    print("-" * 120)
    
    for p in problems:
        win_str = "✓" if p['wins_sigma'] else "✗"
        issues_str = ", ".join(p['likely_issues'][:2])
        print(f"{p['name']:<15} {p['rms_sigma']:>8.1f} {p['rms_mond']:>8.1f} {win_str:>5} "
              f"{p['V_flat']:>8.1f} {p['gas_fraction']:>6.2f} {p['bulge_fraction']:>7.2f} {issues_str:<40}")
    
    # Systematic trends
    print(f"\n{'='*100}")
    print("5. SYSTEMATIC TREND ANALYSIS")
    print("=" * 100)
    
    trends = analyze_systematic_trends(galaxies)
    
    print(f"\n{'Correlation':<30} {'r':>8} {'Interpretation':<50}")
    print("-" * 90)
    
    for name, data in trends.items():
        if name != 'strongest_trend' and isinstance(data, dict) and 'correlation' in data:
            print(f"{name:<30} {data['correlation']:>8.3f} {data['interpretation']:<50}")
    
    if 'strongest_trend' in trends:
        st = trends['strongest_trend']
        print(f"\nStrongest trend: {st['name']} (r = {st['correlation']:.3f})")
    
    # Recommendations
    print(f"\n{'='*100}")
    print("6. RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    print("=" * 100)
    
    recommendations = generate_recommendations(category_reports, problems, trends)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    # Save detailed results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save category reports
        with open(output_dir / "category_reports.json", 'w') as f:
            json.dump(category_reports, f, indent=2)
        
        # Save problem galaxies (convert numpy types)
        problems_serializable = []
        for p in problems:
            p_copy = {}
            for k, v in p.items():
                if isinstance(v, (np.bool_, bool)):
                    p_copy[k] = bool(v)
                elif isinstance(v, (np.integer, np.floating)):
                    p_copy[k] = float(v)
                else:
                    p_copy[k] = v
            problems_serializable.append(p_copy)
        
        with open(output_dir / "problem_galaxies.json", 'w') as f:
            json.dump(problems_serializable, f, indent=2)
        
        # Save trends
        with open(output_dir / "systematic_trends.json", 'w') as f:
            json.dump(trends, f, indent=2)
        
        # Save recommendations
        with open(output_dir / "recommendations.txt", 'w') as f:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n\n")
        
        # Save full galaxy data (convert numpy types)
        galaxy_data = []
        for g in galaxies:
            galaxy_data.append({
                'name': g.name,
                'V_flat': float(g.V_flat),
                'R_max': float(g.R_max),
                'gas_fraction': float(g.gas_fraction),
                'bulge_fraction': float(g.bulge_fraction),
                'mass_class': g.mass_class,
                'sb_class': g.sb_class,
                'gas_class': g.gas_class,
                'bulge_class': g.bulge_class,
                'fit_class': g.fit_class,
                'rms_sigma': float(g.rms_sigma),
                'rms_mond': float(g.rms_mond),
                'wins_sigma': bool(g.wins_sigma),
                'mean_residual': float(g.mean_residual),
                'inner_residual': float(g.inner_residual),
                'outer_residual': float(g.outer_residual),
                'residual_trend': float(g.residual_trend)
            })
        
        with open(output_dir / "galaxy_classifications.json", 'w') as f:
            json.dump(galaxy_data, f, indent=2)
        
        print(f"\n\nDetailed results saved to: {output_dir}")


def main():
    import sys
    
    # Parse arguments
    output_dir = None
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])
    
    # Default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "sparc_classification_report"
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    
    print("Loading SPARC galaxies...")
    galaxies = load_sparc_galaxies(data_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Classify
    print("Classifying galaxies...")
    classify_galaxies(galaxies)
    
    # Generate report
    print_report(galaxies, output_dir)


if __name__ == "__main__":
    main()

