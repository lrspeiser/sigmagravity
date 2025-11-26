"""
Hierarchical Bayesian p-Morphology Correlation Test
====================================================

Tests whether the decoherence exponent p correlates with galaxy morphology
using a hierarchical model with partial pooling.

Key advantages over per-galaxy fitting:
1. Partial pooling prevents bound-hitting
2. Jointly estimates intrinsic scatter vs measurement noise
3. Borrows strength across galaxies for poorly-constrained cases
4. Provides proper uncertainty on morphology slope β

Model:
------
Population level:
    μ_p ~ Normal(1.0, 0.5)          # population mean p
    β_morph ~ Normal(0, 0.2)        # morphology slope (key parameter!)
    σ_p ~ HalfNormal(0.5)           # intrinsic scatter in p
    
Galaxy level:
    p_g ~ Normal(μ_p + β_morph × morph_code_g, σ_p)
    
Likelihood:
    v_obs_g ~ Normal(v_pred_g(p_g), σ_obs_g)

If β_morph posterior excludes 0 → correlation detected
If β_morph ≈ 0 but σ_p small → p is universal

Author: Leonard Speiser
Date: 2025-11-26
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Add paths
SCRIPT_DIR = Path(__file__).parent.parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "many_path_model"))

from validation_suite_winding import ValidationSuite

# Check for PyMC
try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    HAS_PYMC = True
    print(f"PyMC version: {pm.__version__}")
except ImportError:
    HAS_PYMC = False
    print("PyMC not installed. Install with: pip install pymc arviz")


# Physical constants
KPC_TO_M = 3.0856776e19
KM_TO_M = 1000.0
G_DAGGER = 1.2e-10  # m/s^2


def classify_morphology(hubble_type_code):
    """
    Classify SPARC Hubble type code into morphological groups.
    Returns: (group_name, group_code 0-4, numeric_T)
    
    Morphology code increases with earlier type (opposite of T):
    4 = Early (S0-Sa), 0 = Irregular (Sm-BCD)
    """
    try:
        T = int(hubble_type_code)
    except (ValueError, TypeError):
        return 'Unknown', -1, -1
    
    if T <= 1:  # S0, Sa
        return 'Early (S0-Sa)', 4, T
    elif T <= 3:  # Sab, Sb
        return 'Early-Spiral (Sab-Sb)', 3, T
    elif T <= 5:  # Sbc, Sc
        return 'Intermediate (Sbc-Sc)', 2, T
    elif T <= 8:  # Scd, Sd, Sdm
        return 'Late-Spiral (Scd-Sdm)', 1, T
    elif T <= 11:  # Sm, Im, BCD
        return 'Irregular (Sm-BCD)', 0, T
    else:
        return 'Unknown', -1, T


def prepare_galaxy_data(max_galaxies=None, min_points=10):
    """
    Load and prepare SPARC data for hierarchical fitting.
    
    Returns list of dicts with R, v_obs, v_bar, v_err, morph_code
    """
    print("Loading SPARC data...")
    output_dir = SCRIPT_DIR / "outputs" / "p_morphology_hierarchical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    print(f"Loaded {len(df_valid)} galaxies")
    
    galaxies = []
    
    for idx, (row_idx, galaxy) in enumerate(df_valid.iterrows()):
        if max_galaxies and len(galaxies) >= max_galaxies:
            break
            
        v_all = galaxy['v_all']
        r_all = galaxy['r_all']
        
        if v_all is None or r_all is None or len(r_all) < min_points:
            continue
        
        # Get baryonic velocity
        v_disk = galaxy.get('v_disk_all', np.zeros_like(v_all))
        v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_all))
        v_gas = galaxy.get('v_gas_all', np.zeros_like(v_all))
        
        if v_disk is None: v_disk = np.zeros_like(v_all)
        if v_bulge is None: v_bulge = np.zeros_like(v_all)
        if v_gas is None: v_gas = np.zeros_like(v_all)
        
        v_bar = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
        
        # Velocity error
        v_err = galaxy.get('e_v_all', v_all * 0.1)
        if v_err is None:
            v_err = v_all * 0.1
        
        # Morphology
        hubble_T = galaxy.get('T', galaxy.get('hubble_T', -999))
        morph_group, morph_code, T_numeric = classify_morphology(hubble_T)
        
        if morph_code < 0:
            continue
        
        # Filter valid points
        mask = (r_all > 0.5) & (v_bar > 1) & (v_err > 0) & np.isfinite(v_all)
        if np.sum(mask) < min_points:
            continue
        
        galaxies.append({
            'name': galaxy.get('Galaxy', galaxy.get('name', f'Galaxy_{idx}')),
            'R': r_all[mask],
            'v_obs': v_all[mask],
            'v_bar': v_bar[mask],
            'v_err': np.maximum(v_err[mask], 1.0),  # minimum 1 km/s error
            'morph_code': morph_code,
            'morph_group': morph_group,
            'T_numeric': T_numeric,
            'n_points': np.sum(mask)
        })
    
    print(f"Prepared {len(galaxies)} galaxies for fitting")
    
    # Summary by morphology
    morph_counts = {}
    for g in galaxies:
        mg = g['morph_group']
        morph_counts[mg] = morph_counts.get(mg, 0) + 1
    print("Morphology distribution:")
    for mg, count in sorted(morph_counts.items()):
        print(f"  {mg}: {count}")
    
    return galaxies, output_dir


def build_hierarchical_model(galaxies, A0_fixed=0.591, ell_0_fixed=4.993, n_coh_fixed=0.5):
    """
    Build hierarchical PyMC model for p-morphology correlation.
    
    Key parameter: β_morph - if posterior excludes 0, correlation exists
    """
    n_galaxies = len(galaxies)
    morph_codes = np.array([g['morph_code'] for g in galaxies])
    
    # Standardize morphology codes for better sampling
    morph_mean = morph_codes.mean()
    morph_std = morph_codes.std()
    morph_standardized = (morph_codes - morph_mean) / morph_std
    
    print(f"\nBuilding hierarchical model for {n_galaxies} galaxies...")
    print(f"Morphology codes: mean={morph_mean:.2f}, std={morph_std:.2f}")
    
    # Prepare data arrays
    # We'll use a "long format" - all data points concatenated
    all_R = []
    all_v_obs = []
    all_v_bar = []
    all_v_err = []
    galaxy_idx = []
    
    for g_idx, gal in enumerate(galaxies):
        n_pts = len(gal['R'])
        all_R.extend(gal['R'])
        all_v_obs.extend(gal['v_obs'])
        all_v_bar.extend(gal['v_bar'])
        all_v_err.extend(gal['v_err'])
        galaxy_idx.extend([g_idx] * n_pts)
    
    all_R = np.array(all_R)
    all_v_obs = np.array(all_v_obs)
    all_v_bar = np.array(all_v_bar)
    all_v_err = np.array(all_v_err)
    galaxy_idx = np.array(galaxy_idx, dtype=int)
    
    print(f"Total data points: {len(all_R)}")
    
    with pm.Model() as model:
        # ============================================
        # Population-level priors (hyperpriors)
        # ============================================
        
        # Population mean p (expect ~0.75 based on global fit)
        mu_p = pm.Normal('mu_p', mu=0.75, sigma=0.3)
        
        # Morphology slope - THE KEY PARAMETER
        # Prior centered at 0 (no correlation), but allows detection
        # Positive β means p increases with morph_code (i.e., p_Early > p_Late)
        beta_morph = pm.Normal('beta_morph', mu=0.0, sigma=0.2)
        
        # Intrinsic scatter in p across galaxies
        sigma_p = pm.HalfNormal('sigma_p', sigma=0.3)
        
        # ============================================
        # Galaxy-level p values (partial pooling)
        # ============================================
        
        # Each galaxy's p is drawn from population distribution
        # Centered on population mean + morphology effect
        p_offset = pm.Normal('p_offset', mu=0, sigma=1, shape=n_galaxies)
        p_galaxy = pm.Deterministic(
            'p_galaxy',
            mu_p + beta_morph * morph_standardized + sigma_p * p_offset
        )
        
        # Soft constraint: keep p in reasonable range [0.2, 2.5]
        # Using a potential (log probability penalty)
        p_penalty = pm.Potential(
            'p_bounds',
            -100 * pt.sum(pt.switch(p_galaxy < 0.2, (0.2 - p_galaxy)**2, 0)) +
            -100 * pt.sum(pt.switch(p_galaxy > 2.5, (p_galaxy - 2.5)**2, 0))
        )
        
        # ============================================
        # Likelihood
        # ============================================
        
        # Get p value for each data point based on galaxy index
        p_per_point = p_galaxy[galaxy_idx]
        
        # Coherence window: C(R) = 1 - [1 + (R/ℓ₀)^p]^{-n_coh}
        # Use log-space for numerical stability: (R/ℓ₀)^p = exp(p * log(R/ℓ₀))
        log_ratio = pt.log(all_R / ell_0_fixed)
        # Clip to prevent extreme values
        log_ratio_clipped = pt.clip(log_ratio, -5, 5)  # R/ℓ₀ in [0.007, 148]
        p_clipped = pt.clip(p_per_point, 0.1, 3.0)
        ratio = pt.exp(p_clipped * log_ratio_clipped)
        ratio_safe = pt.clip(ratio, 1e-10, 1e10)  # Prevent overflow
        C = 1 - (1 + ratio_safe) ** (-n_coh_fixed)
        
        # Boost factor (simplified): K = A₀ × C
        K = A0_fixed * C
        
        # Predicted velocity: v_pred = v_bar × sqrt(1 + K)
        v_pred = all_v_bar * pt.sqrt(1 + K)
        
        # Likelihood
        v_likelihood = pm.Normal(
            'v_obs',
            mu=v_pred,
            sigma=all_v_err,
            observed=all_v_obs
        )
        
        # ============================================
        # Derived quantities
        # ============================================
        
        # Un-standardized slope (for interpretation)
        beta_morph_unstd = pm.Deterministic(
            'beta_morph_unstd',
            beta_morph / morph_std
        )
        
        # Predicted p for each morphology group
        p_early = pm.Deterministic('p_early', mu_p + beta_morph * (4 - morph_mean) / morph_std)
        p_late = pm.Deterministic('p_late', mu_p + beta_morph * (1 - morph_mean) / morph_std)
        p_irregular = pm.Deterministic('p_irregular', mu_p + beta_morph * (0 - morph_mean) / morph_std)
        
        # Difference (for easy interpretation)
        p_early_minus_irregular = pm.Deterministic(
            'p_early_minus_irregular',
            p_early - p_irregular
        )
    
    # Store metadata for later
    model.morph_mean = morph_mean
    model.morph_std = morph_std
    model.n_galaxies = n_galaxies
    model.galaxies = galaxies
    
    return model


def run_hierarchical_inference(model, n_samples=2000, n_tune=1000, n_chains=4, target_accept=0.9):
    """
    Run MCMC inference on the hierarchical model.
    """
    print(f"\nRunning MCMC: {n_samples} samples, {n_tune} tuning, {n_chains} chains")
    print("This may take 10-30 minutes...")
    
    with model:
        # Use NUTS sampler with higher target acceptance for this complex model
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=42,
            return_inferencedata=True,
            progressbar=True
        )
    
    return trace


def analyze_results(trace, model, output_dir):
    """
    Analyze the hierarchical model results.
    """
    print("\n" + "=" * 80)
    print("HIERARCHICAL BAYESIAN ANALYSIS RESULTS")
    print("=" * 80)
    
    # Summary statistics
    summary = az.summary(trace, var_names=['mu_p', 'beta_morph', 'beta_morph_unstd', 
                                           'sigma_p', 'p_early', 'p_late', 'p_irregular',
                                           'p_early_minus_irregular'])
    print("\n=== Key Parameters ===")
    print(summary)
    
    # Extract posterior samples
    beta_samples = trace.posterior['beta_morph'].values.flatten()
    beta_unstd_samples = trace.posterior['beta_morph_unstd'].values.flatten()
    sigma_p_samples = trace.posterior['sigma_p'].values.flatten()
    mu_p_samples = trace.posterior['mu_p'].values.flatten()
    
    # Key test: does β_morph exclude 0?
    beta_mean = beta_samples.mean()
    beta_std = beta_samples.std()
    prob_positive = (beta_samples > 0).mean()
    
    # 95% credible interval
    beta_ci_low = np.percentile(beta_samples, 2.5)
    beta_ci_high = np.percentile(beta_samples, 97.5)
    
    print("\n=== MAIN RESULT: Morphology Slope β ===")
    print(f"β_morph (standardized): {beta_mean:.4f} ± {beta_std:.4f}")
    print(f"95% CI: [{beta_ci_low:.4f}, {beta_ci_high:.4f}]")
    print(f"P(β > 0): {prob_positive:.3f}")
    
    # Interpretation
    print("\n=== Interpretation ===")
    if beta_ci_low > 0:
        print("✓ SIGNIFICANT POSITIVE CORRELATION DETECTED")
        print("  → p increases from Irregular to Early types")
        print("  → Supports interaction network interpretation")
        result = "significant_positive"
    elif beta_ci_high < 0:
        print("✓ SIGNIFICANT NEGATIVE CORRELATION DETECTED")
        print("  → p decreases from Irregular to Early types")
        print("  → Opposite to prediction!")
        result = "significant_negative"
    elif prob_positive > 0.9:
        print("⚠ SUGGESTIVE POSITIVE CORRELATION (not significant)")
        print(f"  → 90% probability β > 0, but 95% CI includes 0")
        result = "suggestive_positive"
    elif prob_positive < 0.1:
        print("⚠ SUGGESTIVE NEGATIVE CORRELATION (not significant)")
        result = "suggestive_negative"
    else:
        print("✗ NO SIGNIFICANT CORRELATION")
        print("  → p appears universal across morphologies")
        print("  → May indicate fundamental quantum gravity origin")
        result = "null"
    
    # Intrinsic scatter
    sigma_p_mean = sigma_p_samples.mean()
    sigma_p_ci = np.percentile(sigma_p_samples, [2.5, 97.5])
    print(f"\n=== Intrinsic Scatter ===")
    print(f"σ_p: {sigma_p_mean:.3f} (95% CI: [{sigma_p_ci[0]:.3f}, {sigma_p_ci[1]:.3f}])")
    
    if sigma_p_mean < 0.2:
        print("→ Small scatter: p is tightly constrained across galaxies")
    elif sigma_p_mean < 0.5:
        print("→ Moderate scatter: some galaxy-to-galaxy variation")
    else:
        print("→ Large scatter: significant galaxy-to-galaxy variation in p")
    
    # Population mean
    mu_p_mean = mu_p_samples.mean()
    mu_p_ci = np.percentile(mu_p_samples, [2.5, 97.5])
    print(f"\n=== Population Mean p ===")
    print(f"μ_p: {mu_p_mean:.3f} (95% CI: [{mu_p_ci[0]:.3f}, {mu_p_ci[1]:.3f}])")
    print(f"Global fit value: p = 0.757")
    
    # Predicted p by morphology
    print("\n=== Predicted p by Morphology Group ===")
    for var_name, label in [('p_early', 'Early (S0-Sa)'), 
                            ('p_late', 'Late (Scd-Sdm)'),
                            ('p_irregular', 'Irregular (Sm-BCD)')]:
        samples = trace.posterior[var_name].values.flatten()
        print(f"  {label}: {samples.mean():.3f} ± {samples.std():.3f}")
    
    # Save results
    results = {
        'mu_p': {'mean': float(mu_p_mean), 'std': float(mu_p_samples.std()),
                 'ci_95': [float(mu_p_ci[0]), float(mu_p_ci[1])]},
        'beta_morph': {'mean': float(beta_mean), 'std': float(beta_std),
                       'ci_95': [float(beta_ci_low), float(beta_ci_high)],
                       'prob_positive': float(prob_positive)},
        'sigma_p': {'mean': float(sigma_p_mean), 
                    'ci_95': [float(sigma_p_ci[0]), float(sigma_p_ci[1])]},
        'result': result,
        'n_galaxies': int(model.n_galaxies),
        'interpretation': {
            'significant': bool(result.startswith('significant')),
            'direction': 'positive' if 'positive' in result else ('negative' if 'negative' in result else 'none'),
            'p_universal': bool(result == 'null' and sigma_p_mean < 0.3)
        }
    }
    
    with open(output_dir / "hierarchical_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, trace


def create_diagnostic_plots(trace, model, results, output_dir):
    """
    Create diagnostic and results plots.
    """
    print("\n=== Creating Diagnostic Plots ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: β_morph posterior
    ax1 = axes[0, 0]
    beta_samples = trace.posterior['beta_morph'].values.flatten()
    ax1.hist(beta_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='β = 0 (no correlation)')
    ax1.axvline(beta_samples.mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Mean = {beta_samples.mean():.3f}')
    
    # Shade 95% CI
    ci_low, ci_high = np.percentile(beta_samples, [2.5, 97.5])
    ax1.axvspan(ci_low, ci_high, alpha=0.2, color='green', label='95% CI')
    
    ax1.set_xlabel('β_morph (morphology slope)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Morphology Effect on p', fontsize=14)
    ax1.legend(fontsize=9)
    
    # Panel 2: μ_p posterior
    ax2 = axes[0, 1]
    mu_samples = trace.posterior['mu_p'].values.flatten()
    ax2.hist(mu_samples, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(0.757, color='purple', linestyle='--', linewidth=2, label='Global fit (0.757)')
    ax2.axvline(mu_samples.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean = {mu_samples.mean():.3f}')
    ax2.set_xlabel('μ_p (population mean)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Population Mean p', fontsize=14)
    ax2.legend(fontsize=9)
    
    # Panel 3: σ_p posterior
    ax3 = axes[0, 2]
    sigma_samples = trace.posterior['sigma_p'].values.flatten()
    ax3.hist(sigma_samples, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(sigma_samples.mean(), color='red', linestyle='-', linewidth=2,
                label=f'Mean = {sigma_samples.mean():.3f}')
    ax3.set_xlabel('σ_p (intrinsic scatter)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Intrinsic Scatter in p', fontsize=14)
    ax3.legend(fontsize=9)
    
    # Panel 4: Predicted p by morphology
    ax4 = axes[1, 0]
    morph_labels = ['Irregular', 'Late', 'Intermediate', 'Early-Spiral', 'Early']
    morph_codes = [0, 1, 2, 3, 4]
    
    # Calculate predicted p for each morphology code
    mu_p = trace.posterior['mu_p'].values
    beta = trace.posterior['beta_morph'].values
    morph_mean = model.morph_mean
    morph_std = model.morph_std
    
    predicted_p = []
    for mc in morph_codes:
        p_samples = mu_p + beta * (mc - morph_mean) / morph_std
        predicted_p.append(p_samples.flatten())
    
    # Box plot
    bp = ax4.boxplot(predicted_p, labels=morph_labels, patch_artist=True)
    colors = ['blue', 'green', 'gold', 'orange', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.axhline(0.757, color='purple', linestyle='--', linewidth=2, label='Global fit')
    ax4.set_ylabel('Predicted p', fontsize=12)
    ax4.set_xlabel('Morphology', fontsize=12)
    ax4.set_title('Predicted p by Morphology (Hierarchical)', fontsize=14)
    ax4.tick_params(axis='x', rotation=15)
    ax4.legend(fontsize=9)
    
    # Panel 5: Individual galaxy p values
    ax5 = axes[1, 1]
    p_galaxy_samples = trace.posterior['p_galaxy'].values
    p_galaxy_mean = p_galaxy_samples.mean(axis=(0, 1))
    p_galaxy_std = p_galaxy_samples.std(axis=(0, 1))
    
    morph_codes_data = np.array([g['morph_code'] for g in model.galaxies])
    
    # Add jitter for visibility
    jitter = np.random.normal(0, 0.1, len(morph_codes_data))
    
    ax5.errorbar(morph_codes_data + jitter, p_galaxy_mean, yerr=p_galaxy_std,
                fmt='o', alpha=0.5, markersize=4, capsize=0)
    ax5.axhline(0.757, color='purple', linestyle='--', linewidth=2)
    
    # Add trend line from posterior
    x_trend = np.array([0, 4])
    beta_mean = trace.posterior['beta_morph'].values.mean()
    mu_mean = trace.posterior['mu_p'].values.mean()
    y_trend = mu_mean + beta_mean * (x_trend - morph_mean) / morph_std
    ax5.plot(x_trend, y_trend, 'r-', linewidth=2, alpha=0.7, label='Posterior trend')
    
    ax5.set_xlabel('Morphology Code (0=Irregular, 4=Early)', fontsize=12)
    ax5.set_ylabel('Galaxy p (mean ± std)', fontsize=12)
    ax5.set_title('Individual Galaxy p Values', fontsize=14)
    ax5.legend(fontsize=9)
    
    # Panel 6: Trace plot for β_morph (convergence check)
    ax6 = axes[1, 2]
    for chain in range(trace.posterior['beta_morph'].shape[0]):
        ax6.plot(trace.posterior['beta_morph'].values[chain, :], alpha=0.5, label=f'Chain {chain+1}')
    ax6.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Sample', fontsize=12)
    ax6.set_ylabel('β_morph', fontsize=12)
    ax6.set_title('Trace Plot (Convergence Check)', fontsize=14)
    ax6.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / "hierarchical_p_morphology.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Also save to figures folder
    figures_dir = REPO_ROOT / "figures"
    fig_path = figures_dir / "hierarchical_p_morphology.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Plot also saved to: {fig_path}")
    
    plt.close()
    
    # Additional: ArviZ diagnostic plots
    print("Creating ArviZ diagnostic plots...")
    
    # Trace plot
    az.plot_trace(trace, var_names=['mu_p', 'beta_morph', 'sigma_p'])
    plt.savefig(output_dir / "trace_plot.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # Posterior plot
    az.plot_posterior(trace, var_names=['mu_p', 'beta_morph', 'sigma_p', 'p_early_minus_irregular'],
                      ref_val={'beta_morph': 0})
    plt.savefig(output_dir / "posterior_plot.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Diagnostic plots saved")


def main():
    """Run the complete hierarchical Bayesian analysis."""
    
    print("=" * 80)
    print("HIERARCHICAL BAYESIAN p-MORPHOLOGY CORRELATION TEST")
    print("=" * 80)
    
    if not HAS_PYMC:
        print("\nERROR: PyMC not installed. Please run:")
        print("  pip install pymc arviz")
        return None
    
    # Prepare data
    galaxies, output_dir = prepare_galaxy_data(max_galaxies=None, min_points=10)
    
    if len(galaxies) < 20:
        print("ERROR: Not enough galaxies for hierarchical analysis")
        return None
    
    # Build model
    model = build_hierarchical_model(galaxies)
    
    # Run inference (reduce samples for faster initial run)
    trace = run_hierarchical_inference(
        model, 
        n_samples=1000,  # Reduce for faster run; increase to 2000+ for publication
        n_tune=500,      # Reduce for faster run; increase to 1000+ for publication
        n_chains=4,
        target_accept=0.9
    )
    
    # Save trace
    trace.to_netcdf(output_dir / "trace.nc")
    print(f"Trace saved to: {output_dir / 'trace.nc'}")
    
    # Analyze results
    results, trace = analyze_results(trace, model, output_dir)
    
    # Create plots
    create_diagnostic_plots(trace, model, results, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"""
Hierarchical Bayesian Analysis Complete
=======================================
Galaxies: {results['n_galaxies']}
Population mean p: {results['mu_p']['mean']:.3f} ± {results['mu_p']['std']:.3f}
Intrinsic scatter σ_p: {results['sigma_p']['mean']:.3f}

MORPHOLOGY EFFECT:
  β_morph = {results['beta_morph']['mean']:.4f} ± {results['beta_morph']['std']:.4f}
  95% CI: [{results['beta_morph']['ci_95'][0]:.4f}, {results['beta_morph']['ci_95'][1]:.4f}]
  P(β > 0) = {results['beta_morph']['prob_positive']:.3f}

CONCLUSION: {results['result'].upper()}
""")
    
    if results['interpretation']['significant']:
        print("→ Significant correlation detected between p and morphology")
    elif results['interpretation']['p_universal']:
        print("→ p appears universal: no morphology dependence, small scatter")
        print("→ Suggests fundamental quantum gravity origin")
    else:
        print("→ Inconclusive: no significant correlation, but scatter present")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
