"""
Constrained GPU-Accelerated Hierarchical Bayesian p-Morphology Test
====================================================================

This version fixes A₀=0.591 and ℓ₀=4.993 at calibrated values for direct 
comparison with the PyMC analysis. Only {μ_p, β_morph, σ_p} are fitted
at the population level.

This eliminates the parameter degeneracy issue seen in the unconstrained
GPU model.

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
import time
import warnings
warnings.filterwarnings('ignore')

# Add paths
SCRIPT_DIR = Path(__file__).parent.parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "many_path_model"))

# CuPy for GPU
try:
    import cupy as cp
    HAS_GPU = True
    GPU_NAME = "RTX 5090 (CuPy)"
    print(f"GPU enabled: CuPy {cp.__version__}, CUDA {cp.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    cp = np
    HAS_GPU = False
    GPU_NAME = "CPU (NumPy)"
    print("CuPy not available, using NumPy")

from validation_suite_winding import ValidationSuite

# Fixed parameters from calibration
A0_FIXED = 0.591
ELL0_FIXED = 4.993
N_COH_FIXED = 0.5


def classify_morphology(hubble_type_code):
    """Classify SPARC Hubble type into morphological groups."""
    try:
        T = int(hubble_type_code)
    except (ValueError, TypeError):
        return 'Unknown', -1
    
    if T <= 1:
        return 'Early (S0-Sa)', 4
    elif T <= 3:
        return 'Early-Spiral (Sab-Sb)', 3
    elif T <= 5:
        return 'Intermediate (Sbc-Sc)', 2
    elif T <= 8:
        return 'Late-Spiral (Scd-Sdm)', 1
    elif T <= 11:
        return 'Irregular (Sm-BCD)', 0
    else:
        return 'Unknown', -1


def prepare_data():
    """Load SPARC data and prepare arrays."""
    print("Loading SPARC data...")
    output_dir = SCRIPT_DIR / "outputs" / "p_morphology_gpu_constrained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    galaxies = []
    
    for idx, (row_idx, galaxy) in enumerate(df_valid.iterrows()):
        v_all = galaxy['v_all']
        r_all = galaxy['r_all']
        
        if v_all is None or r_all is None or len(r_all) < 10:
            continue
        
        v_disk = galaxy.get('v_disk_all', np.zeros_like(v_all))
        v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_all))
        v_gas = galaxy.get('v_gas_all', np.zeros_like(v_all))
        
        if v_disk is None: v_disk = np.zeros_like(v_all)
        if v_bulge is None: v_bulge = np.zeros_like(v_all)
        if v_gas is None: v_gas = np.zeros_like(v_all)
        
        v_bar = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
        
        v_err = galaxy.get('e_v_all', v_all * 0.1)
        if v_err is None:
            v_err = v_all * 0.1
        
        hubble_T = galaxy.get('T', galaxy.get('hubble_T', -999))
        morph_group, morph_code = classify_morphology(hubble_T)
        
        if morph_code < 0:
            continue
        
        mask = (r_all > 0.5) & (v_bar > 1) & (v_err > 0) & np.isfinite(v_all)
        if np.sum(mask) < 10:
            continue
        
        galaxies.append({
            'name': galaxy.get('Galaxy', f'Galaxy_{idx}'),
            'R': r_all[mask].astype(np.float32),
            'v_obs': v_all[mask].astype(np.float32),
            'v_bar': v_bar[mask].astype(np.float32),
            'v_err': np.maximum(v_err[mask], 1.0).astype(np.float32),
            'morph_code': morph_code,
            'morph_group': morph_group,
        })
    
    print(f"Prepared {len(galaxies)} galaxies")
    
    # Concatenate for GPU
    all_R = np.concatenate([g['R'] for g in galaxies])
    all_v_obs = np.concatenate([g['v_obs'] for g in galaxies])
    all_v_bar = np.concatenate([g['v_bar'] for g in galaxies])
    all_v_err = np.concatenate([g['v_err'] for g in galaxies])
    
    galaxy_idx = []
    for g_idx, g in enumerate(galaxies):
        galaxy_idx.extend([g_idx] * len(g['R']))
    galaxy_idx = np.array(galaxy_idx, dtype=np.int32)
    
    morph_codes = np.array([g['morph_code'] for g in galaxies], dtype=np.float32)
    
    # Transfer to GPU
    if HAS_GPU:
        gpu_data = {
            'R': cp.asarray(all_R),
            'v_obs': cp.asarray(all_v_obs),
            'v_bar': cp.asarray(all_v_bar),
            'v_err': cp.asarray(all_v_err),
            'galaxy_idx': cp.asarray(galaxy_idx),
            'morph_codes': cp.asarray(morph_codes),
        }
    else:
        gpu_data = {
            'R': all_R,
            'v_obs': all_v_obs,
            'v_bar': all_v_bar,
            'v_err': all_v_err,
            'galaxy_idx': galaxy_idx,
            'morph_codes': morph_codes,
        }
    
    print(f"Total data points: {len(all_R)}")
    
    return galaxies, gpu_data, output_dir


def log_likelihood_constrained(params, gpu_data, n_galaxies, morph_mean, morph_std):
    """
    Compute log-likelihood with FIXED A₀ and ℓ₀.
    
    params: [mu_p, beta_morph, sigma_p, *p_offsets]
    """
    xp = cp if HAS_GPU else np
    
    # Unpack population parameters
    mu_p = params[0]
    beta_morph = params[1]
    sigma_p = max(params[2], 0.01)
    
    # Galaxy-level p offsets
    p_offsets = params[3:3 + n_galaxies]
    
    # Compute galaxy-level p with morphology effect
    morph_standardized = (gpu_data['morph_codes'] - morph_mean) / morph_std
    
    if HAS_GPU:
        p_offsets_gpu = cp.asarray(p_offsets)
        morph_standardized = cp.asarray(morph_standardized) if isinstance(morph_standardized, np.ndarray) else morph_standardized
    else:
        p_offsets_gpu = p_offsets
    
    p_galaxy = mu_p + beta_morph * morph_standardized + sigma_p * p_offsets_gpu
    
    # Clip p to valid range
    p_galaxy = xp.clip(p_galaxy, 0.1, 3.0)
    
    # Get p for each data point
    galaxy_idx = gpu_data['galaxy_idx']
    p_per_point = p_galaxy[galaxy_idx]
    
    # Coherence window with FIXED parameters
    R = gpu_data['R']
    log_ratio = xp.log(R / ELL0_FIXED)
    log_ratio = xp.clip(log_ratio, -5, 5)
    ratio = xp.exp(p_per_point * log_ratio)
    ratio = xp.clip(ratio, 1e-10, 1e10)
    
    C = 1 - (1 + ratio) ** (-N_COH_FIXED)
    
    # Predicted velocity with FIXED A₀
    K = A0_FIXED * C
    v_pred = gpu_data['v_bar'] * xp.sqrt(1 + K)
    
    # Log-likelihood
    residuals = (gpu_data['v_obs'] - v_pred) / gpu_data['v_err']
    log_lik = -0.5 * xp.sum(residuals**2)
    
    # Prior on offsets (standard normal)
    log_prior_offsets = -0.5 * xp.sum(p_offsets_gpu**2)
    
    # Priors on hyperparameters
    log_prior_hyper = (
        -0.5 * ((mu_p - 0.75) / 0.5)**2 +     # mu_p ~ N(0.75, 0.5)
        -0.5 * (beta_morph / 0.3)**2 +         # beta_morph ~ N(0, 0.3)
        -0.5 * (sigma_p / 0.5)**2              # sigma_p ~ HalfNormal(0.5)
    )
    
    total = log_lik + log_prior_offsets + log_prior_hyper
    
    if HAS_GPU:
        return float(cp.asnumpy(total))
    return float(total)


def run_mcmc_constrained(gpu_data, n_galaxies, morph_mean, morph_std,
                         n_samples=4000, n_tune=2000, seed=42):
    """Run MCMC with constrained model."""
    
    np.random.seed(seed)
    
    # Parameters: [mu_p, beta_morph, sigma_p, *p_offsets]
    n_hyperparams = 3
    n_params = n_hyperparams + n_galaxies
    
    # Initialize
    params = np.zeros(n_params, dtype=np.float64)
    params[0] = 0.75 + np.random.randn() * 0.1   # mu_p
    params[1] = np.random.randn() * 0.05          # beta_morph
    params[2] = 0.3 + np.random.rand() * 0.2      # sigma_p
    # p_offsets start at 0
    
    # Proposal standard deviations
    proposal_std = np.ones(n_params) * 0.1
    proposal_std[:3] = [0.03, 0.015, 0.02]  # Tighter proposals for hyperparams
    
    current_log_prob = log_likelihood_constrained(
        params, gpu_data, n_galaxies, morph_mean, morph_std
    )
    
    # Storage for all chains
    all_samples = []
    all_log_probs = []
    
    n_chains = 4
    
    for chain in range(n_chains):
        print(f"\n=== Chain {chain + 1}/{n_chains} ===")
        
        # Reset for each chain with different initialization
        np.random.seed(seed + chain * 1000)
        params = np.zeros(n_params, dtype=np.float64)
        params[0] = 0.75 + np.random.randn() * 0.15
        params[1] = np.random.randn() * 0.1
        params[2] = 0.3 + np.random.rand() * 0.3
        
        proposal_std_chain = proposal_std.copy()
        
        current_log_prob = log_likelihood_constrained(
            params, gpu_data, n_galaxies, morph_mean, morph_std
        )
        
        samples = np.zeros((n_samples, n_params))
        log_probs = np.zeros(n_samples)
        n_accepted = 0
        
        for i in range(n_tune + n_samples):
            # Propose
            proposal = params + proposal_std_chain * np.random.randn(n_params)
            proposal[2] = max(0.01, proposal[2])  # sigma_p > 0
            
            proposal_log_prob = log_likelihood_constrained(
                proposal, gpu_data, n_galaxies, morph_mean, morph_std
            )
            
            # Accept/reject
            if np.log(np.random.rand()) < proposal_log_prob - current_log_prob:
                params = proposal
                current_log_prob = proposal_log_prob
                if i >= n_tune:
                    n_accepted += 1
            
            # Store
            if i >= n_tune:
                samples[i - n_tune] = params
                log_probs[i - n_tune] = current_log_prob
            
            # Adapt during tuning
            if i < n_tune and i > 0 and i % 100 == 0:
                window_start = max(0, i - 200)
                recent_accept = n_accepted / max(1, i - window_start) if i > n_tune else 0.25
                if i < n_tune:
                    # Estimate from recent proposals
                    pass
                if i % 200 == 0:
                    if n_accepted / max(1, i) < 0.15:
                        proposal_std_chain *= 0.8
                    elif n_accepted / max(1, i) > 0.35:
                        proposal_std_chain *= 1.2
            
            # Progress
            if (i + 1) % 1000 == 0:
                phase = "Tune" if i < n_tune else "Sample"
                acc_rate = n_accepted / max(1, i - n_tune + 1) if i >= n_tune else 0
                print(f"  {phase} {i+1}/{n_tune + n_samples}, "
                      f"accept={acc_rate:.1%}, log_prob={current_log_prob:.1f}")
        
        final_accept = n_accepted / n_samples
        print(f"  Chain {chain + 1} done. Accept rate: {final_accept:.1%}")
        
        all_samples.append(samples)
        all_log_probs.append(log_probs)
    
    return np.array(all_samples), np.array(all_log_probs)


def analyze_results(samples, n_galaxies, morph_mean, morph_std, output_dir):
    """Analyze MCMC results."""
    
    print("\n" + "=" * 80)
    print("CONSTRAINED HIERARCHICAL ANALYSIS RESULTS")
    print(f"Fixed parameters: A₀ = {A0_FIXED}, ℓ₀ = {ELL0_FIXED} kpc")
    print("=" * 80)
    
    # Combine chains (discard burn-in)
    n_chains, n_samples, n_params = samples.shape
    burn_in = n_samples // 4
    combined = samples[:, burn_in:, :].reshape(-1, n_params)
    
    print(f"Total samples after burn-in: {len(combined)}")
    
    # Extract parameters
    mu_p = combined[:, 0]
    beta_morph = combined[:, 1]
    sigma_p = combined[:, 2]
    
    def summarize(arr, name):
        mean = arr.mean()
        std = arr.std()
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        print(f"  {name}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
        return {'mean': float(mean), 'std': float(std), 
                'ci_95': [float(ci_low), float(ci_high)]}
    
    print("\n=== Population Parameters ===")
    mu_p_stats = summarize(mu_p, "μ_p")
    beta_stats = summarize(beta_morph, "β_morph")
    sigma_p_stats = summarize(sigma_p, "σ_p")
    
    # Key test
    prob_positive = (beta_morph > 0).mean()
    ci_low = np.percentile(beta_morph, 2.5)
    ci_high = np.percentile(beta_morph, 97.5)
    
    print(f"\n=== MAIN RESULT ===")
    print(f"P(β_morph > 0) = {prob_positive:.3f}")
    
    if ci_low > 0:
        result = "SIGNIFICANT POSITIVE CORRELATION"
        interp = "p increases from Irregular to Early - supports interaction network"
    elif ci_high < 0:
        result = "SIGNIFICANT NEGATIVE CORRELATION"
        interp = "p decreases from Irregular to Early - opposite to prediction"
    elif prob_positive > 0.9:
        result = "SUGGESTIVE POSITIVE (not significant)"
        interp = f"{prob_positive:.0%} probability of positive correlation"
    elif prob_positive < 0.1:
        result = "SUGGESTIVE NEGATIVE (not significant)"
        interp = f"{1-prob_positive:.0%} probability of negative correlation"
    else:
        result = "NULL"
        interp = "No significant correlation detected"
    
    print(f"CONCLUSION: {result}")
    print(f"  → {interp}")
    
    # Predicted p by morphology
    print("\n=== Predicted p by Morphology ===")
    for mc, label in [(4, 'Early'), (2, 'Intermediate'), (0, 'Irregular')]:
        p_pred = mu_p + beta_morph * (mc - morph_mean) / morph_std
        print(f"  {label}: p = {p_pred.mean():.3f} ± {p_pred.std():.3f}")
    
    # Save
    results = {
        'mu_p': mu_p_stats,
        'beta_morph': {**beta_stats, 'prob_positive': float(prob_positive)},
        'sigma_p': sigma_p_stats,
        'A0_fixed': A0_FIXED,
        'ell0_fixed': ELL0_FIXED,
        'result': result,
        'interpretation': interp,
        'n_galaxies': int(n_galaxies),
        'n_samples': int(len(combined)),
        'gpu': GPU_NAME
    }
    
    with open(output_dir / "constrained_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, combined


def create_plots(samples, results, n_galaxies, morph_mean, morph_std, output_dir):
    """Create diagnostic plots."""
    
    print("\nCreating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: β_morph posterior
    ax1 = axes[0, 0]
    beta = samples[:, 1]
    ax1.hist(beta, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='β = 0')
    ax1.axvline(beta.mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Mean = {beta.mean():.3f}')
    ci_low, ci_high = np.percentile(beta, [2.5, 97.5])
    ax1.axvspan(ci_low, ci_high, alpha=0.2, color='green', label='95% CI')
    ax1.set_xlabel('β_morph', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Morphology Effect on p\n(Constrained Model)', fontsize=14)
    ax1.legend(fontsize=9)
    
    # Panel 2: μ_p posterior
    ax2 = axes[0, 1]
    mu_p = samples[:, 0]
    ax2.hist(mu_p, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(0.757, color='purple', linestyle='--', linewidth=2, label='Global fit (0.757)')
    ax2.axvline(mu_p.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean = {mu_p.mean():.3f}')
    ax2.set_xlabel('μ_p', fontsize=12)
    ax2.set_title('Population Mean p', fontsize=14)
    ax2.legend(fontsize=9)
    
    # Panel 3: σ_p posterior
    ax3 = axes[0, 2]
    sigma_p = samples[:, 2]
    ax3.hist(sigma_p, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(sigma_p.mean(), color='red', linestyle='-', linewidth=2,
                label=f'Mean = {sigma_p.mean():.3f}')
    ax3.set_xlabel('σ_p', fontsize=12)
    ax3.set_title('Intrinsic Scatter in p', fontsize=14)
    ax3.legend(fontsize=9)
    
    # Panel 4: Predicted p by morphology
    ax4 = axes[1, 0]
    morph_labels = ['Irregular', 'Late', 'Interm.', 'Early-Sp', 'Early']
    morph_codes = [0, 1, 2, 3, 4]
    
    predicted_p = []
    for mc in morph_codes:
        p_pred = mu_p + beta * (mc - morph_mean) / morph_std
        predicted_p.append(p_pred)
    
    bp = ax4.boxplot(predicted_p, tick_labels=morph_labels, patch_artist=True)
    colors = ['blue', 'green', 'gold', 'orange', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.axhline(0.757, color='purple', linestyle='--', linewidth=2, label='Global fit')
    ax4.set_ylabel('Predicted p', fontsize=12)
    ax4.set_title('p by Morphology (Constrained)', fontsize=14)
    ax4.legend(fontsize=9)
    
    # Panel 5: Comparison with PyMC
    ax5 = axes[1, 1]
    
    # PyMC results (from earlier)
    pymc_beta = 0.082
    pymc_ci = [-0.064, 0.217]
    gpu_beta = beta.mean()
    gpu_ci = [ci_low, ci_high]
    
    methods = ['PyMC\n(10 hrs)', f'GPU\n({GPU_NAME})']
    betas = [pymc_beta, gpu_beta]
    errors = [[pymc_beta - pymc_ci[0], gpu_beta - gpu_ci[0]],
              [pymc_ci[1] - pymc_beta, gpu_ci[1] - gpu_beta]]
    
    x = np.arange(len(methods))
    ax5.bar(x, betas, yerr=errors, capsize=10, color=['steelblue', 'orange'], alpha=0.7)
    ax5.axhline(0, color='red', linestyle='--', linewidth=2)
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods)
    ax5.set_ylabel('β_morph', fontsize=12)
    ax5.set_title('Comparison: PyMC vs GPU\n(Both Constrained)', fontsize=14)
    
    # Panel 6: Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
CONSTRAINED HIERARCHICAL ANALYSIS
=================================
Fixed: A₀ = {A0_FIXED}, ℓ₀ = {ELL0_FIXED} kpc
GPU: {GPU_NAME}
Galaxies: {n_galaxies}

POPULATION PARAMETERS:
• μ_p = {results['mu_p']['mean']:.3f} ± {results['mu_p']['std']:.3f}
• σ_p = {results['sigma_p']['mean']:.3f} ± {results['sigma_p']['std']:.3f}

MORPHOLOGY EFFECT:
• β = {results['beta_morph']['mean']:.4f} ± {results['beta_morph']['std']:.4f}
• 95% CI: [{results['beta_morph']['ci_95'][0]:.3f}, {results['beta_morph']['ci_95'][1]:.3f}]
• P(β > 0) = {results['beta_morph']['prob_positive']:.1%}

RESULT: {results['result']}
{results['interpretation']}

COMPARISON WITH PyMC:
• PyMC β = 0.082 ± 0.074, P(β>0) = 86%
• GPU  β = {results['beta_morph']['mean']:.3f} ± {results['beta_morph']['std']:.3f}
"""
    ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = output_dir / "constrained_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    figures_dir = REPO_ROOT / "figures"
    fig_path = figures_dir / "p_morphology_constrained_gpu.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    
    print(f"Plots saved to: {plot_path}")
    print(f"Also saved to: {fig_path}")
    
    plt.close()


def main():
    """Run constrained GPU hierarchical analysis."""
    
    print("=" * 80)
    print("CONSTRAINED GPU HIERARCHICAL BAYESIAN p-MORPHOLOGY TEST")
    print(f"Fixed: A₀ = {A0_FIXED}, ℓ₀ = {ELL0_FIXED} kpc")
    print("=" * 80)
    
    start_time = time.time()
    
    # Prepare data
    galaxies, gpu_data, output_dir = prepare_data()
    n_galaxies = len(galaxies)
    morph_codes = np.array([g['morph_code'] for g in galaxies])
    morph_mean = float(morph_codes.mean())
    morph_std = float(morph_codes.std())
    
    print(f"\nMorphology: mean={morph_mean:.2f}, std={morph_std:.2f}")
    
    # Run MCMC
    samples, log_probs = run_mcmc_constrained(
        gpu_data, n_galaxies, morph_mean, morph_std,
        n_samples=4000,
        n_tune=2000,
        seed=42
    )
    
    elapsed = time.time() - start_time
    print(f"\nTotal MCMC time: {elapsed:.1f}s")
    
    # Save samples
    np.savez(output_dir / "mcmc_samples.npz", 
             samples=samples, log_probs=log_probs,
             morph_mean=morph_mean, morph_std=morph_std)
    
    # Analyze
    results, combined = analyze_results(
        samples, n_galaxies, morph_mean, morph_std, output_dir
    )
    
    # Plot
    create_plots(combined, results, n_galaxies, morph_mean, morph_std, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
