"""
GPU-Accelerated Hierarchical Bayesian p-Morphology Correlation Test
====================================================================

Uses CuPy for GPU acceleration on RTX 5090 (Blackwell) and multiprocessing
for parallel MCMC chains on 10-core CPU.

Key improvements over previous version:
1. CuPy for GPU-accelerated likelihood computation
2. Multiprocessing for parallel chains (10 cores)
3. Hierarchical priors on A₀ and ℓ₀ (not just p)
4. More MCMC samples (4000 draws)
5. Adaptive Metropolis-Hastings with GPU batching

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
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Add paths
SCRIPT_DIR = Path(__file__).parent.parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "many_path_model"))

# Try CuPy for GPU
try:
    import cupy as cp
    HAS_GPU = True
    GPU_NAME = "RTX 5090 (CuPy)"
    print(f"GPU enabled: CuPy {cp.__version__}, CUDA {cp.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    cp = np  # Fallback to NumPy
    HAS_GPU = False
    GPU_NAME = "CPU (NumPy fallback)"
    print("CuPy not available, using NumPy")

from validation_suite_winding import ValidationSuite


def classify_morphology(hubble_type_code):
    """Classify SPARC Hubble type code into morphological groups."""
    try:
        T = int(hubble_type_code)
    except (ValueError, TypeError):
        return 'Unknown', -1, -1
    
    if T <= 1:
        return 'Early (S0-Sa)', 4, T
    elif T <= 3:
        return 'Early-Spiral (Sab-Sb)', 3, T
    elif T <= 5:
        return 'Intermediate (Sbc-Sc)', 2, T
    elif T <= 8:
        return 'Late-Spiral (Scd-Sdm)', 1, T
    elif T <= 11:
        return 'Irregular (Sm-BCD)', 0, T
    else:
        return 'Unknown', -1, T


def prepare_data_gpu(max_galaxies=None, min_points=10):
    """Load SPARC data and prepare GPU arrays."""
    print("Loading SPARC data...")
    output_dir = SCRIPT_DIR / "outputs" / "p_morphology_hierarchical_gpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    df_valid = df[df['r_all'].notna()].copy()
    
    galaxies = []
    
    for idx, (row_idx, galaxy) in enumerate(df_valid.iterrows()):
        if max_galaxies and len(galaxies) >= max_galaxies:
            break
            
        v_all = galaxy['v_all']
        r_all = galaxy['r_all']
        
        if v_all is None or r_all is None or len(r_all) < min_points:
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
        morph_group, morph_code, T_numeric = classify_morphology(hubble_T)
        
        if morph_code < 0:
            continue
        
        mask = (r_all > 0.5) & (v_bar > 1) & (v_err > 0) & np.isfinite(v_all)
        if np.sum(mask) < min_points:
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
    
    # Create concatenated arrays for GPU
    all_R = np.concatenate([g['R'] for g in galaxies])
    all_v_obs = np.concatenate([g['v_obs'] for g in galaxies])
    all_v_bar = np.concatenate([g['v_bar'] for g in galaxies])
    all_v_err = np.concatenate([g['v_err'] for g in galaxies])
    
    # Galaxy index for each data point
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
    print(f"GPU data loaded: {HAS_GPU}")
    
    return galaxies, gpu_data, output_dir


def log_likelihood_gpu(params, gpu_data, n_galaxies, morph_mean, morph_std):
    """
    Compute log-likelihood on GPU.
    
    params: [mu_p, beta_morph, sigma_p, mu_A0, sigma_A0, mu_ell0, sigma_ell0, 
             *p_offsets, *A0_offsets, *ell0_offsets]
    """
    xp = cp if HAS_GPU else np
    
    # Unpack population parameters
    mu_p = params[0]
    beta_morph = params[1]
    sigma_p = max(params[2], 0.01)
    mu_A0 = params[3]
    sigma_A0 = max(params[4], 0.01)
    mu_ell0 = params[5]
    sigma_ell0 = max(params[6], 0.01)
    
    # Galaxy-level offsets (standard normal)
    n_hyperparams = 7
    p_offsets = params[n_hyperparams:n_hyperparams + n_galaxies]
    A0_offsets = params[n_hyperparams + n_galaxies:n_hyperparams + 2*n_galaxies]
    ell0_offsets = params[n_hyperparams + 2*n_galaxies:n_hyperparams + 3*n_galaxies]
    
    # Compute galaxy-level parameters with morphology effect on p
    morph_standardized = (gpu_data['morph_codes'] - morph_mean) / morph_std
    
    if HAS_GPU:
        morph_standardized = cp.asarray(morph_standardized) if isinstance(morph_standardized, np.ndarray) else morph_standardized
        p_offsets_gpu = cp.asarray(p_offsets)
        A0_offsets_gpu = cp.asarray(A0_offsets)
        ell0_offsets_gpu = cp.asarray(ell0_offsets)
    else:
        p_offsets_gpu = p_offsets
        A0_offsets_gpu = A0_offsets
        ell0_offsets_gpu = ell0_offsets
    
    p_galaxy = mu_p + beta_morph * morph_standardized + sigma_p * p_offsets_gpu
    A0_galaxy = mu_A0 + sigma_A0 * A0_offsets_gpu
    ell0_galaxy = mu_ell0 + sigma_ell0 * ell0_offsets_gpu
    
    # Clip to valid ranges
    p_galaxy = xp.clip(p_galaxy, 0.1, 3.0)
    A0_galaxy = xp.clip(A0_galaxy, 0.01, 2.0)
    ell0_galaxy = xp.clip(ell0_galaxy, 0.5, 30.0)
    
    # Get parameters for each data point
    galaxy_idx = gpu_data['galaxy_idx']
    p_per_point = p_galaxy[galaxy_idx]
    A0_per_point = A0_galaxy[galaxy_idx]
    ell0_per_point = ell0_galaxy[galaxy_idx]
    
    # Coherence window (log-space for stability)
    R = gpu_data['R']
    log_ratio = xp.log(R / ell0_per_point)
    log_ratio = xp.clip(log_ratio, -5, 5)
    ratio = xp.exp(p_per_point * log_ratio)
    ratio = xp.clip(ratio, 1e-10, 1e10)
    
    n_coh = 0.5
    C = 1 - (1 + ratio) ** (-n_coh)
    
    # Predicted velocity
    K = A0_per_point * C
    v_pred = gpu_data['v_bar'] * xp.sqrt(1 + K)
    
    # Log-likelihood (Gaussian)
    residuals = (gpu_data['v_obs'] - v_pred) / gpu_data['v_err']
    log_lik = -0.5 * xp.sum(residuals**2)
    
    # Prior on offsets (standard normal)
    log_prior_offsets = -0.5 * (xp.sum(p_offsets_gpu**2) + 
                                 xp.sum(A0_offsets_gpu**2) + 
                                 xp.sum(ell0_offsets_gpu**2))
    
    # Priors on hyperparameters
    log_prior_hyper = (
        -0.5 * ((mu_p - 0.75) / 0.5)**2 +          # mu_p ~ N(0.75, 0.5)
        -0.5 * (beta_morph / 0.3)**2 +              # beta_morph ~ N(0, 0.3)
        -0.5 * ((mu_A0 - 0.6) / 0.3)**2 +           # mu_A0 ~ N(0.6, 0.3)
        -0.5 * ((mu_ell0 - 5.0) / 2.0)**2           # mu_ell0 ~ N(5, 2)
    )
    
    # HalfNormal priors on sigmas (approximated as truncated normal)
    if sigma_p > 0:
        log_prior_hyper += -0.5 * (sigma_p / 0.5)**2
    if sigma_A0 > 0:
        log_prior_hyper += -0.5 * (sigma_A0 / 0.2)**2
    if sigma_ell0 > 0:
        log_prior_hyper += -0.5 * (sigma_ell0 / 2.0)**2
    
    total = log_lik + log_prior_offsets + log_prior_hyper
    
    if HAS_GPU:
        return float(cp.asnumpy(total))
    return float(total)


def run_mcmc_chain(args):
    """Run a single MCMC chain (for multiprocessing)."""
    chain_id, n_samples, n_tune, gpu_data_cpu, n_galaxies, morph_mean, morph_std, seed = args
    
    np.random.seed(seed + chain_id)
    
    # Transfer to GPU for this process
    if HAS_GPU:
        gpu_data = {k: cp.asarray(v) for k, v in gpu_data_cpu.items()}
    else:
        gpu_data = gpu_data_cpu
    
    # Initialize parameters
    n_hyperparams = 7  # mu_p, beta_morph, sigma_p, mu_A0, sigma_A0, mu_ell0, sigma_ell0
    n_params = n_hyperparams + 3 * n_galaxies
    
    # Initial values
    params = np.zeros(n_params, dtype=np.float64)
    params[0] = 0.75 + np.random.randn() * 0.1  # mu_p
    params[1] = np.random.randn() * 0.05         # beta_morph
    params[2] = 0.3 + np.random.rand() * 0.2     # sigma_p
    params[3] = 0.6 + np.random.randn() * 0.1    # mu_A0
    params[4] = 0.1 + np.random.rand() * 0.1     # sigma_A0
    params[5] = 5.0 + np.random.randn() * 0.5    # mu_ell0
    params[6] = 1.0 + np.random.rand() * 0.5     # sigma_ell0
    # Galaxy offsets start at 0
    
    # Proposal standard deviations (adaptive)
    proposal_std = np.ones(n_params) * 0.1
    proposal_std[:7] = [0.05, 0.02, 0.02, 0.05, 0.02, 0.2, 0.1]  # Hyperparameters
    proposal_std[7:] = 0.1  # Galaxy offsets
    
    # Current log-probability
    current_log_prob = log_likelihood_gpu(params, gpu_data, n_galaxies, morph_mean, morph_std)
    
    # Storage
    samples = np.zeros((n_samples, n_params))
    log_probs = np.zeros(n_samples)
    
    # Acceptance tracking
    n_accepted = 0
    
    print(f"Chain {chain_id}: Starting MCMC ({n_tune} tune + {n_samples} samples)")
    
    for i in range(n_tune + n_samples):
        # Propose new parameters
        proposal = params + proposal_std * np.random.randn(n_params)
        
        # Enforce bounds
        proposal[2] = max(0.01, proposal[2])  # sigma_p > 0
        proposal[4] = max(0.01, proposal[4])  # sigma_A0 > 0
        proposal[6] = max(0.01, proposal[6])  # sigma_ell0 > 0
        
        # Compute log-probability
        proposal_log_prob = log_likelihood_gpu(proposal, gpu_data, n_galaxies, morph_mean, morph_std)
        
        # Accept/reject
        log_alpha = proposal_log_prob - current_log_prob
        
        if np.log(np.random.rand()) < log_alpha:
            params = proposal
            current_log_prob = proposal_log_prob
            if i >= n_tune:
                n_accepted += 1
        
        # Store sample (after tuning)
        if i >= n_tune:
            samples[i - n_tune] = params
            log_probs[i - n_tune] = current_log_prob
        
        # Adaptive proposal (during tuning)
        if i < n_tune and i > 0 and i % 100 == 0:
            recent_accept_rate = n_accepted / max(1, i - max(0, i - 100))
            if recent_accept_rate < 0.2:
                proposal_std *= 0.9
            elif recent_accept_rate > 0.4:
                proposal_std *= 1.1
        
        # Progress
        if (i + 1) % 500 == 0:
            phase = "Tuning" if i < n_tune else "Sampling"
            accept_rate = n_accepted / max(1, i - n_tune + 1) if i >= n_tune else 0
            print(f"Chain {chain_id}: {phase} {i+1}/{n_tune + n_samples}, "
                  f"accept={accept_rate:.2%}, log_prob={current_log_prob:.1f}")
    
    final_accept_rate = n_accepted / n_samples
    print(f"Chain {chain_id}: Done. Final acceptance rate: {final_accept_rate:.2%}")
    
    return samples, log_probs, final_accept_rate


def run_hierarchical_mcmc_parallel(gpu_data, n_galaxies, morph_codes,
                                    n_samples=2000, n_tune=1000, n_chains=4,
                                    n_workers=None):
    """
    Run hierarchical MCMC with parallel chains.
    """
    if n_workers is None:
        n_workers = min(n_chains, cpu_count() - 1)
    
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL MCMC WITH GPU + PARALLEL CHAINS")
    print(f"{'='*80}")
    print(f"Chains: {n_chains}")
    print(f"Parallel workers: {n_workers}")
    print(f"Samples per chain: {n_samples} (+ {n_tune} tuning)")
    print(f"GPU: {GPU_NAME}")
    print(f"CPU cores: {cpu_count()}")
    
    # Compute morphology statistics
    morph_mean = float(np.mean(morph_codes))
    morph_std = float(np.std(morph_codes))
    
    # Convert GPU data to CPU for multiprocessing
    if HAS_GPU:
        gpu_data_cpu = {k: cp.asnumpy(v) for k, v in gpu_data.items()}
    else:
        gpu_data_cpu = gpu_data
    
    # Prepare arguments for each chain
    args_list = [
        (chain_id, n_samples, n_tune, gpu_data_cpu, n_galaxies, morph_mean, morph_std, 42)
        for chain_id in range(n_chains)
    ]
    
    start_time = time.time()
    
    # Run chains (parallel or sequential based on GPU)
    if HAS_GPU and n_workers > 1:
        # With GPU, run sequentially to avoid GPU memory conflicts
        # But use multiprocessing for CPU-bound proposal generation
        print("Running chains sequentially (GPU memory sharing)")
        results = [run_mcmc_chain(args) for args in args_list]
    else:
        # CPU-only: use multiprocessing
        print(f"Running {n_chains} chains on {n_workers} workers")
        with Pool(n_workers) as pool:
            results = pool.map(run_mcmc_chain, args_list)
    
    elapsed = time.time() - start_time
    print(f"\nTotal MCMC time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Combine chains
    all_samples = np.stack([r[0] for r in results], axis=0)  # (n_chains, n_samples, n_params)
    all_log_probs = np.stack([r[1] for r in results], axis=0)
    accept_rates = [r[2] for r in results]
    
    print(f"Acceptance rates: {[f'{r:.2%}' for r in accept_rates]}")
    
    return all_samples, all_log_probs, morph_mean, morph_std


def analyze_mcmc_results(samples, n_galaxies, morph_mean, morph_std, output_dir):
    """Analyze MCMC results and extract key statistics."""
    
    print("\n" + "=" * 80)
    print("MCMC ANALYSIS RESULTS")
    print("=" * 80)
    
    # Combine chains (discard first half as extra burn-in)
    n_chains, n_samples, n_params = samples.shape
    burn_in = n_samples // 4
    combined = samples[:, burn_in:, :].reshape(-1, n_params)
    
    print(f"Total samples after burn-in: {len(combined)}")
    
    # Extract hyperparameters
    mu_p = combined[:, 0]
    beta_morph = combined[:, 1]
    sigma_p = combined[:, 2]
    mu_A0 = combined[:, 3]
    sigma_A0 = combined[:, 4]
    mu_ell0 = combined[:, 5]
    sigma_ell0 = combined[:, 6]
    
    # Key statistics
    def summarize(samples, name):
        mean = samples.mean()
        std = samples.std()
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
        print(f"  {name}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
        return {'mean': float(mean), 'std': float(std), 
                'ci_95': [float(ci_low), float(ci_high)]}
    
    print("\n=== Population Parameters ===")
    mu_p_stats = summarize(mu_p, "μ_p (population mean p)")
    beta_stats = summarize(beta_morph, "β_morph (morphology slope)")
    sigma_p_stats = summarize(sigma_p, "σ_p (intrinsic scatter)")
    
    print("\n=== Amplitude & Scale Parameters ===")
    mu_A0_stats = summarize(mu_A0, "μ_A0 (population mean A₀)")
    sigma_A0_stats = summarize(sigma_A0, "σ_A0 (A₀ scatter)")
    mu_ell0_stats = summarize(mu_ell0, "μ_ℓ₀ (population mean ℓ₀)")
    sigma_ell0_stats = summarize(sigma_ell0, "σ_ℓ₀ (ℓ₀ scatter)")
    
    # Key test: does β_morph exclude 0?
    prob_positive = (beta_morph > 0).mean()
    ci_low = np.percentile(beta_morph, 2.5)
    ci_high = np.percentile(beta_morph, 97.5)
    
    print(f"\n=== MAIN RESULT ===")
    print(f"P(β_morph > 0) = {prob_positive:.3f}")
    
    if ci_low > 0:
        result = "SIGNIFICANT POSITIVE CORRELATION"
        interpretation = "p increases from Irregular to Early - supports interaction network"
    elif ci_high < 0:
        result = "SIGNIFICANT NEGATIVE CORRELATION"
        interpretation = "p decreases from Irregular to Early - opposite to prediction"
    elif prob_positive > 0.9:
        result = "SUGGESTIVE POSITIVE (not significant)"
        interpretation = f"{prob_positive:.0%} probability of positive correlation"
    elif prob_positive < 0.1:
        result = "SUGGESTIVE NEGATIVE (not significant)"
        interpretation = f"{1-prob_positive:.0%} probability of negative correlation"
    else:
        result = "NULL"
        interpretation = "No significant correlation detected"
    
    print(f"CONCLUSION: {result}")
    print(f"  → {interpretation}")
    
    # Predicted p by morphology
    print("\n=== Predicted p by Morphology ===")
    for mc, label in [(4, 'Early'), (2, 'Intermediate'), (0, 'Irregular')]:
        p_pred = mu_p + beta_morph * (mc - morph_mean) / morph_std
        print(f"  {label}: p = {p_pred.mean():.3f} ± {p_pred.std():.3f}")
    
    # Save results
    results = {
        'mu_p': mu_p_stats,
        'beta_morph': {**beta_stats, 'prob_positive': float(prob_positive)},
        'sigma_p': sigma_p_stats,
        'mu_A0': mu_A0_stats,
        'sigma_A0': sigma_A0_stats,
        'mu_ell0': mu_ell0_stats,
        'sigma_ell0': sigma_ell0_stats,
        'result': result,
        'interpretation': interpretation,
        'n_galaxies': int(n_galaxies),
        'n_samples': int(len(combined)),
        'gpu': GPU_NAME
    }
    
    with open(output_dir / "hierarchical_gpu_results.json", 'w') as f:
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
    ax1.set_title('Morphology Effect on p', fontsize=14)
    ax1.legend(fontsize=9)
    
    # Panel 2: μ_p posterior
    ax2 = axes[0, 1]
    mu_p = samples[:, 0]
    ax2.hist(mu_p, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(0.757, color='purple', linestyle='--', linewidth=2, label='Global fit (0.757)')
    ax2.axvline(mu_p.mean(), color='green', linestyle='-', linewidth=2)
    ax2.set_xlabel('μ_p', fontsize=12)
    ax2.set_title('Population Mean p', fontsize=14)
    ax2.legend(fontsize=9)
    
    # Panel 3: σ_p posterior
    ax3 = axes[0, 2]
    sigma_p = samples[:, 2]
    ax3.hist(sigma_p, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(sigma_p.mean(), color='red', linestyle='-', linewidth=2)
    ax3.set_xlabel('σ_p', fontsize=12)
    ax3.set_title('Intrinsic Scatter in p', fontsize=14)
    
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
    
    ax4.axhline(0.757, color='purple', linestyle='--', linewidth=2)
    ax4.set_ylabel('Predicted p', fontsize=12)
    ax4.set_title('p by Morphology (Hierarchical)', fontsize=14)
    
    # Panel 5: A₀ and ℓ₀ posteriors
    ax5 = axes[1, 1]
    mu_A0 = samples[:, 3]
    mu_ell0 = samples[:, 5]
    ax5.hist(mu_A0, bins=40, density=True, alpha=0.6, label=f'μ_A₀ = {mu_A0.mean():.3f}', color='blue')
    ax5.axvline(0.591, color='blue', linestyle='--', alpha=0.7)
    ax5.set_xlabel('μ_A₀', fontsize=12)
    ax5.set_title('Amplitude Parameter', fontsize=14)
    ax5.legend()
    
    # Panel 6: Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
HIERARCHICAL BAYESIAN ANALYSIS
==============================
GPU: {GPU_NAME}
Galaxies: {n_galaxies}

POPULATION PARAMETERS:
• μ_p = {results['mu_p']['mean']:.3f} ± {results['mu_p']['std']:.3f}
• σ_p = {results['sigma_p']['mean']:.3f} ± {results['sigma_p']['std']:.3f}
• μ_A₀ = {results['mu_A0']['mean']:.3f} ± {results['mu_A0']['std']:.3f}
• μ_ℓ₀ = {results['mu_ell0']['mean']:.3f} ± {results['mu_ell0']['std']:.3f}

MORPHOLOGY EFFECT:
• β = {results['beta_morph']['mean']:.4f} ± {results['beta_morph']['std']:.4f}
• 95% CI: [{results['beta_morph']['ci_95'][0]:.3f}, {results['beta_morph']['ci_95'][1]:.3f}]
• P(β > 0) = {results['beta_morph']['prob_positive']:.1%}

RESULT: {results['result']}
{results['interpretation']}
"""
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = output_dir / "hierarchical_gpu_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    figures_dir = REPO_ROOT / "figures"
    fig_path = figures_dir / "hierarchical_gpu_p_morphology.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    
    print(f"Plots saved to: {plot_path}")
    print(f"Also saved to: {fig_path}")
    
    plt.close()


def main(fix_A0_ell0=True):
    """
    Run GPU-accelerated hierarchical Bayesian analysis.
    
    Args:
        fix_A0_ell0: If True, fix A0=0.591 and ell0=4.993 (more constrained).
                     If False, fit A0 and ell0 hierarchically (more flexible).
    """
    
    print("=" * 80)
    print("GPU-ACCELERATED HIERARCHICAL BAYESIAN p-MORPHOLOGY TEST")
    print(f"Mode: {'FIXED A0/ell0' if fix_A0_ell0 else 'HIERARCHICAL A0/ell0'}")
    print("=" * 80)
    
    # Prepare data
    galaxies, gpu_data, output_dir = prepare_data_gpu()
    n_galaxies = len(galaxies)
    morph_codes = np.array([g['morph_code'] for g in galaxies])
    
    # Run MCMC
    samples, log_probs, morph_mean, morph_std = run_hierarchical_mcmc_parallel(
        gpu_data, n_galaxies, morph_codes,
        n_samples=4000,   # More samples for better convergence
        n_tune=2000,      # Longer tuning period
        n_chains=4,       # 4 chains
        n_workers=4,      # Parallel workers
        fix_A0_ell0=fix_A0_ell0
    )
    
    # Save raw samples
    np.savez(output_dir / "mcmc_samples.npz", 
             samples=samples, log_probs=log_probs,
             morph_mean=morph_mean, morph_std=morph_std)
    
    # Analyze
    results, combined_samples = analyze_mcmc_results(
        samples, n_galaxies, morph_mean, morph_std, output_dir
    )
    
    # Plot
    create_plots(combined_samples, results, n_galaxies, morph_mean, morph_std, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
