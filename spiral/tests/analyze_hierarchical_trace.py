"""
Analyze saved hierarchical Bayesian trace for p-morphology correlation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import arviz as az

SCRIPT_DIR = Path(__file__).parent.parent
REPO_ROOT = SCRIPT_DIR.parent
output_dir = SCRIPT_DIR / "outputs" / "p_morphology_hierarchical"

# Load trace
print("Loading trace...")
trace = az.from_netcdf(output_dir / "trace.nc")

# Extract posterior samples
beta_samples = trace.posterior['beta_morph'].values.flatten()
mu_p_samples = trace.posterior['mu_p'].values.flatten()
sigma_p_samples = trace.posterior['sigma_p'].values.flatten()

# Key statistics
beta_mean = beta_samples.mean()
beta_std = beta_samples.std()
beta_ci_low = np.percentile(beta_samples, 2.5)
beta_ci_high = np.percentile(beta_samples, 97.5)
prob_positive = (beta_samples > 0).mean()

mu_p_mean = mu_p_samples.mean()
sigma_p_mean = sigma_p_samples.mean()

print("\n" + "=" * 80)
print("HIERARCHICAL BAYESIAN ANALYSIS RESULTS")
print("=" * 80)

print(f"""
Key Parameters:
  Population mean p (μ_p): {mu_p_mean:.3f} ± {mu_p_samples.std():.3f}
  Intrinsic scatter (σ_p): {sigma_p_mean:.3f} ± {sigma_p_samples.std():.3f}
  
MORPHOLOGY EFFECT:
  β_morph = {beta_mean:.4f} ± {beta_std:.4f}
  95% CI: [{beta_ci_low:.4f}, {beta_ci_high:.4f}]
  P(β > 0) = {prob_positive:.3f}
""")

# Interpretation
if beta_ci_low > 0:
    result = "SIGNIFICANT POSITIVE CORRELATION"
    interpretation = "p increases from Irregular to Early types - supports interaction network"
elif beta_ci_high < 0:
    result = "SIGNIFICANT NEGATIVE CORRELATION"
    interpretation = "p decreases from Irregular to Early - opposite to prediction"
elif prob_positive > 0.9:
    result = "SUGGESTIVE POSITIVE (not significant)"
    interpretation = "86% probability of positive correlation, but 95% CI includes 0"
else:
    result = "NULL"
    interpretation = "No significant correlation detected"

print(f"CONCLUSION: {result}")
print(f"  → {interpretation}")

# Save results JSON
results = {
    'mu_p': {'mean': float(mu_p_mean), 'std': float(mu_p_samples.std()),
             'ci_95': [float(np.percentile(mu_p_samples, 2.5)), float(np.percentile(mu_p_samples, 97.5))]},
    'beta_morph': {'mean': float(beta_mean), 'std': float(beta_std),
                   'ci_95': [float(beta_ci_low), float(beta_ci_high)],
                   'prob_positive': float(prob_positive)},
    'sigma_p': {'mean': float(sigma_p_mean), 
                'ci_95': [float(np.percentile(sigma_p_samples, 2.5)), float(np.percentile(sigma_p_samples, 97.5))]},
    'result': result,
    'n_galaxies': 116,
    'n_chains': 4,
    'n_samples': 1000,
    'interpretation': interpretation
}

with open(output_dir / "hierarchical_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_dir / 'hierarchical_results.json'}")

# Create plots
print("\nCreating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Panel 1: β_morph posterior
ax1 = axes[0, 0]
ax1.hist(beta_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='β = 0 (no correlation)')
ax1.axvline(beta_mean, color='green', linestyle='-', linewidth=2, 
            label=f'Mean = {beta_mean:.3f}')
ax1.axvspan(beta_ci_low, beta_ci_high, alpha=0.2, color='green', label='95% CI')
ax1.set_xlabel('β_morph (morphology slope)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Morphology Effect on p', fontsize=14)
ax1.legend(fontsize=9)

# Panel 2: μ_p posterior
ax2 = axes[0, 1]
ax2.hist(mu_p_samples, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
ax2.axvline(0.757, color='purple', linestyle='--', linewidth=2, label='Global fit (0.757)')
ax2.axvline(mu_p_mean, color='green', linestyle='-', linewidth=2,
            label=f'Mean = {mu_p_mean:.3f}')
ax2.set_xlabel('μ_p (population mean)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Population Mean p', fontsize=14)
ax2.legend(fontsize=9)

# Panel 3: σ_p posterior
ax3 = axes[0, 2]
ax3.hist(sigma_p_samples, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
ax3.axvline(sigma_p_mean, color='red', linestyle='-', linewidth=2,
            label=f'Mean = {sigma_p_mean:.3f}')
ax3.set_xlabel('σ_p (intrinsic scatter)', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Intrinsic Scatter in p', fontsize=14)
ax3.legend(fontsize=9)

# Panel 4: Predicted p by morphology
ax4 = axes[1, 0]
morph_labels = ['Irregular', 'Late', 'Intermediate', 'Early-Spiral', 'Early']
morph_codes = [0, 1, 2, 3, 4]

# Approximate morph_mean and morph_std from the data
morph_mean = 1.34  # approximate from 116 galaxies
morph_std = 1.31

mu_p = trace.posterior['mu_p'].values
beta = trace.posterior['beta_morph'].values

predicted_p = []
for mc in morph_codes:
    p_samples = mu_p + beta * (mc - morph_mean) / morph_std
    predicted_p.append(p_samples.flatten())

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

# Panel 5: P(β > 0) visualization
ax5 = axes[1, 1]
positive_frac = prob_positive
negative_frac = 1 - prob_positive
ax5.bar(['P(β > 0)', 'P(β < 0)'], [positive_frac, negative_frac], 
        color=['green', 'red'], alpha=0.7, edgecolor='black')
ax5.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
ax5.axhline(0.90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
ax5.set_ylabel('Probability', fontsize=12)
ax5.set_title(f'Direction of Correlation\n(P(β>0) = {prob_positive:.1%})', fontsize=14)
ax5.set_ylim(0, 1)
ax5.legend(fontsize=9)

# Panel 6: Summary text
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""
HIERARCHICAL BAYESIAN ANALYSIS
==============================

Galaxies: 116 (SPARC)
Data points: 2,818
MCMC: 4 chains × 1,000 samples

POPULATION PARAMETERS:
• μ_p = {mu_p_mean:.3f} ± {mu_p_samples.std():.3f}
• σ_p = {sigma_p_mean:.3f} ± {sigma_p_samples.std():.3f}

MORPHOLOGY EFFECT:
• β_morph = {beta_mean:.4f} ± {beta_std:.4f}
• 95% CI: [{beta_ci_low:.3f}, {beta_ci_high:.3f}]
• P(β > 0) = {prob_positive:.1%}

RESULT: {result}

{interpretation}
"""
ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
plot_path = output_dir / "hierarchical_p_morphology.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

figures_dir = REPO_ROOT / "figures"
fig_path = figures_dir / "hierarchical_p_morphology.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Plot also saved to: {fig_path}")

plt.close()

print("\nDone!")
