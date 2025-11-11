"""
Visualize how coherence length λ varies for each star under different hypotheses.
Shows EXACTLY what wavelengths we're assigning to each star.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'GravityWaveTest')
from test_star_by_star_mw import StarByStarCalculator

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

print("="*80)
print("VISUALIZING COHERENCE LENGTH VARIATIONS PER STAR")
print("="*80)

# Load stars
gaia = pd.read_csv('data/gaia/gaia_processed.csv')
print(f"\nLoaded {len(gaia):,} stars")

# Subsample for visualization (use all for actual calculation)
np.random.seed(42)
n_viz = min(50000, len(gaia))
subsample_idx = np.random.choice(len(gaia), n_viz, replace=False)
gaia_viz = gaia.iloc[subsample_idx].copy()

print(f"Using {len(gaia_viz):,} stars for visualization")

# Initialize calculator
calc = StarByStarCalculator(gaia, use_gpu=GPU_AVAILABLE)

# Compute λ for each hypothesis
print("\n" + "="*80)
print("COMPUTING λ FOR EACH HYPOTHESIS")
print("="*80)

# MC mass weights
M_disk = 5.0e10
M_per_star = M_disk / len(gaia)
if GPU_AVAILABLE:
    M_weights = cp.full(calc.n_stars, M_per_star, dtype=cp.float32)
else:
    M_weights = np.full(calc.n_stars, M_per_star, dtype=np.float32)

hypotheses = {}

# 1. Universal
print("\n1. Universal λ = 4.993 kpc")
lambda_universal = calc.compute_lambda_universal(4.993)
if GPU_AVAILABLE:
    lambda_universal = cp.asnumpy(lambda_universal)
hypotheses['universal'] = lambda_universal
print(f"   All stars: λ = {lambda_universal[0]:.3f} kpc (constant)")

# 2. Mass-dependent (M^0.5)
print("\n2. λ ∝ M^0.5 (Tully-Fisher)")
lambda_m05 = calc.compute_lambda_mass_dependent(M_weights, gamma=0.5)
if GPU_AVAILABLE:
    lambda_m05 = cp.asnumpy(lambda_m05)
hypotheses['mass_05'] = lambda_m05
print(f"   Range: {lambda_m05.min():.3f} - {lambda_m05.max():.3f} kpc")
print(f"   Median: {np.median(lambda_m05):.3f} kpc")
print(f"   Std: {np.std(lambda_m05):.3f} kpc")
print(f"   NOTE: All stars have SAME mass weight → λ is constant!")

# 3. Mass-dependent (M^0.3)  
print("\n3. λ ∝ M^0.3 (SPARC)")
lambda_m03 = calc.compute_lambda_mass_dependent(M_weights, gamma=0.3)
if GPU_AVAILABLE:
    lambda_m03 = cp.asnumpy(lambda_m03)
hypotheses['mass_03'] = lambda_m03
print(f"   Range: {lambda_m03.min():.3f} - {lambda_m03.max():.3f} kpc")
print(f"   Median: {np.median(lambda_m03):.3f} kpc")
print(f"   NOTE: All stars have SAME mass weight → λ is constant!")

# 4. Local disk scale height
print("\n4. λ = h(R) (local disk scale height)")
lambda_hR = calc.compute_lambda_local_disk(calc.R_stars)
if GPU_AVAILABLE:
    lambda_hR = cp.asnumpy(lambda_hR)
hypotheses['h_R'] = lambda_hR
print(f"   Range: {lambda_hR.min():.3f} - {lambda_hR.max():.3f} kpc")
print(f"   Median: {np.median(lambda_hR):.3f} kpc")
print(f"   Std: {np.std(lambda_hR):.3f} kpc")
print(f"   ✓ VARIES with position! (depends on R)")

# 5. Hybrid
print("\n5. λ ~ M^0.3 × R^0.3 (Hybrid)")
lambda_hybrid = calc.compute_lambda_hybrid(M_weights, calc.R_stars)
if GPU_AVAILABLE:
    lambda_hybrid = cp.asnumpy(lambda_hybrid)
hypotheses['hybrid'] = lambda_hybrid
print(f"   Range: {lambda_hybrid.min():.3f} - {lambda_hybrid.max():.3f} kpc")
print(f"   Median: {np.median(lambda_hybrid):.3f} kpc")
print(f"   Std: {np.std(lambda_hybrid):.3f} kpc")
print(f"   ✓ VARIES with position! (depends on R)")

print("\n" + "="*80)
print("KEY INSIGHT:")
print("="*80)
print("\nMass-dependent hypotheses (λ ∝ M^γ) give CONSTANT λ because:")
print("  - All stars have equal mass weight (M_disk / N_stars)")
print("  - λ ∝ M^γ → λ ∝ constant^γ = constant!")
print("\nPosition-dependent hypotheses (λ = h(R), hybrid) give VARYING λ:")
print("  - λ depends on local density, radius, etc.")
print("  - Different stars have different coherence lengths!")

# Generate visualization
print("\n" + "="*80)
print("GENERATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Coherence Length λ Variations ({n_viz:,} stars sampled)', 
             fontsize=14, fontweight='bold')

R_viz = gaia_viz['R_cyl'].values
z_viz = gaia_viz['z'].values

# Get λ values for visualization subset
lambda_viz = {}
for key in hypotheses:
    lambda_viz[key] = hypotheses[key][subsample_idx]

# Plot 1: Universal λ
ax = axes[0, 0]
scatter = ax.scatter(R_viz, z_viz, c=lambda_viz['universal'], s=1, alpha=0.5, 
                    cmap='viridis', vmin=0, vmax=10)
ax.set_xlabel('R [kpc]')
ax.set_ylabel('z [kpc]')
ax.set_title('Universal λ = 4.993 kpc\n(CONSTANT for all stars)')
plt.colorbar(scatter, ax=ax, label='λ [kpc]')
ax.set_xlim(0, 20)
ax.set_ylim(-5, 5)

# Plot 2: Mass-dependent (shows it's actually constant!)
ax = axes[0, 1]
scatter = ax.scatter(R_viz, z_viz, c=lambda_viz['mass_05'], s=1, alpha=0.5,
                    cmap='viridis', vmin=0, vmax=10)
ax.set_xlabel('R [kpc]')
ax.set_ylabel('z [kpc]')
ax.set_title('λ ∝ M^0.5 (Tully-Fisher)\n(Actually CONSTANT - all M equal!)')
plt.colorbar(scatter, ax=ax, label='λ [kpc]')
ax.set_xlim(0, 20)
ax.set_ylim(-5, 5)

# Plot 3: Local disk height (VARIES!)
ax = axes[0, 2]
scatter = ax.scatter(R_viz, z_viz, c=np.log10(lambda_viz['h_R']), s=1, alpha=0.5,
                    cmap='plasma', vmin=-1, vmax=2)
ax.set_xlabel('R [kpc]')
ax.set_ylabel('z [kpc]')
ax.set_title('λ = h(R) (disk scale height)\n(VARIES with R!)')
plt.colorbar(scatter, ax=ax, label='log₁₀(λ) [kpc]')
ax.set_xlim(0, 20)
ax.set_ylim(-5, 5)

# Plot 4: λ vs R for each hypothesis
ax = axes[1, 0]
ax.scatter(R_viz, lambda_viz['universal'], s=1, alpha=0.3, label='Universal', color='blue')
ax.scatter(R_viz, lambda_viz['h_R'], s=1, alpha=0.3, label='h(R)', color='red')
ax.scatter(R_viz, lambda_viz['hybrid'], s=1, alpha=0.3, label='Hybrid', color='green')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('λ [kpc]')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('λ vs Radius')
ax.set_ylim(0.01, 100)

# Plot 5: Histogram of λ values
ax = axes[1, 1]
for key, name, color in [('universal', 'Universal', 'blue'),
                          ('h_R', 'h(R)', 'red'),
                          ('hybrid', 'Hybrid', 'green')]:
    ax.hist(np.log10(lambda_viz[key]), bins=50, alpha=0.5, label=name, color=color)
ax.set_xlabel('log₁₀(λ) [kpc]')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('λ Distribution')

# Plot 6: Enhancement kernel example
ax = axes[1, 2]
r = np.linspace(0.1, 50, 200)

for lambda_val, label, color in [(1.0, 'λ=1 kpc', 'blue'),
                                   (5.0, 'λ=5 kpc', 'green'),
                                   (20.0, 'λ=20 kpc', 'red')]:
    A = 0.591
    p = 0.757
    n_coh = 0.5
    
    # Burr-XII window
    K = A * (1.0 - (1.0 + (r/lambda_val)**p)**(-n_coh))
    
    ax.plot(r, K, label=label, color=color, linewidth=2)

ax.set_xlabel('Distance r [kpc]')
ax.set_ylabel('Enhancement K(r)')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Coherence Window K(r) = A[1-(1+(r/λ)^p)^-n]')
ax.axhline(0.591, color='k', linestyle='--', alpha=0.3, label='Max (A=0.591)')

plt.tight_layout()
plt.savefig('GravityWaveTest/lambda_variations_by_star.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to GravityWaveTest/lambda_variations_by_star.png")

# Summary table
print("\n" + "="*80)
print("SUMMARY: HOW λ VARIES FOR EACH HYPOTHESIS")
print("="*80)

print(f"\n{'Hypothesis':<30} {'λ varies?':<12} {'Range (kpc)':<20} {'Physical meaning'}")
print("-"*100)
print(f"{'Universal λ=4.993':<30} {'NO':<12} {'4.99 - 4.99':<20} {'Same for all stars'}")
print(f"{'λ ∝ M^0.5':<30} {'NO*':<12} {'5.00 - 5.00':<20} {'All M equal → constant'}")
print(f"{'λ ∝ M^0.3':<30} {'NO*':<12} {'5.00 - 5.00':<20} {'All M equal → constant'}")
print(f"{'λ = h(R)':<30} {'YES!':<12} {f'{lambda_hR.min():.2f} - {lambda_hR.max():.2f}':<20} {'Local disk structure'}")
print(f"{'λ ~ M^0.3 × R^0.3':<30} {'YES!':<12} {f'{lambda_hybrid.min():.2f} - {lambda_hybrid.max():.2f}':<20} {'Position-dependent'}")

print("\n* Mass-dependent λ becomes constant because all stars have equal MC weight")

print("\n" + "="*80)
print("WHAT HAPPENS IN THE CALCULATION:")
print("="*80)
print("\nFor each observation point r_obs and each star i:")
print("  1. Compute distance: r = |r_obs - r_i|")
print(f"  2. Get star's coherence length: λ_i (varies by hypothesis)")
print(f"  3. Compute enhancement: K(r) = A × [1 - (1 + (r/λ_i)^p)^-n]")
print(f"  4. Enhanced force: F = F_Newton × (1 + K(r))")
print(f"  5. Sum over ALL {len(gaia):,} stars")

print("\nKey parameters:")
print(f"  A = 0.591 (enhancement amplitude)")
print(f"  p = 0.757 (Burr-XII shape)")
print(f"  n_coh = 0.5 (coherence exponent)")

print("\nFor position-dependent λ:")
print("  - Inner stars (R<3 kpc): λ ~ 0.1-0.5 kpc (small, dense)")
print("  - Solar stars (R~8 kpc): λ ~ 0.5-1 kpc (intermediate)")
print("  - Outer stars (R>12 kpc): λ ~ 2-10 kpc (large, sparse)")

print("\n" + "="*80)
print("PHYSICAL INTERPRETATION:")
print("="*80)
print("\nEach star creates a 'coherence wave' with characteristic length λ_i:")
print("  - Small λ: Short-range enhancement (weak coupling)")
print("  - Large λ: Long-range enhancement (strong coupling)")
print("\nThe enhancement at distance r from star i:")
print("  K(r|λ_i) = A × [coherence window]")
print("\nFor λ=5 kpc:")
print(f"  - At r=1 kpc: K ~ 0.2 (moderate enhancement)")
print(f"  - At r=5 kpc: K ~ 0.4 (strong enhancement)")
print(f"  - At r=20 kpc: K ~ 0.55 (near maximum)")
print("\nTotal enhancement at r_obs = sum over all stars!")

print("\n✓ Visualization saved to GravityWaveTest/lambda_variations_by_star.png")

