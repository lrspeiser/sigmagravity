"""
Compute ACTUAL stellar masses from Gaia photometry.
Uses color-magnitude relation to estimate individual stellar masses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimate_stellar_mass(G_mag, bp_rp, parallax):
    """
    Estimate stellar mass from Gaia photometry.
    
    Uses empirical mass-luminosity-color relations.
    
    Parameters:
    -----------
    G_mag : array
        Apparent G magnitude
    bp_rp : array  
        BP-RP color (mag)
    parallax : array
        Parallax (mas)
    
    Returns:
    --------
    M_star : array
        Stellar mass (M_☉)
    """
    
    # Distance in parsecs
    distance_pc = 1000.0 / parallax
    
    # Absolute G magnitude
    M_G = G_mag - 5 * np.log10(distance_pc) + 5
    
    # Empirical mass-luminosity relation (Mann et al. 2015, simplified)
    # For main sequence stars:
    # log10(M/M_sun) ≈ a × M_G + b × (BP-RP) + c
    
    # Different relations for different color ranges
    # Blue stars (BP-RP < 0.5): Hot, massive
    # Red stars (BP-RP > 2.0): Cool, low-mass
    
    M_star = np.zeros_like(M_G)
    
    # Very blue (O, B stars): M > 2 M_sun
    mask_hot = bp_rp < 0.5
    M_star[mask_hot] = 10**(-0.15 * M_G[mask_hot] + 0.8)
    
    # Blue-white (A, F stars): 1-2 M_sun
    mask_warm = (bp_rp >= 0.5) & (bp_rp < 1.0)
    M_star[mask_warm] = 10**(-0.20 * M_G[mask_warm] + 0.6)
    
    # Solar-type (G stars): 0.8-1.2 M_sun
    mask_solar = (bp_rp >= 1.0) & (bp_rp < 1.5)
    M_star[mask_solar] = 10**(-0.25 * M_G[mask_solar] + 0.4)
    
    # Orange-red (K stars): 0.5-0.8 M_sun
    mask_cool = (bp_rp >= 1.5) & (bp_rp < 2.5)
    M_star[mask_cool] = 10**(-0.30 * M_G[mask_cool] + 0.2)
    
    # Very red (M dwarfs): < 0.5 M_sun
    mask_red = bp_rp >= 2.5
    M_star[mask_red] = 10**(-0.35 * M_G[mask_red] + 0.0)
    
    # Clip to reasonable range
    M_star = np.clip(M_star, 0.08, 100.0)  # 0.08 M_sun (brown dwarf limit) to 100 M_sun
    
    return M_star

def compute_actual_stellar_masses():
    """
    Compute actual stellar masses for all 1.8M Gaia stars.
    """
    
    print("="*80)
    print("COMPUTING ACTUAL STELLAR MASSES FROM GAIA PHOTOMETRY")
    print("="*80)
    
    # Load Gaia data
    print("\nLoading Gaia data...")
    gaia_raw = pd.read_csv('data/gaia/gaia_large_sample_raw.csv')
    print(f"Loaded {len(gaia_raw):,} stars")
    
    # Estimate masses
    print("\nEstimating stellar masses from color-magnitude relation...")
    
    M_star = estimate_stellar_mass(
        gaia_raw['phot_g_mean_mag'].values,
        gaia_raw['bp_rp'].fillna(1.0).values,  # Fill missing colors with solar-type
        gaia_raw['parallax'].values
    )
    
    # Add to processed data
    gaia_processed = pd.read_csv('data/gaia/gaia_processed.csv')
    gaia_processed['M_star_estimated'] = M_star
    
    # Statistics
    print(f"\nStellar mass statistics:")
    print(f"  Mean: {M_star.mean():.3f} M_☉")
    print(f"  Median: {np.median(M_star):.3f} M_☉")
    print(f"  Std: {M_star.std():.3f} M_☉")
    print(f"  Min: {M_star.min():.3f} M_☉")
    print(f"  Max: {M_star.max():.3f} M_☉")
    print(f"  Total stellar mass: {M_star.sum():.2e} M_☉")
    
    # Compare to MW stellar disk
    print(f"\nComparison to MW:")
    print(f"  MW stellar disk mass (literature): ~5×10^10 M_☉")
    print(f"  Our Gaia sample total: {M_star.sum():.2e} M_☉")
    print(f"  Fraction captured: {M_star.sum() / 5e10 * 100:.1f}%")
    
    # Distribution by mass
    print(f"\nMass distribution:")
    mass_bins = [0.08, 0.5, 0.8, 1.2, 2.0, 5.0, 100]
    for i in range(len(mass_bins)-1):
        mask = (M_star >= mass_bins[i]) & (M_star < mass_bins[i+1])
        count = mask.sum()
        total_mass = M_star[mask].sum()
        print(f"  {mass_bins[i]:.2f}-{mass_bins[i+1]:.1f} M_☉: {count:>8,} stars, total: {total_mass:.2e} M_☉")
    
    # Distribution by radius
    print(f"\nMass by galactic region:")
    R = gaia_processed['R_cyl'].values
    R_bins = [0, 3, 5, 10, 15, 25]
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        count = mask.sum()
        total_mass = M_star[mask].sum()
        mean_mass = M_star[mask].mean() if count > 0 else 0
        print(f"  {R_bins[i]}-{R_bins[i+1]} kpc: {count:>8,} stars, M_total={total_mass:.2e} M_☉, M_mean={mean_mass:.3f} M_☉")
    
    # Save
    gaia_processed.to_csv('data/gaia/gaia_processed.csv', index=False)
    print(f"\n✓ Updated gaia_processed.csv with stellar masses")
    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Stellar Masses from Gaia Photometry ({len(M_star):,} stars)',
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Mass distribution
    ax = axes[0, 0]
    ax.hist(M_star, bins=np.logspace(-1, 2, 50), alpha=0.7, edgecolor='black')
    ax.axvline(M_star.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {M_star.mean():.2f} M_☉')
    ax.axvline(np.median(M_star), color='b', linestyle='--', linewidth=2, label=f'Median: {np.median(M_star):.2f} M_☉')
    ax.set_xlabel('Stellar mass [M_☉]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Stellar Mass Distribution')
    
    # Plot 2: Mass vs color
    ax = axes[0, 1]
    subsample = np.random.choice(len(M_star), min(10000, len(M_star)), replace=False)
    color = gaia_raw['bp_rp'].fillna(1.0).values[subsample]
    ax.scatter(color, M_star[subsample], s=1, alpha=0.3)
    ax.set_xlabel('BP-RP color [mag]', fontsize=12)
    ax.set_ylabel('Estimated mass [M_☉]', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Mass vs Color')
    
    # Plot 3: Total mass by radius
    ax = axes[1, 0]
    R_bins_plot = np.linspace(0, 20, 40)
    R_centers = 0.5 * (R_bins_plot[:-1] + R_bins_plot[1:])
    mass_in_bins = []
    
    for i in range(len(R_bins_plot)-1):
        mask = (R >= R_bins_plot[i]) & (R < R_bins_plot[i+1])
        mass_in_bins.append(M_star[mask].sum())
    
    ax.plot(R_centers, mass_in_bins, 'b-', linewidth=2)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Total stellar mass in bin [M_☉]', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Stellar Mass Distribution by Radius')
    
    # Plot 4: Mean mass by radius
    ax = axes[1, 1]
    mean_mass_in_bins = []
    for i in range(len(R_bins_plot)-1):
        mask = (R >= R_bins_plot[i]) & (R < R_bins_plot[i+1])
        mean_mass_in_bins.append(M_star[mask].mean() if mask.sum() > 0 else np.nan)
    
    ax.plot(R_centers, mean_mass_in_bins, 'g-', linewidth=2)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Mean stellar mass [M_☉]', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('Mean Stellar Mass vs Radius')
    
    plt.tight_layout()
    plt.savefig('GravityWaveTest/stellar_masses_from_gaia.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to GravityWaveTest/stellar_masses_from_gaia.png")
    plt.close()
    
    return M_star

if __name__ == "__main__":
    M_star = compute_actual_stellar_masses()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Now we have ACTUAL stellar masses for all 1.8M stars!")
    print("2. But there's still a conceptual issue...")
    print("\n   The individual STELLAR masses (0.1-10 M_☉ each)")
    print("   are NOT the same as the GRAVITATING disk mass!")
    print("\n   Stars are TRACERS of a continuous mass distribution.")
    print("   Total stellar mass ≠ Total gravitating mass (includes gas, dark matter)")
    print("\nSee STELLAR_VS_GRAVITATING_MASS.md for full explanation")

