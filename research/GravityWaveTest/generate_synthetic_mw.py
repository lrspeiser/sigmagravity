"""
Generate synthetic Milky Way stellar distribution for testing
star-by-star calculation. CORRECTED for proper disk sampling.
"""

import numpy as np
import pandas as pd
import os

def sample_R_exponential_disk(n_stars, R_d, R_max):
    """
    Sample R from p(R) ∝ R × exp(-R/R_d) using rejection sampling.
    
    This is the CORRECT 2D disk profile (includes Jacobian factor R).
    """
    R_samples = []
    
    # Proposal: exponential (easy to sample)
    # Accept with probability R / (R + R_d) to get R × exp(-R/R_d)
    
    while len(R_samples) < n_stars:
        # Generate proposal samples (1.5x for efficiency)
        n_prop = int(1.5 * (n_stars - len(R_samples)))
        
        u = np.random.random(n_prop)
        R_prop = -R_d * np.log(1 - u * (1 - np.exp(-R_max / R_d)))  # Truncated exponential
        
        # Accept/reject with prob R / (R + R_d)
        accept_prob = R_prop / (R_prop + R_d)
        u_accept = np.random.random(n_prop)
        
        accepted = R_prop[u_accept < accept_prob]
        R_samples.extend(accepted)
    
    return np.array(R_samples[:n_stars])

def generate_synthetic_mw(n_stars: int = 100000,
                         R_max: float = 20.0,
                         z_max: float = 5.0,
                         seed: int = 42):
    """
    Generate synthetic MW-like stellar distribution.
    
    CORRECTED:
    - Proper 2D disk sampling: p(R) ∝ R × exp(-R/R_d)
    - Stars are SAMPLES of density field, not literal masses
    - Vertical sampling via sech² is correct
    """
    
    print("="*80)
    print("GENERATING SYNTHETIC MILKY WAY (CORRECTED)")
    print("="*80)
    print(f"\nParameters:")
    print(f"  N_stars: {n_stars:,}")
    print(f"  R_max: {R_max} kpc")
    print(f"  z_max: {z_max} kpc")
    
    np.random.seed(seed)
    
    # Disk parameters
    R_d = 2.5    # kpc (disk scale length)
    h_0 = 0.3    # kpc (scale height at R=0)
    R_h = 10.0   # kpc (scale height scale)
    
    print(f"\nDisk structure:")
    print(f"  Scale length: {R_d} kpc")
    print(f"  Scale height: {h_0} kpc -> {h_0 * (1 + R_max/R_h):.2f} kpc at R={R_max} kpc")
    
    # Generate radial positions - CORRECTED!
    print(f"\nSampling radial positions from p(R) ∝ R × exp(-R/{R_d})...")
    R_cyl = sample_R_exponential_disk(n_stars, R_d, R_max)
    
    # Generate phi uniformly
    phi = np.random.uniform(0, 2*np.pi, n_stars)
    
    # Generate z from sech² profile (this was already correct)
    h_R = h_0 * (1 + R_cyl / R_h)
    u_z = np.random.random(n_stars)
    z = h_R * np.arctanh(2 * u_z - 1)
    z = np.clip(z, -z_max, z_max)
    
    # IMPORTANT: Stars are SAMPLES, not real stellar masses
    # The gravitational mass will be assigned as M_disk / N_stars
    # But we can still track a "stellar mass" for visualization
    M_star = np.ones(n_stars)  # Placeholder - actual mass weights assigned in calculator
    
    # Velocity data (for completeness)
    v_rad = np.random.normal(0, 10, n_stars)  # km/s
    v_circ = 220.0 * np.ones(n_stars)  # km/s (flat rotation curve)
    sigma_v = 20.0 * (1 + np.abs(z) / h_R)
    v_circ += np.random.normal(0, sigma_v)
    
    distance_pc = R_cyl * 1000.0
    pmra = v_circ / (4.74 * distance_pc) + np.random.normal(0, 1, n_stars)
    pmdec = v_rad / (4.74 * distance_pc) + np.random.normal(0, 1, n_stars)
    
    # Create dataframe
    stars = pd.DataFrame({
        'source_id': np.arange(n_stars) + 1000000000,
        'R_cyl': R_cyl,
        'z': z,
        'phi': phi,
        'M_star': M_star,  # Placeholder
        'v_rad': v_rad,
        'pmra': pmra,
        'pmdec': pmdec,
        'distance_pc': distance_pc
    })
    
    print(f"\nGenerated stellar population:")
    print(f"  Total stars: {len(stars):,}")
    print(f"  R range: {stars['R_cyl'].min():.2f} - {stars['R_cyl'].max():.2f} kpc")
    print(f"  z range: {stars['z'].min():.2f} - {stars['z'].max():.2f} kpc")
    print(f"\n  NOTE: Stars are Monte Carlo samples of the disk density.")
    print(f"        Gravitational masses will be M_disk / N_stars each.")
    
    # Save
    output_dir = "data/gaia"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/gaia_processed.csv"
    
    stars.to_csv(output_path, index=False)
    
    print(f"\nSaved synthetic MW to {output_path}")
    print(f"\nYou can now run: python GravityWaveTest/test_star_by_star_mw.py")
    
    # Generate diagnostic plot
    print("\nGenerating diagnostic plot...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Synthetic Milky Way Properties (CORRECTED)', fontsize=14, fontweight='bold')
    
    # Plot 1: Surface density profile
    ax = axes[0, 0]
    R_bins = np.linspace(0, R_max, 50)
    counts, _ = np.histogram(R_cyl, bins=R_bins)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    area = np.pi * (R_bins[1:]**2 - R_bins[:-1]**2)
    surface_density = counts / area
    
    ax.plot(R_centers, surface_density / surface_density[0], 'b-', linewidth=2, label='Generated')
    R_theory = np.linspace(0.1, R_max, 100)
    ax.plot(R_theory, np.exp(-R_theory / R_d), 'r--', linewidth=2, label=f'Theory: exp(-R/{R_d})')
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Σ / Σ₀', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Surface Density Profile (should match theory!)')
    
    # Plot 2: Radial histogram
    ax = axes[0, 1]
    ax.hist(R_cyl, bins=50, alpha=0.7, edgecolor='black', density=True)
    
    # Theory curve: p(R) ∝ R × exp(-R/R_d)
    R_th = np.linspace(0, R_max, 200)
    # Normalize: integral of R × exp(-R/R_d) from 0 to inf is R_d²
    p_th = R_th * np.exp(-R_th / R_d)
    # Truncate at R_max
    norm = R_d**2 * (1 - np.exp(-R_max/R_d) * (1 + R_max/R_d))
    p_th = p_th / norm
    
    ax.plot(R_th, p_th, 'r-', linewidth=2, label='Theory: R×exp(-R/R_d)')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Radial Distribution')
    
    # Plot 3: Vertical profile
    ax = axes[1, 0]
    for R_bin in [2, 5, 8, 12]:
        mask = (R_cyl > R_bin - 0.5) & (R_cyl < R_bin + 0.5)
        if mask.sum() > 100:
            z_hist, z_edges = np.histogram(z[mask], bins=30, range=(-3, 3))
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            z_hist = z_hist / z_hist.max()
            ax.plot(z_centers, z_hist, label=f'R = {R_bin} kpc', linewidth=2)
    
    ax.set_xlabel('z [kpc]', fontsize=12)
    ax.set_ylabel('Normalized density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Vertical Profiles (sech²)')
    
    # Plot 4: R-z distribution
    ax = axes[1, 1]
    subsample = np.random.choice(len(stars), size=min(10000, len(stars)), replace=False)
    ax.scatter(R_cyl[subsample], z[subsample], s=1, alpha=0.3, c='blue')
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('z [kpc]', fontsize=12)
    ax.set_xlim(0, R_max)
    ax.set_ylim(-z_max, z_max)
    ax.grid(True, alpha=0.3)
    ax.set_title('Spatial Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/synthetic_mw_properties.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_dir}/synthetic_mw_properties.png")
    plt.close()
    
    return stars

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CHOOSE SAMPLE SIZE")
    print("="*80)
    print("\nOptions:")
    print("  1. Quick test (10k stars) - ~1 second per hypothesis")
    print("  2. Medium test (100k stars) - ~5 seconds per hypothesis") 
    print("  3. Realistic test (1M stars) - ~30 seconds per hypothesis")
    
    choice = input("\nEnter choice (1-3) or press Enter for option 2: ").strip()
    
    if choice == '1':
        n_stars = 10000
    elif choice == '3':
        n_stars = 1000000
    else:
        n_stars = 100000
    
    stars = generate_synthetic_mw(n_stars=n_stars)
    
    print("\n" + "="*80)
    print("READY TO TEST!")
    print("="*80)
    print("\nRun: python GravityWaveTest/test_star_by_star_mw.py")

