"""
Prepare REAL Gaia data for star-by-star calculation.
Uses actual Gaia DR3 observations instead of synthetic data.
"""

import numpy as np
import pandas as pd
import os

def prepare_real_gaia_for_star_test():
    """
    Convert real Gaia data to format needed for star-by-star test.
    
    Input: data/gaia/mw/gaia_mw_real.csv (143,995 real stars)
    Output: data/gaia/gaia_processed.csv (ready for test)
    """
    
    print("="*80)
    print("PREPARING REAL GAIA DATA FOR STAR-BY-STAR TEST")
    print("="*80)
    
    # Load real Gaia data
    gaia_file = "data/gaia/mw/gaia_mw_real.csv"
    
    if not os.path.exists(gaia_file):
        print(f"\nERROR: {gaia_file} not found!")
        print("Falling back to synthetic data...")
        return None
    
    print(f"\nLoading real Gaia data from {gaia_file}...")
    gaia = pd.read_csv(gaia_file)
    
    print(f"Loaded {len(gaia):,} REAL Gaia stars")
    print(f"  R range: {gaia['R_kpc'].min():.2f} - {gaia['R_kpc'].max():.2f} kpc")
    print(f"  z range: {gaia['z_kpc'].min():.2f} - {gaia['z_kpc'].max():.2f} kpc")
    
    # Convert to format needed by test_star_by_star_mw.py
    # Need: R_cyl, z, phi, M_star (placeholder), v_rad, pmra, pmdec, distance_pc
    
    # Sample phi uniformly (not in Gaia data - stars are mass tracers)
    # For force calculation, azimuthal distribution doesn't matter (axisymmetric disk)
    phi = np.random.uniform(0, 2*np.pi, len(gaia))
    
    # Create dataframe in expected format
    stars = pd.DataFrame({
        'source_id': gaia['source_id'].values.astype(int),
        'R_cyl': gaia['R_kpc'].values,
        'z': gaia['z_kpc'].values,
        'phi': phi,
        'M_star': np.ones(len(gaia)),  # Placeholder - will use M_disk/N_stars
        'v_rad': gaia['vR'].values,  # Radial velocity
        'pmra': np.zeros(len(gaia)),  # Not used in test
        'pmdec': np.zeros(len(gaia)),  # Not used in test
        'distance_pc': gaia['R_kpc'].values * 1000.0  # Approximate
    })
    
    # Also save observed rotation velocities for comparison
    stars['v_obs'] = gaia['vphi'].values
    stars['v_obs_err'] = gaia['vphi_err'].values if 'vphi_err' in gaia.columns else 5.0
    
    # Save
    output_dir = "data/gaia"
    output_path = f"{output_dir}/gaia_processed.csv"
    
    stars.to_csv(output_path, index=False)
    
    print(f"\nSUCCESS!")
    print(f"Converted {len(stars):,} real Gaia stars to test format")
    print(f"Saved to {output_path}")
    
    # Generate diagnostic plot
    print("\nGenerating diagnostic plot...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Real Gaia DR3 Data ({len(stars):,} stars)', fontsize=14, fontweight='bold')
    
    # Plot 1: R-z distribution
    ax = axes[0, 0]
    subsample = np.random.choice(len(stars), size=min(10000, len(stars)), replace=False)
    scatter = ax.scatter(stars['R_cyl'].iloc[subsample], 
                        stars['z'].iloc[subsample], 
                        c=stars['v_obs'].iloc[subsample],
                        s=1, alpha=0.5, cmap='viridis', vmin=150, vmax=300)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('z [kpc]', fontsize=12)
    ax.set_title('Spatial Distribution (colored by v_obs)')
    plt.colorbar(scatter, ax=ax, label='v_obs [km/s]')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Radial distribution
    ax = axes[0, 1]
    ax.hist(stars['R_cyl'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Radial Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Vertical distribution
    ax = axes[1, 0]
    for R_bin in [4, 6, 8, 10, 12]:
        mask = (stars['R_cyl'] > R_bin - 0.5) & (stars['R_cyl'] < R_bin + 0.5)
        if mask.sum() > 100:
            z_hist, z_edges = np.histogram(stars.loc[mask, 'z'], bins=30, range=(-1, 1))
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            z_hist = z_hist / z_hist.max()
            ax.plot(z_centers, z_hist, label=f'R = {R_bin} kpc', linewidth=2)
    
    ax.set_xlabel('z [kpc]', fontsize=12)
    ax.set_ylabel('Normalized density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Vertical Profiles')
    
    # Plot 4: Observed rotation curve
    ax = axes[1, 1]
    
    # Bin by radius
    R_bins = np.linspace(3, 17, 30)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    v_median = []
    v_p16 = []
    v_p84 = []
    
    for i in range(len(R_bins)-1):
        mask = (stars['R_cyl'] >= R_bins[i]) & (stars['R_cyl'] < R_bins[i+1])
        if mask.sum() > 10:
            v_median.append(np.median(stars.loc[mask, 'v_obs']))
            v_p16.append(np.percentile(stars.loc[mask, 'v_obs'], 16))
            v_p84.append(np.percentile(stars.loc[mask, 'v_obs'], 84))
        else:
            v_median.append(np.nan)
            v_p16.append(np.nan)
            v_p84.append(np.nan)
    
    ax.plot(R_centers, v_median, 'b-', linewidth=2, label='Median')
    ax.fill_between(R_centers, v_p16, v_p84, alpha=0.3, color='blue', label='16-84 percentile')
    ax.axhline(220, color='r', linestyle='--', linewidth=2, label='Expected (220 km/s)')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_phi [km/s]', fontsize=12)
    ax.set_title('Observed Rotation Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_gaia_properties.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_dir}/real_gaia_properties.png")
    plt.close()
    
    # Summary statistics
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"\nSpatial coverage:")
    print(f"  R: {stars['R_cyl'].min():.2f} - {stars['R_cyl'].max():.2f} kpc")
    print(f"  z: {stars['z'].min():.2f} - {stars['z'].max():.2f} kpc")
    print(f"\nObserved velocities:")
    print(f"  v_phi: {stars['v_obs'].median():.1f} ± {stars['v_obs'].std():.1f} km/s")
    print(f"  Range: {stars['v_obs'].min():.1f} - {stars['v_obs'].max():.1f} km/s")
    print(f"\nReady to run: python GravityWaveTest/test_star_by_star_mw.py")
    
    return stars

if __name__ == "__main__":
    stars = prepare_real_gaia_for_star_test()
    
    if stars is not None:
        print("\n" + "="*80)
        print("READY TO TEST WITH REAL DATA!")
        print("="*80)
        print("\nRun: python GravityWaveTest/test_star_by_star_mw.py")
        print("\nThis will use REAL Gaia DR3 observations (143,995 stars)")
        print("instead of synthetic data!")

