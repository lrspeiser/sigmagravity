"""
Prepare SPARC data with all required columns for scale tests.
"""

import pandas as pd
import numpy as np
import os

def prepare_sparc_combined():
    """
    Prepare SPARC data by combining existing files and adding required columns.
    
    Required columns:
    - galaxy_name
    - M_baryon (M_sun)
    - M_stellar (M_sun)
    - M_gas (M_sun)
    - v_flat (km/s)
    - R_disk (kpc)
    - sigma_velocity (km/s) - estimated if not available
    - bulge_frac (0-1) - estimated if not available
    - morphology_code
    """
    
    print("="*80)
    print("SPARC DATA PREPARATION")
    print("="*80)
    
    # Try to find existing SPARC data files
    possible_paths = [
        "many_path_model/data/sparc_masses.csv",
        "data/sparc/sparc_masses.csv",
        "many_path_model/sparc_masses.csv"
    ]
    
    sparc_file = None
    for path in possible_paths:
        if os.path.exists(path):
            sparc_file = path
            break
    
    if sparc_file is None:
        print("\n❌ ERROR: Could not find SPARC data file!")
        print("Looked in:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease ensure SPARC data is available in one of these locations.")
        return None
    
    print(f"\n✓ Found SPARC data: {sparc_file}")
    print(f"Loading...")
    
    df = pd.read_csv(sparc_file)
    print(f"Loaded {len(df)} galaxies")
    
    # Check what columns we have
    print(f"\nAvailable columns: {list(df.columns)}")
    
    # Create combined dataframe with required columns
    combined = pd.DataFrame()
    
    # Map existing columns (adjust based on your actual column names)
    column_mapping = {
        'galaxy': 'galaxy_name',
        'Galaxy': 'galaxy_name',
        'name': 'galaxy_name',
        'M_baryonic': 'M_baryon',
        'M_baryon': 'M_baryon',
        'M_b': 'M_baryon',
        'Mstar': 'M_stellar',
        'M_stellar': 'M_stellar',
        'M_star': 'M_stellar',
        'Mgas': 'M_gas',
        'M_gas': 'M_gas',
        'v_out': 'v_flat',
        'V_flat': 'v_flat',
        'v_flat': 'v_flat',
        'Rdisk': 'R_disk',
        'R_disk': 'R_disk',
        'R_eff': 'R_disk',
        'L3.6': 'L_3.6',
        'L_3.6': 'L_3.6'
    }
    
    # Map columns
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            combined[new_name] = df[old_name]
    
    # Ensure M_baryon exists (compute from stellar + gas if needed)
    if 'M_baryon' not in combined.columns:
        if 'M_stellar' in combined.columns and 'M_gas' in combined.columns:
            print("\nComputing M_baryon = M_stellar + M_gas")
            combined['M_baryon'] = combined['M_stellar'] + combined['M_gas']
        else:
            print("\n❌ ERROR: Cannot compute M_baryon - missing stellar/gas masses")
            return None
    
    # Add sigma_velocity (estimate if not available)
    if 'sigma_velocity' not in combined.columns:
        print("\nEstimating sigma_velocity ~ 0.15 * v_flat")
        combined['sigma_velocity'] = 0.15 * combined['v_flat']
    
    # Add bulge_frac (estimate if not available)
    if 'bulge_frac' not in combined.columns:
        print("\nEstimating bulge_frac from morphology")
        # Default to 0.1 for spirals, can refine based on morphology
        combined['bulge_frac'] = 0.1
    
    # Add morphology_code if missing
    if 'morphology_code' not in combined.columns:
        combined['morphology_code'] = 'Spiral'  # Default
    
    # Check for required columns
    required = ['galaxy_name', 'M_baryon', 'v_flat', 'R_disk']
    missing = [col for col in required if col not in combined.columns]
    
    if missing:
        print(f"\n❌ ERROR: Missing required columns: {missing}")
        return None
    
    # Remove rows with NaN in critical columns
    n_before = len(combined)
    combined = combined.dropna(subset=['M_baryon', 'v_flat', 'R_disk'])
    n_after = len(combined)
    
    if n_before > n_after:
        print(f"\nRemoved {n_before - n_after} galaxies with missing data")
    
    # Save combined file
    output_dir = "data/sparc"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/sparc_combined.csv"
    
    combined.to_csv(output_path, index=False)
    
    print(f"\n✓ SUCCESS!")
    print(f"Saved {len(combined)} galaxies to {output_path}")
    print(f"\nFinal columns: {list(combined.columns)}")
    print(f"\nData summary:")
    print(f"  M_baryon: {combined['M_baryon'].min():.2e} - {combined['M_baryon'].max():.2e} M_sun")
    print(f"  v_flat: {combined['v_flat'].min():.1f} - {combined['v_flat'].max():.1f} km/s")
    print(f"  R_disk: {combined['R_disk'].min():.2f} - {combined['R_disk'].max():.2f} kpc")
    
    return combined

if __name__ == "__main__":
    prepare_sparc_combined()

