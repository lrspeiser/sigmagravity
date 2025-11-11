"""
Prepare SPARC data from Rotmod_LTG files with all required columns.
"""

import pandas as pd
import numpy as np
import os
import glob

def load_rotmod_galaxy(filepath):
    """Load a single Rotmod galaxy file."""
    try:
        df = pd.read_csv(filepath, delim_whitespace=True, comment='#', 
                        names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
        return df
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None

def compute_galaxy_properties(df_rotcurve, name):
    """Compute galaxy-level properties from rotation curve."""
    # Get outer velocity (flat part)
    v_flat = np.median(df_rotcurve['Vobs'].iloc[-10:])  # Last 10 points
    
    # Get outer radius
    R_disk = df_rotcurve['Rad'].iloc[-1]  # Last radial point
    
    # Compute baryonic mass from velocity components
    # M_baryon ≈ integral of (V_gas² + V_disk² + V_bul²) × r / G
    G = 4.302e-6  # kpc (km/s)² M_sun^-1
    
    rad = df_rotcurve['Rad'].values
    V_bar = np.sqrt(df_rotcurve['Vgas']**2 + df_rotcurve['Vdisk']**2 + df_rotcurve['Vbul']**2)
    
    # Approximate M(<R) from V²R/G at outer radius
    M_baryon = v_flat**2 * R_disk / G
    
    # Estimate stellar vs gas from velocity ratios
    V_disk = df_rotcurve['Vdisk'].iloc[-1]
    V_gas = df_rotcurve['Vgas'].iloc[-1]
    V_bul = df_rotcurve['Vbul'].iloc[-1]
    
    f_disk = (V_disk**2) / (V_disk**2 + V_gas**2 + V_bul**2 + 1e-10)
    f_gas = (V_gas**2) / (V_disk**2 + V_gas**2 + V_bul**2 + 1e-10)
    f_bul = (V_bul**2) / (V_disk**2 + V_gas**2 + V_bul**2 + 1e-10)
    
    M_stellar = M_baryon * (f_disk + f_bul)
    M_gas = M_baryon * f_gas
    
    return {
        'galaxy_name': name,
        'M_baryon': M_baryon,
        'M_stellar': M_stellar,
        'M_gas': M_gas,
        'v_flat': v_flat,
        'R_disk': R_disk,
        'sigma_velocity': 0.15 * v_flat,  # Typical estimate
        'bulge_frac': f_bul,
        'morphology_code': 'Spiral',
        'n_points': len(df_rotcurve)
    }

def prepare_sparc_from_rotmod():
    """Prepare SPARC data from Rotmod_LTG directory."""
    
    print("="*80)
    print("SPARC DATA PREPARATION FROM ROTMOD FILES")
    print("="*80)
    
    # Find Rotmod files
    rotmod_dir = "many_path_model/paper_release/data/Rotmod_LTG"
    if not os.path.exists(rotmod_dir):
        rotmod_dir = "data/Rotmod_LTG"
    
    if not os.path.exists(rotmod_dir):
        print(f"\nERROR: Could not find Rotmod directory")
        return None
    
    rotmod_files = glob.glob(f"{rotmod_dir}/*_rotmod.dat")
    print(f"\nFound {len(rotmod_files)} Rotmod files")
    
    # Process each galaxy
    galaxies = []
    for filepath in rotmod_files:
        name = os.path.basename(filepath).replace('_rotmod.dat', '')
        df_rot = load_rotmod_galaxy(filepath)
        
        if df_rot is not None and len(df_rot) > 5:
            props = compute_galaxy_properties(df_rot, name)
            galaxies.append(props)
    
    # Create combined dataframe
    combined = pd.DataFrame(galaxies)
    
    # Remove invalid entries
    n_before = len(combined)
    combined = combined[(combined['M_baryon'] > 0) & 
                       (combined['v_flat'] > 0) & 
                       (combined['R_disk'] > 0)]
    n_after = len(combined)
    
    if n_before > n_after:
        print(f"\nRemoved {n_before - n_after} galaxies with invalid data")
    
    # Save
    output_dir = "data/sparc"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/sparc_combined.csv"
    
    combined.to_csv(output_path, index=False)
    
    print(f"\nSUCCESS!")
    print(f"Saved {len(combined)} galaxies to {output_path}")
    print(f"\nData summary:")
    print(f"  M_baryon: {combined['M_baryon'].min():.2e} - {combined['M_baryon'].max():.2e} M_sun")
    print(f"  v_flat: {combined['v_flat'].min():.1f} - {combined['v_flat'].max():.1f} km/s")
    print(f"  R_disk: {combined['R_disk'].min():.2f} - {combined['R_disk'].max():.2f} kpc")
    
    return combined

if __name__ == "__main__":
    prepare_sparc_from_rotmod()

