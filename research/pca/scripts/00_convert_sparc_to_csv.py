#!/usr/bin/env python3
"""
Convert SPARC .dat files from data/Rotmod_LTG/ to CSV format
and parse the MasterSheet metadata into a CSV.
"""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

def parse_mastersheet(path):
    """Parse the fixed-width SPARC MasterSheet format"""
    # Read as whitespace-separated, skip header
    df = pd.read_csv(path, delim_whitespace=True, skiprows=99, header=None,
                     names=['Galaxy', 'T', 'D', 'e_D', 'f_D', 'Inc', 'e_Inc', 
                            'L36', 'e_L36', 'Reff', 'SBeff', 'Rdisk', 'SBdisk',
                            'MHI', 'RHI', 'Vflat', 'e_Vflat', 'Q', 'Ref'])
    
    # Clean up the data
    df = df[df['Galaxy'].notna()]  # Remove empty rows
    
    # Convert numeric columns
    numeric_cols = ['T', 'D', 'e_D', 'f_D', 'Inc', 'e_Inc', 'L36', 'e_L36',
                    'Reff', 'SBeff', 'Rdisk', 'SBdisk', 'MHI', 'RHI', 'Vflat', 'e_Vflat', 'Q']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute Mbar (baryonic mass) from luminosity and HI mass
    # Mbar ≈ M_* + M_HI, using mass-to-light ratio ~0.5 for [3.6]
    # L36 is in 10^9 solar luminosities, MHI is in 10^9 solar masses
    df['Mstar'] = df['L36'] * 0.5
    df['Mbar'] = df['Mstar'] + df['MHI']
    
    # Compute Sigma0 from SBdisk (convert from L_sun/pc^2 to M_sun/pc^2)
    df['Sigma0'] = df['SBdisk'] * 0.5
    
    # Add HSB/LSB classification
    df['HSB_LSB'] = df['Sigma0'].apply(
        lambda x: 'HSB' if x > 100 else ('LSB' if pd.notna(x) else 'Unknown')
    )
    
    # Rename columns to match expected format
    result = pd.DataFrame({
        'name': df['Galaxy'].str.strip(),
        'T': df['T'],
        'D': df['D'],
        'e_D': df['e_D'],
        'f_D': df['f_D'],
        'Inc': df['Inc'],
        'e_Inc': df['e_Inc'],
        'L36': df['L36'],
        'Reff': df['Reff'],
        'SBeff': df['SBeff'],
        'Rd': df['Rdisk'],
        'SBdisk': df['SBdisk'],
        'MHI': df['MHI'],
        'RHI': df['RHI'],
        'Vf': df['Vflat'],
        'e_Vf': df['e_Vflat'],
        'Q': df['Q'],
        'Mbar': df['Mbar'],
        'Sigma0': df['Sigma0'],
        'HSB_LSB': df['HSB_LSB']
    })
    
    return result

def convert_dat_to_csv(dat_file, csv_file):
    """Convert a single SPARC .dat file to CSV format"""
    # Read the .dat file
    with open(dat_file, 'r') as f:
        lines = f.readlines()
    
    # First line has comment with distance
    distance = None
    if lines[0].startswith('# Distance'):
        match = re.search(r'([\d.]+)\s*Mpc', lines[0])
        if match:
            distance = float(match.group(1))
    
    # Header on line 2 (columns), line 3 (units)
    # Data starts at line 4
    data_lines = []
    for line in lines[3:]:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 3:
                data_lines.append(parts)
    
    if not data_lines:
        print(f"Warning: No data found in {dat_file}")
        return False
    
    # Convert to DataFrame
    # Columns: Rad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
    df_data = []
    for parts in data_lines:
        try:
            rad = float(parts[0])
            vobs = float(parts[1])
            errv = float(parts[2])
            vgas = float(parts[3]) if len(parts) > 3 else 0.0
            vdisk = float(parts[4]) if len(parts) > 4 else 0.0
            vbul = float(parts[5]) if len(parts) > 5 else 0.0
            
            # Compute V_bar = sqrt(V_disk^2 + V_gas^2 + V_bul^2)
            vbar = np.sqrt(vdisk**2 + vgas**2 + vbul**2)
            
            df_data.append({
                'R_kpc': rad,
                'V_obs': vobs,
                'eV_obs': errv,
                'V_gas': vgas,
                'V_disk': vdisk,
                'V_bul': vbul,
                'V_bar': vbar
            })
        except (ValueError, IndexError) as e:
            continue
    
    if not df_data:
        return False
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False)
    return True

def main():
    # Set up paths
    repo_root = Path(__file__).parent.parent.parent
    rotmod_dir = repo_root / 'data' / 'Rotmod_LTG'
    mastersheet_path = rotmod_dir / 'MasterSheet_SPARC.mrt'
    
    # Create output directories
    out_curves_dir = repo_root / 'pca' / 'data' / 'raw' / 'sparc_curves'
    out_meta_dir = repo_root / 'pca' / 'data' / 'raw' / 'metadata'
    out_curves_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    
    print("Converting SPARC .dat files to CSV...")
    
    # Convert all .dat files to CSV
    dat_files = list(rotmod_dir.glob('*_rotmod.dat'))
    print(f"Found {len(dat_files)} rotation curve files")
    
    converted = 0
    for dat_file in dat_files:
        # Extract galaxy name (remove _rotmod.dat)
        galaxy_name = dat_file.stem.replace('_rotmod', '')
        csv_file = out_curves_dir / f"{galaxy_name}.csv"
        
        if convert_dat_to_csv(dat_file, csv_file):
            converted += 1
    
    print(f"Converted {converted}/{len(dat_files)} files successfully")
    
    # Parse mastersheet
    print("\nParsing SPARC MasterSheet...")
    if mastersheet_path.exists():
        meta_df = parse_mastersheet(mastersheet_path)
        out_meta_file = out_meta_dir / 'sparc_meta.csv'
        meta_df.to_csv(out_meta_file, index=False)
        print(f"Saved metadata for {len(meta_df)} galaxies to {out_meta_file}")
        print(f"\nMetadata columns: {list(meta_df.columns)}")
        print(f"Sample stats:")
        print(f"  Rd range: {meta_df['Rd'].min():.2f} - {meta_df['Rd'].max():.2f} kpc")
        print(f"  Vf range: {meta_df['Vf'].min():.2f} - {meta_df['Vf'].max():.2f} km/s")
        print(f"  Mbar range: {meta_df['Mbar'].min():.2f} - {meta_df['Mbar'].max():.2f} x10^9 M_sun")
        print(f"  HSB galaxies: {(meta_df['HSB_LSB'] == 'HSB').sum()}")
        print(f"  LSB galaxies: {(meta_df['HSB_LSB'] == 'LSB').sum()}")
    else:
        print(f"Warning: MasterSheet not found at {mastersheet_path}")
    
    print("\nConversion complete!")
    print(f"  Curves: {out_curves_dir}")
    print(f"  Metadata: {out_meta_dir}")

if __name__ == '__main__':
    main()

