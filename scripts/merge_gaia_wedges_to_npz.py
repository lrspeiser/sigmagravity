#!/usr/bin/env python3
"""
Merge multiple Gaia CSV wedges into a single consolidated NPZ file.
Reads existing mw_gaia_144k.npz schema and appends new wedge data.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def load_existing_npz(npz_path):
    """Load existing NPZ and return as dict of arrays."""
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_wedge_csv(csv_path):
    """Load a Gaia wedge CSV and return normalized DataFrame."""
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    required = ['source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 
                'radial_velocity', 'phot_g_mean_mag']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return df


def merge_to_npz(base_npz, wedge_csvs, output_npz):
    """
    Merge base NPZ with new wedge CSVs and write consolidated NPZ.
    
    Args:
        base_npz: path to existing mw_gaia_144k.npz
        wedge_csvs: list of paths to new CSV wedges
        output_npz: path to write merged NPZ
    """
    # Load base
    base = load_existing_npz(base_npz)
    print(f"Loaded base: {len(base['source_id'])} stars")
    
    # Collect new wedge data
    new_dfs = []
    for csv in wedge_csvs:
        df = load_wedge_csv(csv)
        print(f"Loaded {csv}: {len(df)} stars")
        new_dfs.append(df)
    
    if not new_dfs:
        print("No new wedges to merge; copying base to output")
        np.savez_compressed(output_npz, **base)
        return
    
    merged_df = pd.concat(new_dfs, ignore_index=True)
    print(f"Total new stars: {len(merged_df)}")
    
    # Drop duplicates by source_id (prefer new data)
    base_df = pd.DataFrame({
        'source_id': base['source_id'],
        'ra': base.get('ra', np.full(len(base['source_id']), np.nan)),
        'dec': base.get('dec', np.full(len(base['source_id']), np.nan)),
        'parallax': base.get('parallax', np.full(len(base['source_id']), np.nan)),
        'pmra': base.get('pmra', np.full(len(base['source_id']), np.nan)),
        'pmdec': base.get('pmdec', np.full(len(base['source_id']), np.nan)),
        'radial_velocity': base.get('radial_velocity', np.full(len(base['source_id']), np.nan)),
        'phot_g_mean_mag': base.get('phot_g_mean_mag', np.full(len(base['source_id']), np.nan)),
    })
    
    combined = pd.concat([base_df, merged_df], ignore_index=True)
    combined.drop_duplicates(subset='source_id', keep='last', inplace=True)
    print(f"After dedup: {len(combined)} stars")
    
    # Write NPZ
    out_dict = {
        'source_id': combined['source_id'].values,
        'ra': combined['ra'].values,
        'dec': combined['dec'].values,
        'parallax': combined['parallax'].values,
        'pmra': combined['pmra'].values,
        'pmdec': combined['pmdec'].values,
        'radial_velocity': combined['radial_velocity'].values,
        'phot_g_mean_mag': combined['phot_g_mean_mag'].values,
    }
    
    np.savez_compressed(output_npz, **out_dict)
    print(f"Wrote {output_npz}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Base NPZ (e.g., mw_gaia_144k.npz)')
    parser.add_argument('--wedges', nargs='+', required=True, help='New wedge CSV files')
    parser.add_argument('--output', required=True, help='Output merged NPZ')
    args = parser.parse_args()
    
    merge_to_npz(args.base, args.wedges, args.output)


if __name__ == '__main__':
    main()
