#!/usr/bin/env python3
"""
Merge multiple Gaia NPZ wedges into a single consolidated NPZ file.
Combines base NPZ with additional wedge NPZs (all with derived galactocentric columns).
"""
import argparse
import numpy as np
from pathlib import Path


def load_existing_npz(npz_path):
    """Load existing NPZ and return as dict of arrays."""
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def merge_to_npz(base_npz, wedge_npzs, output_npz):
    """
    Merge base NPZ with new wedge NPZs and write consolidated NPZ.
    
    Args:
        base_npz: path to existing mw_gaia_144k.npz
        wedge_npzs: list of paths to new wedge NPZ files (from convert_gaia_csv_to_npz.py)
        output_npz: path to write merged NPZ
    """
    # Load base
    base = load_existing_npz(base_npz)
    required_keys = ['R_kpc', 'z_kpc', 'v_obs_kms', 'v_err_kms', 'gN_kms2_per_kpc', 'Sigma_loc_Msun_pc2']
    
    n_base = len(base['R_kpc'])
    print(f"Loaded base: {n_base:,} stars")
    print(f"  R range: {base['R_kpc'].min():.2f}–{base['R_kpc'].max():.2f} kpc")
    
    # Collect wedge arrays
    wedge_arrays = {k: [] for k in required_keys}
    total_wedge_stars = 0
    
    for wedge_path in wedge_npzs:
        wedge = load_existing_npz(wedge_path)
        n_wedge = len(wedge['R_kpc'])
        total_wedge_stars += n_wedge
        print(f"Loaded wedge {Path(wedge_path).name}: {n_wedge:,} stars")
        print(f"  R range: {wedge['R_kpc'].min():.2f}–{wedge['R_kpc'].max():.2f} kpc")
        
        for k in required_keys:
            wedge_arrays[k].append(wedge[k])
    
    if not wedge_npzs:
        print("No wedges to merge; copying base to output")
        np.savez_compressed(output_npz, **base)
        return
    
    # Concatenate all arrays
    print(f"\nMerging {n_base:,} base + {total_wedge_stars:,} wedge stars...")
    merged = {}
    for k in required_keys:
        merged[k] = np.concatenate([base[k]] + wedge_arrays[k])
    
    n_merged = len(merged['R_kpc'])
    print(f"Total after merge: {n_merged:,} stars")
    print(f"  R range: {merged['R_kpc'].min():.2f}–{merged['R_kpc'].max():.2f} kpc")
    print(f"  Mean v_obs: {merged['v_obs_kms'].mean():.1f} ± {merged['v_obs_kms'].std():.1f} km/s")
    
    # Save merged NPZ
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)
    print(f"\n✓ Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Base NPZ (e.g., mw_gaia_144k.npz)')
    parser.add_argument('--wedges', nargs='+', required=True, help='New wedge NPZ files')
    parser.add_argument('--output', required=True, help='Output merged NPZ')
    args = parser.parse_args()
    
    merge_to_npz(args.base, args.wedges, args.output)


if __name__ == '__main__':
    main()
