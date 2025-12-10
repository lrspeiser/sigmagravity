"""
Load galaxy masses from SPARC master table.

CRITICAL: SPARC velocity components (v_disk, v_gas) are rotation curve
decompositions that don't reach far enough to capture total masses.
Must use master table (MasterSheet_SPARC.mrt) instead.
"""

import numpy as np
import os


def load_sparc_masses(galaxy_name, data_dir=None):
    """
    Load total baryon mass and scale lengths from SPARC master table.
    
    Parameters
    ----------
    galaxy_name : str
        Galaxy name (e.g., 'DDO154')
    data_dir : str, optional
        Path to data directory
        
    Returns
    -------
    masses : dict
        Dictionary with:
        - M_stellar: stellar mass (M☉) from L[3.6] × M/L
        - M_HI: HI gas mass (M☉) from master table
        - M_total: M_stellar + M_HI
        - R_disk: disk scale length (kpc)
        - R_HI: HI radius at 1 Msun/pc^2 (kpc)
        - L_36: luminosity at [3.6μm] (10^9 L☉)
        - Vflat: flat rotation velocity (km/s)
    """
    if data_dir is None:
        # Find data directory
        possible_paths = [
            '../../data',
            '../data',
            'data',
            os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                break
    
    master_file = os.path.join(data_dir, 'Rotmod_LTG', 'MasterSheet_SPARC.mrt')
    
    if not os.path.exists(master_file):
        raise FileNotFoundError(f"SPARC master table not found: {master_file}")
    
    # Read master table
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    # Find galaxy row (galaxy names in columns 1-11, right-padded with spaces)
    galaxy_data = None
    for line in lines:
        # Extract galaxy name (columns 1-11, trim whitespace)
        line_galaxy = line[:11].strip()
        if line_galaxy == galaxy_name:
            galaxy_data = line
            break
    
    if galaxy_data is None:
        raise ValueError(f"Galaxy {galaxy_name} not found in SPARC master table")
    
    # Parse by splitting on whitespace (format is space-separated, not strictly fixed-width)
    # Column order from header:
    # Galaxy, T, D, e_D, f_D, Inc, e_Inc, L[3.6], e_L[3.6], Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, e_Vflat, Q, Ref
    
    parts = galaxy_data.split()
    
    # Extract needed columns by position in split list
    try:
        L_36 = float(parts[7])    # L[3.6] in 10^9 L☉
        R_disk = float(parts[11])  # Rdisk in kpc
        M_HI = float(parts[13])   # MHI in 10^9 M☉
        R_HI = float(parts[14])   # RHI in kpc
        Vflat = float(parts[15])  # Vflat in km/s
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not parse SPARC data for {galaxy_name}: {e}\nData: {galaxy_data}")
    
    # Stellar mass from luminosity
    # Use M/L = 0.5 M☉/L☉ for [3.6μm] band (typical for SPARC)
    M_L = 0.5
    M_stellar = L_36 * M_L  # 10^9 M☉
    
    # Convert to solar masses
    M_stellar *= 1e9
    M_HI *= 1e9
    M_total = M_stellar + M_HI
    
    return {
        'M_stellar': M_stellar,
        'M_HI': M_HI,
        'M_total': M_total,
        'R_disk': R_disk,
        'R_HI': R_HI,
        'L_36': L_36,
        'Vflat': Vflat,
        'galaxy_name': galaxy_name
    }


if __name__ == '__main__':
    # Test with DDO154
    masses = load_sparc_masses('DDO154')
    print(f"\nSPARC Master Table Masses for {masses['galaxy_name']}:")
    print(f"  L[3.6]:     {masses['L_36']:.3f} × 10^9 L☉")
    print(f"  M_stellar:  {masses['M_stellar']:.2e} M☉")
    print(f"  M_HI:       {masses['M_HI']:.2e} M☉")
    print(f"  M_total:    {masses['M_total']:.2e} M☉")
    print(f"  R_disk:     {masses['R_disk']:.2f} kpc")
    print(f"  R_HI:       {masses['R_HI']:.2f} kpc")
    print(f"  V_flat:     {masses['Vflat']:.1f} km/s")
