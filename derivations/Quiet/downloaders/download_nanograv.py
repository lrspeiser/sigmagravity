"""
Download NANOGrav pulsar timing data for gravitational wave background.

Source: NANOGrav (https://nanograv.org/)
Key dataset: NANOGrav 15-year data release

Pulsar timing arrays measure spacetime fluctuations through:
- Timing residuals (pulse arrival time variations)
- The stochastic gravitational wave background h_c(f)
- Individual pulsar noise properties

This provides a direct measure of "metric fluctuation amplitude" -
one of the candidate variables for gravitational quietness.
"""

import sys
from pathlib import Path
import numpy as np
import requests
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PULSAR_DIR

# =============================================================================
# DATA SOURCES
# =============================================================================

NANOGRAV_DATA = {
    # NANOGrav 15-year dataset
    "15yr": {
        "description": "NANOGrav 15-year gravitational wave background",
        "paper": "https://arxiv.org/abs/2306.16213",
        "data_url": "https://data.nanograv.org/15yr/",
        "zenodo": "https://zenodo.org/record/8067508",
        "files": [
            "NANOGrav_15yr_gwb_posteriors.pkl",
            "NANOGrav_15yr_timing_residuals.hdf5",
            "NANOGrav_15yr_pulsar_params.json",
        ]
    },
    
    # Earlier datasets for comparison
    "12p5yr": {
        "description": "NANOGrav 12.5-year dataset",
        "paper": "https://arxiv.org/abs/2009.04496",
        "zenodo": "https://zenodo.org/record/4312888",
    },
}

# ATNF pulsar catalog for Galactic distribution
ATNF_CATALOG_URL = "https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php"


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url: str, dest: Path) -> bool:
    """Download with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_nanograv_15yr():
    """
    Download NANOGrav 15-year dataset.
    
    This dataset contains the first evidence for a stochastic GW background.
    Key outputs:
    - Strain spectrum h_c(f) at nanohertz frequencies
    - Individual pulsar timing residuals
    - Spatial correlations (Hellings-Downs)
    """
    print("\n" + "=" * 60)
    print("NANOGrav 15-year Data")
    print("=" * 60)
    
    ng15_dir = PULSAR_DIR / "nanograv_15yr"
    ng15_dir.mkdir(exist_ok=True)
    
    # The actual data is on Zenodo - provide instructions
    readme_file = ng15_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme_file, 'w') as f:
        f.write("""NANOGrav 15-year Dataset
========================

Primary source: https://zenodo.org/record/8067508

Key files to download:
1. NANOGrav15yr_v1.0.0.zip - Full timing data
2. gwb_posterior_samples.pkl - GWB amplitude posteriors

Paper: Agazie et al. (2023), ApJL 951, L8
arXiv: 2306.16213

After downloading, key quantities:

1. GWB STRAIN SPECTRUM h_c(f):
   The characteristic strain at frequency f (nanohertz range)
   h_c(f) ≈ A * (f / f_yr)^α
   where A ≈ 2.4 × 10^-15, α ≈ -2/3

2. TIMING RESIDUALS:
   δt(t) for each pulsar - fluctuations from expected arrival times
   RMS residuals: 100 ns to 10 μs depending on pulsar

3. HELLINGS-DOWNS CORRELATIONS:
   Spatial correlation ρ(θ) between pulsar pairs
   Characteristic quadrupolar pattern from GW background

Python loading example:
    import pickle
    import h5py
    
    # Load GWB posteriors
    with open('gwb_posterior_samples.pkl', 'rb') as f:
        posteriors = pickle.load(f)
    
    log10_A = posteriors['log10_A_gw']  # log10 of strain amplitude
    gamma = posteriors['gamma_gw']       # spectral index
    
    # Load timing residuals
    with h5py.File('timing_residuals.hdf5', 'r') as f:
        psr_names = list(f.keys())
        for psr in psr_names:
            times = f[psr]['toas'][:]
            residuals = f[psr]['residuals'][:]
""")
    
    print(f"Instructions saved to {readme_file}")
    print("\nTo download manually:")
    print("  wget https://zenodo.org/record/8067508/files/NANOGrav15yr_v1.0.0.zip")


def download_atnf_pulsar_catalog():
    """
    Download ATNF pulsar catalog for Galactic pulsar distribution.
    
    This gives us the spatial distribution of known pulsars,
    which indicates where GW emission from binaries is concentrated.
    """
    print("\n" + "=" * 60)
    print("ATNF Pulsar Catalog")
    print("=" * 60)
    
    # Query ATNF for binary pulsars (GW sources)
    # This uses their web form interface
    
    atnf_dir = PULSAR_DIR / "atnf"
    atnf_dir.mkdir(exist_ok=True)
    
    # Create query parameters for binary pulsars
    # Fields: Name, RA, Dec, Distance, Period, Binary type
    params = {
        "Name": "Name",
        "RaJ": "RaJ",
        "DecJ": "DecJ",
        "Dist": "Dist", 
        "P0": "P0",
        "Binary": "Binary",
        "PB": "PB",  # Binary period
        "Comp": "Comp",  # Companion type
        "condition": "Binary != '*'",  # Only binaries
        "pulsar_names": "",
        "ephession": "short",
        "state": "query",
        "table_bottom.x": "50",
        "table_bottom.y": "10",
        "sort_attr": "jname",
        "sort_order": "asc",
        "output_style": "text",
    }
    
    print("Querying ATNF catalog for binary pulsars...")
    
    try:
        response = requests.post(ATNF_CATALOG_URL, data=params, timeout=60)
        
        if response.status_code == 200:
            output_file = atnf_dir / "binary_pulsars.txt"
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"  Saved to {output_file}")
        else:
            print(f"  Query failed: {response.status_code}")
            
    except Exception as e:
        print(f"  Error: {e}")
        print("  You can manually query at: https://www.atnf.csiro.au/research/pulsar/psrcat/")


def create_gw_background_model():
    """
    Create a model for the local GW energy density.
    
    The GW background comes primarily from:
    1. Supermassive BH binaries (NANOGrav band)
    2. Stellar-mass compact binaries (LIGO band)
    
    We model the local GW energy density as proportional to
    the density of potential sources (compact binaries).
    """
    print("\n" + "=" * 60)
    print("Creating GW background model")
    print("=" * 60)
    
    model_file = PULSAR_DIR / "gw_background_model.py"
    
    model_code = '''"""
GW Background Energy Density Model

This module estimates the local GW energy density ρ_GW
as a function of position, based on the expected density
of GW sources (compact binary populations).
"""

import numpy as np

# NANOGrav 15yr best-fit parameters
NANOGRAV_15YR = {
    "A_gw": 2.4e-15,        # Strain amplitude at f_yr
    "f_yr": 1 / (365.25 * 24 * 3600),  # Reference frequency (1/year in Hz)
    "gamma": -2/3,          # Spectral index for SMBH binaries
    "log10_A_err": 0.3,     # Uncertainty on log10(A)
}


def gw_strain_spectrum(f: np.ndarray, A: float = None, gamma: float = None) -> np.ndarray:
    """
    Characteristic strain h_c(f) from NANOGrav.
    
    h_c(f) = A * (f / f_yr)^gamma
    
    Parameters
    ----------
    f : array
        Frequencies in Hz
    A : float
        Strain amplitude (default: NANOGrav 15yr)
    gamma : float
        Spectral index (default: -2/3 for SMBHBs)
    
    Returns
    -------
    h_c : array
        Characteristic strain
    """
    if A is None:
        A = NANOGRAV_15YR["A_gw"]
    if gamma is None:
        gamma = NANOGRAV_15YR["gamma"]
    
    f_yr = NANOGRAV_15YR["f_yr"]
    return A * (f / f_yr)**gamma


def gw_energy_density(f: np.ndarray, h_c: np.ndarray = None) -> np.ndarray:
    """
    GW energy density Ω_GW(f) from strain spectrum.
    
    Ω_GW(f) = (2 π² / 3 H₀²) f² h_c(f)²
    
    Parameters
    ----------
    f : array
        Frequencies in Hz
    h_c : array, optional
        Characteristic strain (computed if not given)
    
    Returns
    -------
    Omega_gw : array
        Dimensionless energy density (ρ_GW / ρ_crit)
    """
    H0_si = 67.4 * 1000 / 3.086e22  # H0 in s^-1
    
    if h_c is None:
        h_c = gw_strain_spectrum(f)
    
    Omega = (2 * np.pi**2 / 3 / H0_si**2) * f**2 * h_c**2
    return Omega


def local_gw_density_from_binaries(stellar_density: float,
                                    binary_fraction: float = 0.1,
                                    compact_fraction: float = 0.01) -> float:
    """
    Estimate local GW emission from compact binary density.
    
    Very rough model: GW luminosity ∝ density of compact binaries
    
    Parameters
    ----------
    stellar_density : float
        Local stellar mass density (Msun/pc³)
    binary_fraction : float
        Fraction of stars in binaries
    compact_fraction : float
        Fraction of binaries that are compact (NS/BH)
    
    Returns
    -------
    gw_density_proxy : float
        Relative GW emission (arbitrary units)
    """
    return stellar_density * binary_fraction * compact_fraction


def quietness_from_gw(gw_density_proxy: float, 
                      threshold: float = 0.001) -> float:
    """
    Map GW density to quietness factor (0 to 1).
    
    Higher GW density = less quiet = lower quietness factor.
    
    Parameters
    ----------
    gw_density_proxy : float
        GW emission proxy
    threshold : float
        Density scale for transition
    
    Returns
    -------
    quietness : float
        0 (noisy) to 1 (quiet)
    """
    return np.exp(-gw_density_proxy / threshold)


# Test
if __name__ == "__main__":
    # Plot GW spectrum
    import matplotlib.pyplot as plt
    
    f = np.logspace(-10, -6, 100)  # nHz to μHz
    h_c = gw_strain_spectrum(f)
    omega = gw_energy_density(f, h_c)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].loglog(f * 1e9, h_c)
    axes[0].set_xlabel("Frequency (nHz)")
    axes[0].set_ylabel("h_c(f)")
    axes[0].set_title("GW Strain Spectrum")
    axes[0].axvline(1e9/365.25/24/3600, ls='--', c='gray', label='f_yr')
    
    axes[1].loglog(f * 1e9, omega)
    axes[1].set_xlabel("Frequency (nHz)")
    axes[1].set_ylabel("Ω_GW(f)")
    axes[1].set_title("GW Energy Density")
    
    plt.tight_layout()
    plt.savefig("gw_spectrum.png", dpi=150)
    print("Saved gw_spectrum.png")
'''
    
    with open(model_file, 'w') as f:
        f.write(model_code)
    
    print(f"Model saved to {model_file}")


# =============================================================================
# VERIFY
# =============================================================================

def verify_downloads():
    """Check available pulsar/GW data."""
    print("\n" + "=" * 60)
    print("Verifying pulsar timing data")
    print("=" * 60)
    
    # Check NANOGrav
    ng15 = PULSAR_DIR / "nanograv_15yr"
    if ng15.exists():
        files = list(ng15.iterdir())
        print(f"  NANOGrav 15yr: {len(files)} files/dirs")
    
    # Check ATNF
    atnf = PULSAR_DIR / "atnf"
    if atnf.exists():
        files = list(atnf.glob("*.txt"))
        print(f"  ATNF catalog: {len(files)} files")
    
    # Check model
    model = PULSAR_DIR / "gw_background_model.py"
    if model.exists():
        print(f"  GW model: {model}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Pulsar Timing / GW Background Downloader")
    print("=" * 60)
    print(f"Data directory: {PULSAR_DIR}")
    print()
    
    # NANOGrav instructions
    download_nanograv_15yr()
    
    # ATNF catalog
    download_atnf_pulsar_catalog()
    
    # Create model
    create_gw_background_model()
    
    # Verify
    verify_downloads()
