"""
Configuration for gravitational quietness correlation tests.
Edit paths to match your local setup.
"""

import os
from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================

# Root directory for all data
DATA_ROOT = Path(os.environ.get("GRAV_DATA_ROOT", "./data"))

# Create subdirectories
GAIA_DIR = DATA_ROOT / "gaia"
SPARC_DIR = DATA_ROOT / "sparc"
LENSING_DIR = DATA_ROOT / "lensing"
PULSAR_DIR = DATA_ROOT / "pulsar"
COSMIC_WEB_DIR = DATA_ROOT / "cosmic_web"
SURVEYS_DIR = DATA_ROOT / "surveys"
OUTPUT_DIR = DATA_ROOT / "outputs"

# Ensure directories exist
for d in [GAIA_DIR, SPARC_DIR, LENSING_DIR, PULSAR_DIR, COSMIC_WEB_DIR, SURVEYS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA SOURCE URLs
# =============================================================================

DATA_URLS = {
    # SPARC galaxy rotation curves
    "sparc_table": "http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt",
    "sparc_rotcurves": "http://astroweb.cwru.edu/SPARC/",  # Individual files
    
    # NANOGrav pulsar timing
    "nanograv_15yr": "https://data.nanograv.org/static/data/15yr/",
    
    # LIGO/Virgo/KAGRA gravitational wave catalog
    "gwtc3_catalog": "https://gwosc.org/eventapi/json/GWTC-3-confident/",
    "gwtc3_posteriors": "https://zenodo.org/record/5546663",
    
    # DES weak lensing (requires authentication)
    "des_y3_shear": "https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs",
    
    # SDSS spectroscopic catalog
    "sdss_dr17_specobj": "https://data.sdss.org/sas/dr17/sdss/spectro/",
    
    # 2MASS Redshift Survey
    "2mrs_catalog": "https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/ApJS/199/26",
    
    # Cosmic web classifications
    "cosmicflows_catalog": "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/AJ/152/50",
    "disperse_void_catalog": "https://www.astro.rug.nl/~weygaert/tim1publication/cosmicweb/",
    
    # GALEX UV (star formation)
    "galex_catalog": "https://galex.stsci.edu/GR6/",
    
    # Gaia DR3 (use TAP queries, not bulk download)
    "gaia_tap_endpoint": "https://gea.esac.esa.int/tap-server/tap",
    
    # CLASH cluster lensing
    "clash_catalogs": "https://archive.stsci.edu/prepds/clash/",
}

# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

# Gravitational constant in (km/s)^2 * kpc / Msun
G_GRAV = 4.302e-6

# Speed of light in km/s
C_LIGHT = 299792.458

# Hubble constant (km/s/Mpc) - Planck 2018
H0 = 67.4

# Critical density (Msun/Mpc^3)
RHO_CRIT = 2.775e11 * (H0/100)**2

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Radial bins for rotation curve analysis
R_BINS_KPC = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

# Velocity dispersion bins (km/s)
SIGMA_BINS = [10, 20, 50, 100, 150, 200, 300, 500]

# Density contrast bins (rho/rho_mean)
DELTA_BINS = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]

# Dynamical time bins (Gyr)
T_DYN_BINS = [0.01, 0.1, 0.5, 1, 2, 5, 10]

# Correlation significance threshold
CORRELATION_THRESHOLD = 0.05  # p-value

# =============================================================================
# GAIA QUERY PARAMETERS
# =============================================================================

# Limit Gaia queries to manageable sizes
GAIA_MAX_ROWS = 1_000_000
GAIA_PARALLAX_MIN = 0.1  # mas (limits to ~10 kpc)
GAIA_RUWE_MAX = 1.4  # Quality cut

# Sample regions for Gaia (galactic coordinates)
GAIA_SAMPLE_REGIONS = [
    {"name": "solar_neighborhood", "l_min": 0, "l_max": 360, "b_min": -30, "b_max": 30, "dist_max": 1},
    {"name": "halo_north", "l_min": 0, "l_max": 360, "b_min": 60, "b_max": 90, "dist_max": 10},
    {"name": "halo_south", "l_min": 0, "l_max": 360, "b_min": -90, "b_max": -60, "dist_max": 10},
    {"name": "anticenter", "l_min": 150, "l_max": 210, "b_min": -20, "b_max": 20, "dist_max": 15},
]

# =============================================================================
# SIGMA-GRAVITY MODEL PARAMETERS
# =============================================================================

# Your Σ-Gravity enhancement function parameters (adjust as needed)
SIGMA_PARAMS = {
    "sigma_0": 1.0,        # Base enhancement
    "r_scale": 5.0,        # Scale radius (kpc)
    "n_power": 1.5,        # Power law index
    "rho_threshold": 0.1,  # Density threshold (Msun/pc^3)
}
