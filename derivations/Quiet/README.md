# Gravitational Quietness Correlation Testing Framework

## Overview

This framework tests correlations between various "quietness" variables and gravitational anomalies (galaxy rotation curves, cluster lensing) to identify what controls gravitational coherence activation.

## Key Results

From initial analysis of 130 SPARC rotation curve points:

| Test | Result | Σ-Gravity Prediction |
|------|--------|---------------------|
| RAR exponent p | **0.755** | 0.757 (0.3% error!) |
| Cosmic web ordering | **CONFIRMED** | K(void) > K(node) |
| K(void) | 6.17 ± 5.17 | Maximum |
| K(node) | 0.78 ± 0.94 | Minimum |
| Kruskal-Wallis p | **2.02e-13** | Highly significant |

The cosmic web environment strongly predicts gravitational enhancement!
- Void galaxies: 8× more enhanced than cluster galaxies
- This supports the "quietness" hypothesis

## Variables Being Tested

1. **Metric Fluctuation Amplitude** - Spacetime variability from pulsar timing & velocity dispersions
2. **Curvature Gradients** - Spatial derivatives of gravitational potential from lensing
3. **Gravitational Wave Background** - Local GW energy density from binary populations
4. **Matter Density** - Direct density and density gradients from surveys
5. **Dynamical Timescales** - Orbital/crossing times vs coherence times
6. **Entropy Production Rate** - Phase space evolution and star formation
7. **Tidal Tensor Eigenvalues** - Cosmic web classification (void/filament/node)

## Directory Structure

```
gravitational_quietness/
├── README.md
├── requirements.txt
├── config.py                    # Data paths and parameters
├── data/                        # Downloaded data goes here
│   ├── gaia/
│   ├── sparc/
│   ├── lensing/
│   ├── pulsar/
│   ├── cosmic_web/
│   └── surveys/
├── downloaders/                 # Scripts to fetch each dataset
│   ├── download_gaia.py
│   ├── download_sparc.py
│   ├── download_lensing.py
│   ├── download_nanograv.py
│   ├── download_cosmic_web.py
│   └── download_surveys.py
├── variables/                   # Compute each quietness variable
│   ├── metric_fluctuations.py
│   ├── curvature_gradients.py
│   ├── gw_background.py
│   ├── matter_density.py
│   ├── dynamical_timescales.py
│   ├── entropy_production.py
│   └── tidal_tensor.py
├── anomalies/                   # Compute gravitational anomalies
│   ├── rotation_curves.py
│   ├── cluster_lensing.py
│   └── sigma_enhancement.py
├── correlations/                # Statistical correlation tests
│   ├── correlation_analysis.py
│   └── visualization.py
└── run_all_tests.py            # Master script
```

## Data Sources Summary

| Variable | Primary Dataset | URL/Source | Size Est. |
|----------|----------------|------------|-----------|
| Metric fluctuations | NANOGrav 15yr | nanograv.org | ~100 MB |
| Metric fluctuations | Gaia DR3 RVS | ESA Gaia Archive | ~50 GB |
| Curvature gradients | DES Y3 Shear | des.ncsa.illinois.edu | ~10 GB |
| GW background | GWTC-3 | gwosc.org | ~1 GB |
| Matter density | SDSS DR17 | sdss.org | ~5 GB |
| Dynamical times | Gaia DR3 6D | ESA Gaia Archive | ~50 GB |
| Entropy/SFR | GALEX + WISE | MAST/IRSA | ~20 GB |
| Tidal tensor | Cosmic web catalogs | Various | ~1 GB |
| Rotation curves | SPARC | astroweb.cwru.edu | ~10 MB |
| Cluster lensing | CLASH/HFF | MAST | ~5 GB |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure paths
cp config.py.example config.py
# Edit config.py with your data directory

# Download datasets (start with smaller ones)
python downloaders/download_sparc.py
python downloaders/download_cosmic_web.py

# Run individual variable tests
python variables/matter_density.py

# Run full correlation analysis
python run_all_tests.py
```

## Notes

- Start with SPARC + cosmic web data (~1 GB total) for quick initial tests
- Gaia DR3 full catalog is huge; use cone searches or pre-computed samples
- Some datasets require registration (SDSS CasJobs, ESA Gaia Archive)
