# Coherence Field Theory

A unified field theory approach to gravity that explains solar system dynamics, galaxy rotation curves, and cluster lensing without invoking dark matter.

## Theory Overview

We extend General Relativity with a "coherence" scalar field φ(x) that represents coarse-grained gravitational wave coherence:

```
G_μν = 8πG (T_μν^(std) + T_μν^(φ))
```

### Key Components

1. **Cosmological Background**: Scalar field evolution reproducing observed redshift-distance relations
2. **Galaxy Dynamics**: Field clustering provides "extra gravity" for flat rotation curves
3. **Cluster Lensing**: Field distribution matches observed lensing masses
4. **Solar System**: Screening mechanisms ensure GR compatibility in high-density regions

## Project Structure

- `theory/` - Theoretical framework and derivations
- `cosmology/` - Background evolution and expansion history
- `galaxies/` - Rotation curve modeling and SPARC data fitting
- `clusters/` - Lensing profiles and Abell cluster analysis
- `solar_system/` - Local tests and screening mechanisms
- `data_integration/` - Utilities for Gaia, SPARC, and cluster data
- `fitting/` - Parameter optimization framework
- `visualization/` - Plotting and analysis tools

## Data Sources

- **Gaia**: Solar neighborhood dynamics, wide binary tests
- **SPARC**: Galaxy rotation curves (175 galaxies)
- **Abell clusters**: Gravitational lensing data
- **Pantheon**: Supernova redshift-distance (cosmology validation)

## Goals

1. Derive scalar field potential V(φ) from GW coherence principles
2. Fit cosmological parameters to match ΛCDM expansion history
3. Reproduce galaxy rotation curves without dark matter halos
4. Match cluster lensing profiles
5. Ensure solar system tests pass (PPN parameters)

## Getting Started

```bash
pip install -r requirements.txt
python cosmology/run_background_evolution.py
python galaxies/fit_sparc_rotcurves.py
python clusters/analyze_lensing.py
```

