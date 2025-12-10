# Getting Started with Coherence Field Theory

## Quick Start

### 1. Installation

```bash
cd coherence-field-theory
pip install -r requirements.txt
```

### 2. Run Basic Tests

```bash
# Test cosmology module
cd cosmology
python background_evolution.py
cd ..

# Test galaxy module
cd galaxies
python rotation_curves.py
cd ..

# Test cluster module
cd clusters
python lensing_profiles.py
cd ..

# Run all tests
python run_all_tests.py
```

### 3. View Results

Generated plots will be saved in the current directories and `outputs/` folder.

## Understanding the Theory

### Core Concept

We extend General Relativity with a scalar field φ representing gravitational wave coherence:

```
G_μν = 8πG (T_μν^(matter) + T_μν^(φ))
```

The field obeys:
- Klein-Gordon equation: □φ = dV/dφ
- Potential: V(φ) = V₀ exp(-λφ) (quintessence-like)

### What This Explains

1. **Cosmology**: φ mimics dark energy, driving late-time acceleration
2. **Galaxies**: φ clusters around galaxies, producing flat rotation curves
3. **Clusters**: φ enhances lensing masses
4. **Solar System**: Screening mechanisms ensure GR compatibility

## Project Structure

```
coherence-field-theory/
├── theory/              # Theoretical framework and equations
├── cosmology/           # Background evolution, H(z), d_L(z)
├── galaxies/            # Rotation curves, SPARC fitting
├── clusters/            # Lensing profiles
├── solar_system/        # PPN tests, screening
├── data_integration/    # Data loading utilities
├── fitting/             # Multi-scale parameter optimization
├── visualization/       # Dashboard and plots
└── outputs/             # Generated figures and results
```

## Typical Workflow

### 1. Explore Theory

Read `theory/field_equations.md` for complete mathematical framework.

### 2. Test Individual Components

Each module can be run standalone:

```python
# Cosmology
from cosmology.background_evolution import CoherenceCosmology
cosmo = CoherenceCosmology(V0=1e-6, lambda_param=1.0)
results = cosmo.evolve()
cosmo.plot_density_evolution()

# Galaxy
from galaxies.rotation_curves import GalaxyRotationCurve
galaxy = GalaxyRotationCurve()
galaxy.set_baryon_profile(M_disk=1e11, R_disk=3.0)
galaxy.set_coherence_halo_simple(rho_c0=1e8, R_c=10.0)
galaxy.plot_rotation_curve()

# Cluster
from clusters.lensing_profiles import ClusterLensing
lens = ClusterLensing(z_lens=0.3, z_source=1.0)
lens.set_baryonic_profile_NFW(M200=1e15, c=4.0, r_vir=2000)
lens.set_coherence_profile_simple(rho_c0=1e8, R_c=500)
profiles = lens.compute_lensing_profile(R_array)
```

### 3. Fit Real Data

```python
from galaxies.fit_sparc import SPARCFitter
fitter = SPARCFitter()
data = fitter.load_sparc_galaxy('NGC2403')
fit_result = fitter.fit_coherence_halo(data)
fitter.plot_fit(fit_result)
```

### 4. Multi-Scale Optimization

```python
from fitting.parameter_optimization import MultiScaleFitter
fitter = MultiScaleFitter()
fitter.add_cosmology_data(z, dL_obs, dL_err)
fitter.add_galaxy_data('NGC2403', r, v_obs, v_err, M_disk, R_disk)
result = fitter.fit_maximum_likelihood(method='global')
```

### 5. Create Visualization Dashboard

```python
from visualization.dashboard import CoherenceFieldDashboard
dashboard = CoherenceFieldDashboard()
dashboard.add_cosmology_results(z, H_obs, H_model, dL_obs, dL_model)
dashboard.add_galaxy_results('NGC2403', r, v_obs, v_model, v_baryon)
dashboard.create_full_dashboard()
```

## Key Parameters

### Cosmology
- `V0`: Potential energy scale (~10⁻⁶ in H₀² units)
- `lambda_param`: Exponential slope (affects equation of state)

### Galaxies
- `rho_c0`: Coherence halo central density
- `R_c`: Coherence halo core radius (kpc)
- `M_disk`, `R_disk`: Baryonic disk parameters

### Clusters
- `M200`: NFW virial mass
- `c`: NFW concentration
- `rho_c0`: Coherence halo density
- `R_c`: Coherence halo radius

## Available Data

The project can access data from `../data/`:

- **SPARC**: `../data/sparc/` - Galaxy rotation curves
- **Gaia**: `../data/gaia/` - Wide binaries, local dynamics
- **Rotmod_LTG**: `../data/Rotmod_LTG/` - 175 galaxy rotation curves
- **Clusters**: `../data/clusters/` - Abell cluster data
- **Pantheon**: `../data/pantheon/` - Supernova distances

Use `data_integration/load_data.py` to access these datasets.

## Tips and Best Practices

### Parameter Selection

1. **Start simple**: Use exponential potential V(φ) = V₀ exp(-λφ)
2. **Match ΛCDM**: Tune V₀ so Ω_φ0 ≈ 0.7
3. **Galaxy halos**: Start with R_c ~ few kpc (disk scale)
4. **Cluster halos**: Start with R_c ~ 100-500 kpc

### Optimization

1. Use `method='global'` for initial exploration
2. Use `method='local'` for refinement
3. For MCMC, run with n_steps > 5000 after finding good starting point

### Debugging

1. Check that χ²_red ≈ 1 for good fits
2. If χ²_red >> 1: Model doesn't fit data well
3. If χ²_red << 1: Errors may be overestimated
4. Watch for unphysical parameters (negative masses, etc.)

## Next Steps

1. **Fit SPARC sample**: Process all 175 galaxies
2. **Compare with sigma gravity**: Map parameters between theories
3. **Refine potential**: Try chameleon or symmetron variants
4. **Structure formation**: Implement perturbation theory
5. **Publication**: Write up results following ROADMAP.md

## Common Issues

### Import Errors
Make sure you're running from the correct directory or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/coherence-field-theory"
```

### Data Not Found
Check that paths in `data_integration/load_data.py` point to correct locations.

### Slow Optimization
- Reduce `n_steps` in integration
- Use fewer data points initially
- Start with local optimization before global

### Poor Fits
- Check parameter bounds are reasonable
- Verify data quality (outliers, errors)
- Try different potential forms
- Consider adding screening mechanisms

## Support

For questions or issues:
1. Check `theory/field_equations.md` for mathematical details
2. Review `ROADMAP.md` for development plan
3. Examine example code in each module's `__main__` block
4. Compare with sigma gravity results in `../`

## References

Key theory concepts:
- Quintessence dark energy models
- Scalar-tensor gravity theories
- Chameleon/symmetron screening
- Parametrized Post-Newtonian (PPN) formalism
- Galaxy rotation curve phenomenology
- Cluster lensing mass reconstruction

