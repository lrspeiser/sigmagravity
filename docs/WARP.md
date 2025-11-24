# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Σ-Gravity is a research project presenting a scale-dependent gravitational enhancement framework that reproduces galaxy rotation curves and cluster lensing without particle dark matter halos. This is a Python-based astrophysics research codebase focused on analyzing SPARC galaxy data and galaxy cluster lensing observations.

**Core Concept**: The framework introduces a multiplicative kernel `g_eff = g_bar[1+K(R)]` where K(R) vanishes in compact systems (Solar System) and rises in extended structures (galaxies, clusters).

## Development Commands

### Python Environment
```bash
# The project uses Python 3.13.5
python3 --version

# Install core dependencies
python3 -m pip install -U numpy scipy matplotlib pandas sympy markdown mpmath

# Install dependencies for main coherence field theory module
pip install -r coherence-field-theory/requirements.txt

# Install dependencies for PCA analysis module
pip install -r pca/requirements.txt

# Optional packages for external pipelines
pip install corner cupy arviz pymc
```

### Running Tests
Tests are scattered throughout the codebase without a unified test framework. Tests use direct function definitions (def test_*) rather than pytest or unittest:

```bash
# Run validation suite (comprehensive physics checks)
python3 many_path_model/validation_suite.py --all

# Quick validation checklist only
python3 many_path_model/validation_suite.py --quick

# Physics consistency tests
python3 many_path_model/validation_suite.py --physics-checks

# Run gate invariant tests (requires pytest)
cd gates && python3 -m pytest tests/test_section2_invariants.py -v

# Derivation validation (primary regression check)
python3 derivations/derivation_validation.py

# Run quickstart demo (coherence field theory)
python3 coherence-field-theory/quickstart.py

# Run all tests for coherence field theory
python3 coherence-field-theory/run_all_tests.py

# Gravity wave tests
python3 GravityWaveTest/run_all_tests.py
```

### Data Analysis Workflows

```bash
# Analyze SPARC rotation curves
python3 many_path_model/sparc_zero_shot_test.py

# Run SPARC population analysis
python3 many_path_model/sparc_zero_shot_population.py

# Cluster lensing analysis
python3 many_path_model/run_cluster_lensing_b1.py

# Parameter optimization
python3 many_path_model/parameter_optimizer.py

# Run full tuning pipeline
python3 many_path_model/run_full_tuning_pipeline.py
```

### Figure Generation

```bash
# Build paper-style PDF from Markdown
python3 scripts/make_pdf.py --md README.md --out figures/paper_formatted.pdf

# Self-contained theory figures (no external repos needed)
python3 scripts/generate_pn_bounds_plot.py --rotmod data/Rotmod_LTG/NGC2403_rotmod.dat --out figures/pn_bounds_ngc2403.png
python3 scripts/generate_theory_ring_kernel_check.py
python3 scripts/generate_theory_window_check.py

# Galaxy figures (requires many_path_model at expected path)
python3 scripts/generate_rc_gallery.py --sparc_dir data/Rotmod_LTG --master data/Rotmod_LTG/MasterSheet_SPARC.mrt --hp config/hyperparams_track2.json --out figures/rc_gallery.png
python3 scripts/generate_rar_plot.py --sparc_dir data/Rotmod_LTG --hp config/hyperparams_track2.json --out figures/rar_sparc_validation.png

# Gate function visualizations
python3 gates/gate_modeling.py

# Analyze SPARC results
python3 many_path_model/analyze_sparc_results.py
```

## Architecture Overview

### Core Mathematical Framework

**Canonical Kernel** (defined once in Section 2.4 of README.md):
```
g_eff(R) = g_bar(R) × [1 + K(R)]
```

Where the kernel structure is:
```
K(R) = A × C(R; ℓ₀, p, n_coh) × ∏ G_j
```

**Coherence Window** (Burr-XII form):
```
C(R) = 1 - [1 + (R/ℓ₀)^p]^(-n_coh)
```

### Major Components

#### 1. `many_path_model/` - Core Implementation
The heart of the Σ-Gravity model implementation:

- **path_spectrum_kernel.py**: Physics-grounded kernel implementation based on stationary-phase path accumulation. Implements azimuthal loop paths that contribute gravitational coupling beyond direct radial paths.
- **path_spectrum_kernel_track2.py**: Alternate implementation track
- **validation_suite.py**: Comprehensive 8-point validation framework including internal consistency, statistical validation, external astrophysical cross-checks, and outlier triage
- **lensing_utilities.py**: Cluster lensing calculations and convergence maps
- **cluster_data_loader.py**: Load and process galaxy cluster data (CLASH-based catalog)
- **parameter_optimizer.py**: Optimize hyperparameters {A, ℓ₀, p, n_coh}
- **sparc_*.py**: SPARC galaxy sample analysis scripts

**Key Design Principle**: NEVER use synthetic/fake data. All validation uses real SPARC rotation curves and real cluster lensing observations.

#### 2. `gates/` - Geometry Gates and Constraints
Implements morphology-dependent suppression factors:

- **gate_core.py**: Core gate functions (distance, acceleration, exponential)
- **gate_modeling.py**: Visualize gate behaviors under different parameters
- **gate_fitting_tool.py**: Fit gate parameters to rotation curve data
- **tests/test_section2_invariants.py**: Comprehensive tests for gate invariants, PPN safety, curl-free fields
- **README.md**: Complete documentation of gate modeling
- **gate_quick_reference.md**: Quick formulas and decision tree

**Gate Types**:
- Bulge gates: Suppress coherence in bulge-dominated regions
- Shear gates: Account for velocity shear effects
- Bar gates: Handle barred spiral morphologies

**Critical Safety Checks**:
- G(R) ∈ [0, 1] everywhere
- G(1 AU) < 10⁻¹⁵ (Solar System safety via PPN constraints)
- Axisymmetric and curl-free
- Monotonic: dG/dR ≥ 0

#### 3. `coherence-field-theory/` - Multi-Scale Framework
Modular framework for testing coherence field theory across cosmology, galaxies, and clusters:

**Subdirectories**:
- `cosmology/`: Background evolution and H(z) calculations
- `galaxies/`: Rotation curve fitting and SPARC analysis
- `clusters/`: Lensing profiles and convergence calculations
- `solar_system/`: PPN tests and Solar System constraints
- `multiscale/`: Combined optimization across scales
- `data_integration/`: Data loading utilities
- `visualization/`: Plotting and dashboard generation
- `examples/`: Example usage scripts
- `tests/`: Test suite

**Entry Point**: `quickstart.py` - Demonstrates basic capabilities of each module

#### 4. Data Handling

**SPARC Galaxy Data**:
- Located in `data/Rotmod_LTG/` or `data/sparc/`
- Format: `MasterSheet_SPARC.mrt` (machine-readable table) or CSV
- Individual rotation curves: `{galaxy_name}_rotmod.dat`
- 166 galaxies with 80/20 stratified split by morphology

**Cluster Data**:
- CLASH-based catalog
- 10 clusters for training
- Blind holdouts: Abell 2261, MACSJ1149.5+2223
- Inputs: θ_E^obs, z_lens, P(z_source), baryonic profiles (X-ray + BCG/ICL)

**Never Create Synthetic Data**: The codebase has explicit guards against using fake data. Always load real observations.

#### 5. Supporting Modules

- `cosmo/`: Cosmological calculations (Pantheon SN fits, redshift relations)
- `derivation/`: Theoretical derivation tests and parameter sweeps
- `derivations/`: Mathematical verification scripts
- `redshift-tests/`: Cosmological redshift validation
- `time-coherence/`: Time-domain coherence analysis
- `pca/`: Principal component analysis tools
- `scripts/`: Figure and table generators
- `GravityWaveTest/`: Milky Way and validation tests

## Key Validation Requirements

### Physics Constraints (Always Enforced)
1. **Newtonian Limit**: K < 10⁻⁴ at 0.1 kpc
2. **Solar System Safety**: Boost at 1 AU ≲ 7×10⁻¹⁴
3. **Curl-Free Fields**: Axisymmetric K=K(R) ensures conservative potential
4. **Monotonicity**: Coherence window C(R) monotonically increases from 0 to 1

### Statistical Validation (8-Point Checklist)
1. Internal consistency & invariants (Newtonian limit, energy conservation, symmetry)
2. Statistical validation (hold-out, AIC/BIC, model selection)
3. External astrophysical cross-checks (BTFR, RAR, vertical structure)
4. Outlier triage (data hygiene, predictor failure modes, surgical gates)
5. V2.3b recovery & verification
6. Path-spectrum kernel fitting with monotonic constraints
7. Population laws with shape constraints
8. Quick sanity checks (ablations, 80/20 split, BTFR/RAR plots)

### Performance Metrics
- **Galaxy RAR**: Target scatter = 0.087 dex (competitive with MOND)
- **Cluster Lensing**: 88.9% coverage within 68% PPC, 7.9% median fractional error
- **Mass Scaling**: γ = 0.09 ± 0.10 (consistent with universal coherence length)

## Critical Development Patterns

### Error Handling Philosophy
From user rules and codebase patterns:

1. **Never Use Fallbacks**: Don't hide errors with try-except fallbacks that mask root causes
2. **Verbose Logging**: Always print detailed error logs and state leading up to failures
3. **Real Data Only**: Never simulate or use placeholder data in tests
4. **Let LLM Failures Surface**: When using LLMs for generation, show full verbose output rather than catching and hiding failures

### Code Organization
1. **Single Source of Truth**: The canonical kernel K(R) is defined once in Section 2.4 of README.md. Domain-specific implementations (galaxies vs clusters) only select appropriate gates and observables, never redefine the kernel.
2. **Avoid Duplicate Functions**: When refactoring, always remove old code to prevent confusion about which function is authoritative
3. **Empirical Calibration**: Parameters {A, ℓ₀, p, n_coh} are fit to data, not derived from first principles. Appendix H shows simple theoretical predictions fail by factors of 10-2500×.

### Unit Conversions
Physical constants for proper unit conversion:
```python
KPC_TO_M = 3.0856776e19  # 1 kpc in meters
KM_TO_M = 1000.0         # 1 km in meters
```

RAR computations use SI units with inclination hygiene (30°-70° range).

## Path and Layout Assumptions

Many scripts compute `ROOT = Path(__file__).resolve().parents[3]` and then refer to `ROOT/many_path_model/` and `ROOT/projects/SigmaGravity/`. To run those scripts as-is:
- Place this repository so that the directory three levels above scripts/ contains many_path_model and projects/SigmaGravity.
- Or modify sys.path insertions and default paths in the scripts to point to your local clones of those repos.

## Configuration

- `config/hyperparams_track2.json`: Baseline hyperparameters for the Track-2 galaxy kernel
- `config/bars_override.json`: Optional SPARC bar-class overrides (e.g., UGC02953: SB)

## File Organization

```
sigmagravity/
├── README.md                    # 93KB master document with full theory
├── many_path_model/            # Core kernel implementation
├── gates/                      # Morphology gates and constraints
├── coherence-field-theory/     # Multi-scale framework
├── data/                       # SPARC galaxies, cluster observations
├── docs/                       # Documentation and status reports
├── figures/                    # Generated plots and visualizations
├── GravityWaveTest/           # Milky Way and validation tests
├── cosmo/                     # Cosmological calculations
├── derivation/                # Theoretical derivation tests
├── derivations/               # Mathematical verification
├── redshift-tests/            # Cosmological validation
├── time-coherence/            # Time-domain analysis
├── pca/                       # PCA analysis tools
├── scripts/                   # Figure and table generators
├── config/                    # Hyperparameters and overrides
└── tables/                    # Generated tables
```

## Common Development Tasks

### Adding New Validation Tests
1. Tests should go in appropriate subdirectory (many_path_model/, gates/tests/, etc.)
2. Use `def test_*()` naming convention
3. Include physics checks (Newtonian limit, curl-free, monotonicity)
4. Test with real data only - never synthetic

### Modifying the Kernel
1. The canonical kernel is defined in README.md Section 2.4
2. Implementation lives in `many_path_model/path_spectrum_kernel.py`
3. Must maintain: curl-free fields, Solar System safety, monotonicity
4. Run full validation suite after changes

### Adding New Gate Types
1. Implement in `gates/gate_core.py`
2. Add tests in `gates/tests/test_section2_invariants.py`
3. Verify: G ∈ [0,1], monotonic, PPN safe
4. Update `gates/gate_quick_reference.md`

### Working with SPARC Data
1. Load using utilities in `many_path_model/validation_suite.py` or `many_path_model/cluster_data_loader.py`
2. Parse MRT format (whitespace-separated, skip 107 header lines)
3. Load individual rotation curves from `{galaxy}_rotmod.dat` files
4. Apply inclination hygiene (30°-70°)
5. Use 80/20 stratified split by morphology

## Important Constants

**Galaxy-Scale (Fitted)**:
- ℓ₀ = 4.993 kpc (coherence length)
- A₀ = 0.591 (amplitude)
- p = 0.757 (coherence window shape)
- n_coh = 0.5 (coherence damping)
- g† = 1.20 × 10⁻¹⁰ m/s² (acceleration scale)
- β_bulge = 1.759 (bulge suppression)
- α_shear = 0.149 (shear suppression)
- γ_bar = 1.932 (bar suppression)

**Cluster-Scale**:
- ℓ₀ ~ 200 kpc (projected coherence)
- γ = 0.09 ± 0.10 (mass scaling exponent)

## Additional Resources

- **Main Theory**: README.md (comprehensive 93KB document)
- **Gate Documentation**: gates/README.md and gates/gate_quick_reference.md
- **Reproducibility**: docs/REPRODUCTION_GUIDE_COMPLETE.md
- **Status Reports**: docs/ directory contains session summaries and progress reports
- **PDF Building**: Requires Google Chrome or Microsoft Edge for headless rendering
