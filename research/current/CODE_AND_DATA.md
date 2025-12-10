# Code and Data for Coherence Gravity

**Reference to all code files and data sources**

---

## 1. Core Theory Code

### Galactic Dynamics

| File | Description |
|------|-------------|
| `derivations/full_regression_test.py` | Complete regression test across all validation domains |
| `derivations/stress_energy_correlator_test.py` | Current-current correlator implementation |
| `derivations/correlator_manga_validation.py` | Counter-rotation validation with MaNGA |
| `derivations/coherence_hypothesis_tests.py` | Coherence order parameter tests |

### Cosmology

| File | Description |
|------|-------------|
| `derivations/pantheon_coherence_test.py` | Pantheon+ supernova fit |
| `derivations/angular_size_test.py` | BAO angular diameter distance test |
| `cosmology/coherence_cosmology_test.py` | Preliminary cosmology comparison |
| `cosmology/coherence_time_dilation.py` | Time dilation analysis |

### Microphysics

| File | Description |
|------|-------------|
| `cosmology/fundamental_microphysics.py` | Current-current to cosmology connection |
| `cosmology/tz_solution.py` | T(z) = T₀(1+z) solution |
| `cosmology/cmb_analysis.py` | CMB analysis for coherence cosmology |
| `derivations/microphysics_investigation.py` | Microphysics exploration |

---

## 2. Data Files

### SPARC Database

**Location:** `data/sparc/`

| File | Description |
|------|-------------|
| `SPARC_Lelli2016c.mrt` | Main SPARC catalog |
| `RotationCurves/*.dat` | Individual rotation curves |
| `MassModels/*.dat` | Mass models for each galaxy |

**Source:** http://astroweb.cwru.edu/SPARC/

### MaNGA DynPop

**Location:** `data/manga_dynpop/`

| File | Description |
|------|-------------|
| `SDSSDR17_MaNGA_JAM.fits` | JAM dynamical models |

**Source:** SDSS DR17

### Counter-Rotating Galaxies

**Location:** `data/stellar_corgi/`

| File | Description |
|------|-------------|
| `bevacqua2022_counter_rotating.tsv` | Bevacqua et al. 2022 catalog |

**Source:** MNRAS 511, 139 (2022)

### Pantheon+

**Location:** `data/`

| File | Description |
|------|-------------|
| `pantheon_plus.dat` | Pantheon+ supernova compilation |

**Source:** https://pantheonplussh0es.github.io/

---

## 3. Key Functions

### Enhancement Factor

```python
def h_function(g: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Acceleration gate function.
    
    h(g) = sqrt(g†/g) × g†/(g† + g)
    
    Parameters:
        g: Newtonian acceleration (m/s²)
        alpha: Power law index (default 1.0)
    
    Returns:
        h(g): Gate function value
    """
    g_dagger = 9.6e-11  # m/s²
    return np.sqrt(g_dagger / g) * (g_dagger / (g_dagger + g))**alpha
```

### Coherence Window

```python
def W_coherence(r: np.ndarray, xi: float = 1.0) -> np.ndarray:
    """
    Spatial coherence window.
    
    W(r) = r / (ξ + r)
    
    Parameters:
        r: Radius (kpc)
        xi: Coherence scale (kpc)
    
    Returns:
        W(r): Coherence window value
    """
    return r / (xi + r)
```

### Sigma Enhancement

```python
def compute_sigma(R: np.ndarray, g_N: np.ndarray, 
                  A: float = 1.0, xi: float = 1.0) -> np.ndarray:
    """
    Compute enhancement factor Σ.
    
    Σ = 1 + A × W(R) × h(g_N)
    
    Parameters:
        R: Radius (kpc)
        g_N: Newtonian acceleration (m/s²)
        A: Amplitude parameter
        xi: Coherence scale (kpc)
    
    Returns:
        Σ: Enhancement factor
    """
    W = W_coherence(R, xi)
    h = h_function(g_N)
    return 1 + A * W * h
```

### Luminosity Distance (Coherence)

```python
def luminosity_distance_coherence(z: np.ndarray, 
                                   H0: float = 73.0,
                                   alpha: float = 1.4) -> np.ndarray:
    """
    Luminosity distance in coherence cosmology.
    
    d_L = (c/H₀) × [(1+z)^α - 1] / α
    
    Parameters:
        z: Redshift
        H0: Hubble constant (km/s/Mpc)
        alpha: Non-linearity parameter
    
    Returns:
        d_L: Luminosity distance (Mpc)
    """
    c = 299792.458  # km/s
    return (c / H0) * ((1 + z)**alpha - 1) / alpha
```

### Angular Diameter Distance

```python
def angular_diameter_distance_coherence(z: np.ndarray,
                                         H0: float = 73.0,
                                         alpha: float = 1.4,
                                         beta: float = -0.4) -> np.ndarray:
    """
    Angular diameter distance in coherence cosmology.
    
    d_A = d_L / (1+z)^(2+β)
    
    Parameters:
        z: Redshift
        H0: Hubble constant (km/s/Mpc)
        alpha: Luminosity distance parameter
        beta: Anisotropy parameter
    
    Returns:
        d_A: Angular diameter distance (Mpc)
    """
    d_L = luminosity_distance_coherence(z, H0, alpha)
    return d_L / (1 + z)**(2 + beta)
```

---

## 4. Running Tests

### Full Regression Test

```bash
cd derivations
python full_regression_test.py
```

This runs:
- SPARC rotation curve fits
- Cluster mass predictions
- Milky Way tests
- Solar System constraints
- Redshift evolution
- Coherence scale tests
- Counter-rotation tests

### Pantheon+ Test

```bash
cd derivations
python pantheon_coherence_test.py
```

Fits coherence cosmology to Pantheon+ supernovae.

### Counter-Rotation Test

```bash
cd exploratory/coherence_wavelength_test
python counter_rotation_statistical_test.py
```

Statistical comparison of f_DM in counter-rotating vs normal galaxies.

---

## 5. Parameter Values

### Galactic Parameters

| Parameter | Value | Uncertainty | Source |
|-----------|-------|-------------|--------|
| g† | 9.6 × 10⁻¹¹ m/s² | ±0.5 × 10⁻¹¹ | Derived from H₀ |
| ξ | 1.0 kpc | ±0.3 | SPARC fit |
| A (galaxies) | 1.0 | ±0.1 | SPARC fit |
| σ_c | 30 km/s | ±10 | MaNGA fit |

### Cosmological Parameters

| Parameter | Value | Uncertainty | Source |
|-----------|-------|-------------|--------|
| H₀ | 73.0 km/s/Mpc | ±1.0 | SH0ES |
| α | 1.38 | ±0.05 | Pantheon+ fit |
| β | -0.39 | ±0.10 | BAO fit |
| T₀ | 2.725 K | ±0.001 | COBE/FIRAS |

### Derived Quantities

| Quantity | Value | Formula |
|----------|-------|---------|
| g† | 9.6 × 10⁻¹¹ m/s² | cH₀/(4√π) |
| L_coh | 4.1 Gpc | c/H₀ |
| ρ_crit | 9.2 × 10⁻²⁷ kg/m³ | 3H₀²/(8πG) |

---

## 6. Dependencies

### Python Packages

```
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4
astropy >= 5.0
pandas >= 1.3
```

### Optional

```
emcee >= 3.0  # For MCMC fitting
corner >= 2.2  # For corner plots
```

### Installation

```bash
pip install numpy scipy matplotlib astropy pandas
pip install emcee corner  # optional
```

---

## 7. File Structure

```
sigmagravity/
├── README.md                    # Main project README
├── SUPPLEMENTARY_INFORMATION.md # Technical details
│
├── current/                     # Current state of theory
│   ├── PAPER.md                # Main paper draft
│   ├── OBSERVATIONAL_TESTS.md  # Test summary
│   ├── REMAINING_CHALLENGES.md # Open problems
│   ├── CODE_AND_DATA.md        # This file
│   └── derivations/
│       └── MATHEMATICAL_DERIVATIONS.md
│
├── cosmology/                   # Cosmological extension
│   ├── COHERENCE_COSMOLOGY.md  # Cosmology summary
│   ├── NEXT_STEPS.md           # Research agenda
│   ├── fundamental_microphysics.py
│   ├── tz_solution.py
│   └── cmb_analysis.py
│
├── derivations/                 # Core calculations
│   ├── full_regression_test.py
│   ├── pantheon_coherence_test.py
│   ├── angular_size_test.py
│   └── ...
│
├── exploratory/                 # Exploratory analyses
│   └── coherence_wavelength_test/
│       └── counter_rotation_statistical_test.py
│
└── data/                        # Data files
    ├── sparc/
    ├── manga_dynpop/
    ├── stellar_corgi/
    └── pantheon_plus.dat
```

---

## 8. Citation

If using this code or data, please cite:

```bibtex
@misc{coherence_gravity_2025,
  author = {Sigma Gravity Research},
  title = {Coherence Gravity: A Unified Framework for Galaxy Dynamics and Cosmology},
  year = {2025},
  url = {https://github.com/lrspeiser/sigmagravity}
}
```

And the original data sources:
- SPARC: Lelli et al. 2016, AJ 152, 157
- MaNGA: Bundy et al. 2015, ApJ 798, 7
- Pantheon+: Scolnic et al. 2022, ApJ 938, 113
- Bevacqua: Bevacqua et al. 2022, MNRAS 511, 139

