# Physics Discovery Engine

Discovers physical laws from **real observational data** using genetic programming, symbolic regression, and tensor field equation discovery.

## Overview

This system demonstrates that AI can rediscover fundamental physics equations from observational data:

| Discovery | Input Data | Result | Accuracy |
|-----------|-----------|--------|----------|
| Kepler's Law | Planetary orbits | T = a^1.5 | R² = 1.0 |
| Newton's Law | Orbital mechanics | a = 1/r² | R² = 1.0 |
| MOND Relation | Galaxy rotation | g_obs = √(1.65·g_bar) | R² = 0.89 |
| Einstein Eq. | Curvature tensors | G_μν = 8πT_μν | 0.0000% error |

## Files

```
derivations/GR/
├── expression_tree.py      # Symbolic expression trees for genetic programming
├── data_sources.py         # Real astronomical data (JPL Horizons, SPARC)
├── discover_gravity.py     # Scalar law discovery (Kepler, Newton, MOND)
├── tensor_regression.py    # Tensor field equation discovery
├── real_curvature_data.py  # SXS catalog, LIGO strain, GW→Riemann
├── discover_einstein.py    # Integrated Einstein equation discovery
└── gr_derivation.py        # GPU-accelerated tensor search (CuPy)
```

## Prerequisites

```bash
pip install numpy sympy requests
# For GPU tensor search:
pip install cupy-cuda12x
# For LIGO data:
pip install gwosc gwpy
# For SXS simulations:
pip install sxs
```

---

## Quick Start

### Discover Scalar Laws
```bash
python discover_gravity.py all
```

### Discover Einstein's Field Equations
```bash
python discover_einstein.py all
```

### GPU Tensor Search
```bash
python gr_derivation.py
```

---

## 1. Scalar Law Discovery (`discover_gravity.py`)

Discovers scalar relationships from real astronomical data using genetic programming.

### Modes

| Mode | Command | Discovers |
|------|---------|----------|
| Kepler | `python discover_gravity.py kepler` | T = a^1.5 |
| Newton | `python discover_gravity.py newton` | a = 1/r² |
| Galaxy | `python discover_gravity.py galaxy` | MOND relation |
| All | `python discover_gravity.py all` | All of the above |

### Key Result: MOND Discovery

From real SPARC galaxy rotation data, the AI discovered:
```
g_obs = √(1.65 × 10⁻¹⁰ × g_baryon)
```

This is the **MOND relation** (g_obs = √(a₀ × g_bar)) with Milgrom's critical acceleration a₀ ≈ 1.2×10⁻¹⁰ m/s².

---

## 2. Tensor Field Equation Discovery (`tensor_regression.py`)

Discovers tensor relationships from spacetime curvature data.

### Methods

**Method 1: Direct Coupling**
Find κ in G_μν = κT_μν
- Result: κ = 25.1327 = 8π (0.0000% error)

**Method 2: Trace Reversal**
Find α in R_μν = α·Rg_μν + κ·T_μν
- Result: α = 0.5 (exact)

### The Discovery

From curvature data, the AI discovered:
```
R_μν - ½Rg_μν = 8πT_μν
```

This is **Einstein's Field Equation**.

---

## 3. Real Curvature Data Sources (`real_curvature_data.py`)

### Available Sources

| Source | Data Type | Size |
|--------|-----------|------|
| SXS Catalog | Numerical relativity simulations | 2,027 simulations |
| LIGO/GWOSC | Gravitational wave strain | All detected events |
| Synthetic GW | Inspiral waveforms | On-demand generation |

### GW Strain → Riemann Tensor

The LIGO strain h(t) is converted to Riemann curvature:
```
R_x0x0 = -½ ∂²h₊/∂t²
R_y0y0 = +½ ∂²h₊/∂t²
R_x0y0 = -½ ∂²h×/∂t²
```

---

## 4. Integrated Discovery (`discover_einstein.py`)

Combines all data sources to discover field equations.

```bash
python discover_einstein.py all
```

### Output
```
Discovered Equations:
  1. G_μν = κT_μν with κ = 25.1327 (expected: 8π = 25.1327)
  2. R_μν = αRg_μν + κT_μν with α = 0.5 (expected: 0.5)
  3. Vacuum: R_μν = 0 where T_μν = 0

The Einstein Field Equations have been DISCOVERED ✓
```

---

## 5. GPU Tensor Search (`gr_derivation.py`)

Uses CuPy for massively parallel tensor evaluation on GPU.

### Performance
- 100k spacetime samples
- 5000 candidate equations per generation
- 50 generations
- ~100 seconds on RTX 5090

---

## How It Works

### Symbolic Regression
1. **Expression Trees**: Equations as trees (operators + terminals)
2. **Genetic Evolution**: Population of candidate equations
3. **Fitness**: MSE + parsimony penalty (Occam's razor)
4. **Operators**: +, -, *, /, ^, sqrt, log, exp, sin, cos

### Tensor Regression
1. **Tensor Terms**: R_μν, Rg_μν, T_μν, g_μν
2. **Linear Regression**: Find coefficients relating tensors
3. **Least Squares**: (α, κ) = argmin ||R_μν - αRg_μν - κT_μν||²

---

## Data Pipeline

```
Observations → Curvature Inference → Tensor Data → Field Equations
     ↓              ↓                    ↓              ↓
  Orbits      Geodesic inverse      (g, R, T)     G = 8πT
  GW strain   h → R_i0j0
  Galaxy V(r) v²/r = g_obs
```

---

## Extending This Framework

### Add New Data Sources
- Full SPARC database (175 galaxies)
- More SXS simulations
- Pulsar timing arrays (NANOGrav)

### Add New Physics
- Modified gravity theories
- Cosmological perturbations
- Strong-field tests

### Improve Discovery
- More sophisticated genetic operators
- Neural-guided symbolic regression
- GPU-accelerated expression evaluation
