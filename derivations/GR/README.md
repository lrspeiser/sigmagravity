# Physics Discovery Engine

Discovers physical laws from **real observational data** using genetic programming and symbolic regression.

## Two Approaches

### 1. `gr_derivation.py` - Tensor Coefficient Discovery (GPU)
Rediscovers Einstein Field Equations from synthetic spacetime data using CuPy on GPU.

### 2. `discover_gravity.py` - Symbolic Regression (Real Data)
Discovers equations from actual astronomical observations without knowing the answer.

## Prerequisites

```bash
pip install numpy sympy requests
# For GPU tensor search:
pip install cupy-cuda12x
```

---

## Real Physics Discovery (`discover_gravity.py`)

This is the **non-circular** approach - feeding real measurements and letting the AI discover equations.

### Usage

```bash
# Discover all laws
python discover_gravity.py all

# Individual modes
python discover_gravity.py kepler   # T² ∝ a³
python discover_gravity.py newton   # a ∝ 1/r²
python discover_gravity.py galaxy   # Dark matter/MOND relationship
```

### Discovery Modes

**1. Kepler's Third Law**
- Input: Planetary orbital periods (T) and semi-major axes (a)
- Data: 8 planets from Mercury to Neptune
- Expected discovery: T = a^1.5

**2. Newton's Gravitational Law**
- Input: Distance from Sun (r), measured centripetal acceleration (a)
- Data: Real planetary orbital mechanics
- Expected discovery: a ∝ 1/r²

**3. Galaxy Rotation Anomaly**
- Input: Baryonic acceleration (g_baryon) vs observed acceleration (g_obs)
- Data: SPARC galaxy rotation curves (NGC2403, NGC3198, UGC128)
- Expected discovery: g_obs ≠ g_baryon (the dark matter problem!)

### Data Sources

- **Planetary data**: JPL Horizons API (real ephemerides)
- **Galaxy data**: SPARC database (real rotation curves)
- **Constants**: CODATA values for G, c, etc.

### How Symbolic Regression Works

1. **Expression Trees**: Equations represented as trees (operators + terminals)
2. **Genetic Evolution**: Population of candidate equations
3. **Fitness**: Mean squared error + parsimony penalty (Occam's razor)
4. **Operators**: +, -, *, /, ^, sqrt, log, exp, sin, cos

---

## GPU Tensor Search (`gr_derivation.py`)

This approach generates synthetic data where GR is true, then searches for the tensor equation.

### Usage

```bash
python gr_derivation.py
```

### What It Does

1. Generates 100k random spacetime metrics and curvature tensors
2. Enforces Einstein's equation: T_uv = (1/8π)(R_uv - 0.5 R g_uv)
3. AI searches for coefficients [c1, c2, c3, c4] in:
   ```
   c1*R_uv + c2*R*g_uv + c3*T_uv + c4*g_uv = 0
   ```
4. Discovers c1=1, c2=-0.5, c3=-8π, c4=0

---

## Key Differences

| Aspect | `gr_derivation.py` | `discover_gravity.py` |
|--------|-------------------|----------------------|
| Data | Synthetic (GR embedded) | Real observations |
| Search | Coefficient optimization | Symbolic equation discovery |
| Compute | GPU (CuPy) | CPU (NumPy) |
| Circular? | Yes (proves nothing new) | No (genuine discovery) |

## Extending

- Add more galaxies from full SPARC database
- Fetch live JPL Horizons data for any solar system body
- Add GW strain data for testing tensor equations
- Expand operator set for more complex physics
