# Gate Modeling for Σ-Gravity: First-Principles Derivation Tests

## 🎯 Purpose

This directory contains tools to **derive gate locations and parameters from first principles** rather than just fitting them. We test whether the gate structure can emerge from physical constraints and observational scales.

## 📦 What's Included

### 📄 Documentation
1. **gate_quick_reference.md** - Quick formulas & decision tree
2. **gate_mathematical_framework.md** - Complete theory & derivations
3. **THIS FILE (README.md)** - Overview & workflow

### 🐍 Python Tools
1. **gate_core.py** - Core gate functions (distance, acceleration, exponential)
2. **gate_modeling.py** - Visualize different gate behaviors
3. **gate_fitting_tool.py** - Fit parameters to rotation curve data
4. **unified_gate_viz.py** - Combined distance+acceleration visualization
5. **inverse_search.py** - Data-aided derivation (constrained model search)

### 🧪 Tests
1. **tests/test_section2_invariants.py** - Gate invariants, PPN safety, curl-free
2. **tests/test_gate_constraints.py** - Physical constraint validation

### 📊 Outputs
Results and figures go in `outputs/`

---

## 🔑 The Key Equations

### Master Formula: Unified Gate

```
G(R, g_bar) = G_distance(R) × G_acceleration(g_bar)

where:
  G_distance(R) = [1 + (R_min/R)^α_R]^(-β_R)
  
  G_acceleration(g_bar) = [1 + (g_bar/g_crit)^α_g]^(-β_g)
```

**Physical Interpretation:**
- **Distance component**: Suppresses at small R (solar system safety + inner galaxy)
- **Acceleration component**: Suppresses at high g_bar (bulges, dense regions)
- **Product**: Coherence only where BOTH conditions met

### Alternative: Exponential Gate (Bulges)

```
G_bulge(R) = [1 - exp(-(R/R_bulge)^α)]^β
```

Better when you have measured bulge scale length.

---

## 🚀 Quick Start

### Step 1: Run Invariant Tests
```bash
cd gates
python -m pytest tests/test_section2_invariants.py -v
```

This validates:
- Gates stay in [0,1]
- Monotonic/saturating behavior
- Solar System safety (K < 10⁻¹⁴ at 1 AU)
- Curl-free field preservation

### Step 2: Visualize Gate Behavior
```bash
python gate_modeling.py
```

Generates `outputs/gate_functions.png` showing how parameters affect shape.

### Step 3: Fit to Data
```python
from gate_fitting_tool import GateFitter

# Your data
R, v_obs, v_bar = load_rotation_curve()

# Fit
fitter = GateFitter(coherence_length_l0=5.0)
result = fitter.fit_to_rotation_curve(R, v_obs, v_bar, gate_type='exponential')

# Check safety
safety = fitter.check_solar_system_safety(result['gate_params'], 'exponential')
print(f"Safe? {safety['is_safe']}, G(1 AU) = {safety['G_at_1AU']:.2e}")
```

### Step 4: Run Inverse Search (First-Principles Test)
```bash
python inverse_search.py --galaxy-data ../data/sparc_sample.npz --cluster-data ../data/clusters/
```

This tests whether Burr-XII emerges as the minimal-complexity winner under physical constraints.

---

## ✅ Validation Checklist

After fitting gates, verify:

### Mathematical Constraints
- [ ] G(R) ∈ [0, 1] everywhere
- [ ] G(R → 0) → 0
- [ ] G(R → ∞) → 1
- [ ] dG/dR ≥ 0 (monotonic)

### Physics Constraints
- [ ] G(1 AU) < 10⁻¹⁵ (PPN safe)
- [ ] Axisymmetric (depends only on R)
- [ ] Curl-free structure maintained
- [ ] Compatible with coherence window C(R)

### Data Quality
- [ ] χ²_reduced ≈ 1
- [ ] Residuals random
- [ ] Out-of-sample validation passes

---

## 📊 Parameter Determination

### Observable Scales (MEASURED, not fitted)

| Parameter | How to Measure | Typical Value |
|-----------|----------------|---------------|
| R_bulge | Sérsic fit to imaging | 1-3 kpc |
| g_crit | From RAR or g† scale | ~10⁻⁹ m/s² |
| ℓ₀ | Coherence length | 5 kpc (paper value) |

### Shape Parameters (FITTED to data)

| Parameter | Range | Meaning |
|-----------|-------|---------|
| α (steepness) | 1.5-3.0 | Transition sharpness |
| β (strength) | 0.5-2.0 | Suppression strength |

### Fixed by Physics (NON-NEGOTIABLE)

| Parameter | Value | Reason |
|-----------|-------|--------|
| R_min,solar | 0.0001 kpc | Solar system safety |
| α_solar | 3-4 | Steep suppression |

---

## 🧪 Test Suite Overview

### A) Unit Tests (tests/test_section2_invariants.py)

```bash
pytest tests/ -v
```

Tests:
1. **Gate invariants** - Bounds, monotonicity, limits
2. **Newtonian/PPN safety** - K < 10⁻¹⁴ at AU scales
3. **Curl-free check** - Loop integral ≈ 0
4. **Ring kernel** - Elliptic integral match

### B) Integration with Main Validation Suite

```bash
python ../many_path_model/validation_suite.py --all --with-gates
```

### C) Gate-Specific Tests (tests/test_gate_constraints.py)

Tests enforcement of constraints C1-C5 from paper Section 2.3.

---

## 🔬 First-Principles Derivation Strategy

### Goal
Show that gate structure emerges from:
1. Physical constraints (PPN, curl-free, monotone)
2. Observable scales (R_bulge, g_crit from data)
3. Parsimony (minimal parameters)

### Method: Constrained Inverse Search

```python
# Pseudocode from inverse_search.py
candidates = ['burr_xii', 'logistic', 'gompertz', 'hill', 'mixtures']

for window_form in candidates:
    # Enforce constraints BEFORE fitting
    if not satisfies_constraints(window_form):
        continue
    
    # Fit with shared ℓ₀ across galaxies AND clusters
    params = fit_constrained(window_form, galaxy_data, cluster_data)
    
    # Score on complexity vs. fit quality
    score = {
        'waic': compute_waic(params),
        'loo': compute_loo(params),
        'n_params': count_parameters(window_form),
        'transfer_score': cross_domain_transfer(params)
    }
    
    results.append((window_form, params, score))

# Pareto front: expect Burr-XII to dominate
plot_pareto_front(results, 'outputs/inverse_search_pareto.png')
```

**Expected outcome**: Burr-XII sits on or near the fit-vs-complexity frontier.

---

## 📈 Workflow: From Theory to Validation

```
1. Define constraints (C1-C5) ──→ test_gate_constraints.py
                ↓
2. Implement candidate gates ──→ gate_core.py
                ↓
3. Enforce invariants ──────────→ test_section2_invariants.py
                ↓
4. Fit to galaxy data ─────────→ gate_fitting_tool.py
                ↓
5. Check solar system safety ──→ assert G(1 AU) < 10⁻¹⁴
                ↓
6. Cross-validate on clusters ─→ inverse_search.py
                ↓
7. Generate artifacts ─────────→ outputs/
```

---

## 🎯 Key Insights

### Why This Matters

**Referee objection:**
> "Your gates are just ad-hoc fitting functions."

**Response (with this package):**
> "No. Gates emerge from:
> 1. Hard physics constraints (PPN, curl-free) - see tests/test_section2_invariants.py
> 2. Observable scales (R_bulge from imaging, g_crit from RAR) - measured, not fitted
> 3. Parsimony under constrained search - see inverse_search.py Pareto front
> 4. Same functional form works for galaxies AND clusters - see transfer scores"

### Connection to Main Paper

Your Section 2.7 galaxy kernel:
```
K(R) = A₀ · (g†/g_bar)^p · C(R; ℓ₀,p,n_coh) · G_bulge · G_shear · G_bar
```

**This package validates:**
- G_bulge, G_shear, G_bar functional forms
- Parameter determination method
- Solar system safety of full product
- Physical constraint satisfaction

---

## 📚 Next Steps

1. **Run tests**: `pytest tests/ -v`
2. **Visualize**: `python gate_modeling.py`
3. **Fit your data**: Adapt `gate_fitting_tool.py` to your SPARC galaxies
4. **Generate Pareto front**: `python inverse_search.py` (shows Burr-XII optimality)
5. **Export artifacts** for paper:
   - `outputs/gate_functions.png`
   - `outputs/inverse_search_pareto.png`
   - `outputs/gate_fit_examples/`

---

## 🔗 Integration with Main Repository

This package is **self-contained** but integrates with:
- `many_path_model/` - Uses same coherence window C(R)
- `core/` - Can use kernel implementations
- `data/` - Accesses SPARC and cluster catalogs

**No changes to main paper needed** - this is validation infrastructure.

---

## 🤔 FAQ

**Q: Too many files?**
A: Each serves a purpose:
- `gate_core.py` - Reusable functions
- `gate_modeling.py` - Exploration
- `gate_fitting_tool.py` - Application
- `inverse_search.py` - First-principles test
- `tests/*` - Validation

**Q: How long to run?**
A:
- Unit tests: ~10 seconds
- Visualization: ~30 seconds
- Inverse search (full): ~10 minutes (parallelizable)

**Q: What if tests fail?**
A: Likely causes:
1. Parameters violate PPN → Adjust R_min
2. Not monotonic → Check α, β ranges
3. Curl-free fails → Verify K=K(R) only

---

**This package turns "these are our gates" into "these gates emerge from physics."**

