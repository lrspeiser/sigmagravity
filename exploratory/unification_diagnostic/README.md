# Unification Diagnostic Tests

This folder contains exploratory analyses testing the separable unification ansatz:

$$\Sigma = 1 + A(D,L) \cdot W(r) \cdot h(g)$$

## Key Insight

MOND/RAR succeeds for galaxies but fails for clusters. This suggests:
- A **pure 1-variable law** (g_obs = f(g_bar)) is incomplete
- We need a **second state variable** distinguishing "cold, thin, rotation-supported" (galaxies) from "hot, 3D, extended" (clusters)

## The Factorization

The Σ-Gravity formula factors the modification into:

1. **Acceleration law** (h(g)) — keeps RAR success
2. **Coherence/geometry law** (W(r)) — fixes clusters without ruining galaxies
3. **Amplitude law** (A(D,L)) — distinguishes 2D disks from 3D clusters

## Scripts

### `y_collapse_test.py`

Tests the factorization by computing:

$$Y \equiv \frac{\Sigma_{\rm obs} - 1}{h(g_{\rm bar})}$$

If the separable form is correct, then Y ≈ A × W(r).

**Tests:**
1. **SPARC**: Does Y vs r/ξ trace a saturating curve?
2. **Milky Way**: Same W(r) trend?
3. **Clusters**: Does Y ≈ A_cluster at r >> ξ?
4. **Continuous D**: Does A increase with bulge fraction?

**Key Results:**
- Cluster Y values: Mean = 8.56 (expected ~8.45) ✓
- Galaxy-cluster separation: Clear amplitude difference ✓
- A vs D_eff correlation: r = 0.22, p < 10⁻⁴ ✓

### `continuous_D_exploration.py`

Tests continuous D_eff formulations to eliminate the discrete galaxy/cluster switch:

$$D_{\rm eff} = \frac{\sigma^2}{\sigma^2 + v_{\rm rot}^2}$$

**Results:**
- Fitted A₀ = 1.20 (current: 1.17)
- Fitted L₀ = 0.10 kpc (current: 0.40)
- Fitted n = 0.22 (current: 0.27)
- RMS residual: 1.68

**Correlations with A_required:**
- D_bulge: r = 0.27
- D_sqrt_bulge: r = 0.30 (best)
- D_kinematics: r = -0.28

## Key Findings

1. **Cluster amplitude is correct**: Y ≈ 8.5 at r >> ξ matches A_cluster = 8.45

2. **Continuous D works**: A_required correlates with bulge fraction (r = 0.27, p < 0.001)

3. **The factorization is supported**: Separating acceleration (h) from geometry (W, A) captures the galaxy-cluster difference

4. **sqrt(f_bulge) is the best D proxy**: Among tested formulations

## Implications

The results support the view that:
- **Keep the MOND/RAR acceleration dependence** (h(g)) — that's what galaxies confirm
- **Let amplitude depend on geometry/kinematics** (A(D,L)) — that's what clusters require
- **The coherence window W(r)** provides the "near vs far" saturation

A fully continuous model would use:
$$A = A_0 \times [1 - D_{\rm eff} + D_{\rm eff} \times (L/L_0)^n]$$

with D_eff computed from kinematics or morphology, eliminating the discrete switch.

## Running the Tests

```bash
cd /path/to/sigmagravity
python exploratory/unification_diagnostic/y_collapse_test.py
python exploratory/unification_diagnostic/continuous_D_exploration.py
```

Output plots are saved to `results/`.

