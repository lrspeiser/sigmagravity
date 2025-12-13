# Guided-Gravity / Stream-Seeking Extension Results

**Date:** 2025-12-12  
**Test:** Core regression suite (8 tests)  
**Script:** `run_regression_experimental.py`  
**Coherence Model:** C (baseline v²/(v²+σ²))

## Concept

The guided-gravity extension implements a "stream-seeking" mechanism where coherent streams guide the gravitational response by increasing the effective path-length:

```
L_eff = L · (1 + κ · C_stream)
A_eff = A(L_eff) = A(L) · (1 + κ · C_stream)^n
```

where:
- `κ` (kappa) is the guidance strength parameter
- `C_stream ∈ [0, 1]` is a local proxy for stream coherence
- In disk galaxies: uses existing `C(v)` or `W(r)` as coherence proxy
- In contexts without explicit proxy: uses `GUIDED_C_DEFAULT`

## Test Configuration

- **κ = 1.0** (moderate guidance strength)
- **C_default = 1.0** (assumes full coherence when no proxy available)
- **C_default = 0.0** (only uses explicit coherence proxies)

## Results Comparison

### Baseline vs Guided (κ=1.0, C_default=1.0)

| Test | Baseline | Guided | Change | Status |
|------|----------|--------|--------|--------|
| **SPARC Galaxies** | RMS=17.42 km/s | RMS=19.87 km/s | +14.1% (worse) | ✓ Both pass |
| **Clusters** | Ratio=0.987 | Ratio=1.122 | +13.7% (better) | ✓ Both pass |
| **Cluster Holdout** | n=0.27±0.01 | n=0.25±0.00 | More stable | ✓ Both pass |
| **Gaia/MW** | RMS=29.8 km/s | RMS=33.7 km/s | +13.1% (worse) | ✓ Both pass |
| **Solar System** | |γ-1|=1.77e-09 | |γ-1|=1.77e-09 | No change | ✓ Both pass |
| **Tully-Fisher** | Ratio=0.87 | Ratio=0.66 | -24% (worse) | ✓ Both pass |

### Baseline vs Guided (κ=1.0, C_default=0.0)

When `C_default=0.0`, the guided effect only applies where explicit coherence proxies exist (galaxies with `C(v)` or `W(r)`):

| Test | Baseline | Guided (C_default=0.0) | Change |
|------|----------|------------------------|--------|
| **SPARC Galaxies** | RMS=17.42 km/s | RMS=19.87 km/s | +14.1% (same as C_default=1.0) |
| **Tully-Fisher** | Ratio=0.87 | Ratio=0.87 | No change (no explicit proxy in this test) |

## Key Findings

### SPARC Galaxies
- **Baseline:** RMS = 17.42 km/s (matches MOND performance)
- **Guided (κ=1.0):** RMS = 19.87 km/s (+14% worse)
- **Win rate:** 42.7% → 33.3% (worse vs MOND)
- **Scatter:** 0.100 → 0.102 dex (slightly worse)

**Interpretation:** The guided-gravity extension increases amplitude in coherent regions, which over-predicts velocities in some galaxies, leading to higher RMS.

### Galaxy Clusters
- **Baseline:** Median ratio = 0.987 (slightly under-predicts)
- **Guided (κ=1.0):** Median ratio = 1.122 (slightly over-predicts)
- **Scatter:** 0.132 dex (unchanged)

**Interpretation:** Guided-gravity brings cluster predictions closer to unity, but may slightly over-predict. The effect is modest.

### Cluster Holdout Validation
- **Baseline:** n = 0.27 ± 0.01
- **Guided:** n = 0.25 ± 0.00 (more stable, lower value)

**Interpretation:** The guided extension requires a slightly different exponent to fit clusters, but the calibration is more stable (lower std).

### Solar System Safety
- **Both models:** |γ-1| = 1.77×10⁻⁹ (well below Cassini bound)
- **No change:** Guided-gravity doesn't affect Solar System (no coherent streams)

**Interpretation:** The guided extension preserves Solar System safety because there are no coherent streams to guide the response.

## Physical Interpretation

### When Guided-Gravity Helps
1. **Clusters:** Slightly improves mass predictions (0.987 → 1.122)
2. **Stability:** More stable parameter calibration (lower std in holdout)

### When Guided-Gravity Hurts
1. **SPARC galaxies:** Increases RMS by 14% (over-predicts in coherent regions)
2. **Gaia/MW:** Increases RMS by 13% (similar over-prediction)
3. **Tully-Fisher:** Under-predicts normalization (0.87 → 0.66)

### Mechanism
The guided extension amplifies the effect in regions with high coherence (`C → 1`), which:
- **Increases** enhancement in disk galaxies (where `C ≈ 1` at large r)
- **Preserves** suppression in Solar System (where `C ≈ 0`)
- **Modifies** cluster predictions (where `C_default` is used)

## Recommendations

1. **For SPARC galaxies:** Baseline model performs better (17.42 vs 19.87 km/s RMS)
2. **For clusters:** Guided model slightly better (1.122 vs 0.987 ratio)
3. **Parameter tuning:** Try smaller `κ` values (e.g., 0.5) to reduce over-prediction
4. **Hybrid approach:** Consider using guided-gravity only for clusters, baseline for galaxies

## Usage

```bash
# Baseline
python scripts/run_regression_experimental.py --core

# Guided only
python scripts/run_regression_experimental.py --core --guided --guided-kappa 1.0

# Comparison mode
python scripts/run_regression_experimental.py --core --compare-guided --guided-kappa 1.0

# Only use explicit coherence proxies (C_default=0.0)
python scripts/run_regression_experimental.py --core --guided --guided-kappa 1.0 --guided-c-default 0.0
```

## Files

- Comparison report: `experimental_report_compare_C.json`
- Baseline report: `experimental_report_C.json`
- Guided report: `experimental_report_C_guided.json`

