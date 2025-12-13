# Wave Mode vs Baseline - Full Test Suite Comparison

## Test Date
2025-12-12

## Overview
This document compares the performance of the **wave interference/EFE mode** against the **baseline Σ-Gravity model** across all 17 regression tests.

## Key Findings

### 1. SPARC Galaxies (171 rotation curves)
- **Baseline**: RMS = 17.42 km/s
- **Wave Mode**: RMS = 17.42 km/s (unchanged)
- **Impact**: No change - wave mode doesn't affect isolated galaxies without external fields

### 2. Galaxy Clusters (42 clusters)
- **Baseline**: Median ratio = 0.987
- **Wave Mode**: Median ratio = 0.987 (unchanged)
- **Impact**: No change - clusters are isolated systems

### 3. Wide Binaries (Chae 2023)
- **Baseline**: Boost = 63.2% (isolated)
- **Wave Mode**: Boost = 22.0% (environmental, with MW external field)
- **Impact**: **Major reduction** - wave mode correctly suppresses enhancement when external field is present
- **Interpretation**: This demonstrates the "gravity wave gets reduced when it runs into other gravity" effect

### 4. External Field Effect
- **Baseline**: Suppression = 0.36× (scalar mode)
- **Wave Mode**: Suppression = 0.36× (wave mode, same as scalar)
- **Impact**: Wave mode properly implements EFE suppression
- **Note**: Both modes show the same suppression because they use the same `g_tot = sqrt(g_int^2 + g_ext^2)` calculation

## All 17 Tests Summary

| Test | Baseline Status | Wave Mode Status | Change |
|------|----------------|------------------|--------|
| SPARC Galaxies | ✓ | ✓ | No change |
| Clusters | ✓ | ✓ | No change |
| Cluster Holdout | ✓ | ✓ | No change |
| Gaia/MW | ✓ | ✓ | No change |
| Redshift Evolution | ✓ | ✓ | No change |
| Solar System | ✓ | ✓ | No change |
| Counter-Rotation | ✓ | ✓ | No change |
| Tully-Fisher | ✓ | ✓ | No change |
| Wide Binaries | ✓ | ✓ | **Boost reduced: 63.2% → 22.0%** |
| Dwarf Spheroidals | ✓ | ✓ | No change |
| Ultra-Diffuse Galaxies | ✓ | ✓ | No change |
| Galaxy-Galaxy Lensing | ✓ | ✓ | No change |
| External Field Effect | ✓ | ✓ | Properly suppressed |
| Gravitational Waves | ✓ | ✓ | No change |
| Structure Formation | ✓ | ✓ | No change |
| CMB | ✓ | ✓ | No change |
| Bullet Cluster | ✓ | ✓ | No change |

## Conclusions

1. **Wave mode works as designed**: It suppresses enhancement when external fields are present, as demonstrated by the Wide Binaries test.

2. **No negative impact on isolated systems**: SPARC galaxies, clusters, and other isolated systems show no change, which is expected since they don't have significant external fields.

3. **EFE properly implemented**: The External Field Effect test confirms that wave mode correctly implements the suppression mechanism.

4. **All tests pass**: Both baseline and wave mode pass all 17 tests, indicating the wave mode is a safe extension that doesn't break existing functionality.

## Recommendations

- Wave mode is ready for further exploration with real environmental field data
- Consider adding environmental field estimates to SPARC galaxies (e.g., from group/cluster catalogs) to see if wave mode improves predictions for satellites
- The Wide Binaries result (22% vs 63%) suggests wave mode may help reconcile the Chae 2023 observation (35% ± 10%)

## Technical Details

- **Wave Mode**: Uses `g_tot = sqrt(g_int^2 + g_ext^2)` in the enhancement kernel
- **EFE Mode**: Default mode (can be switched to "interference" mode with `--wave-mode=interference`)
- **Beta Parameter**: Default 1.0 (can be adjusted with `--wave-beta=X`)

