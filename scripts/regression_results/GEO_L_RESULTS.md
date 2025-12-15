# Geometry-Based Path-Length (L_gal) Results

## Concept

The geometry-based path-length approach addresses the fact that **bulges are 3D orbital families** requiring bulge-size-scale path lengths, while **disks have thin vertical thickness** (L ≈ L_0).

The model computes a galaxy-level effective path length:

**L_gal = (1 - f_bulge) × L_disk + f_bulge × L_bulge**

where:
- `L_disk = L_0 = 0.4 kpc` (baseline disk scale height)
- `L_bulge = GEO_L_BULGE_MULT × R_d` (geometry scale based on disk scale length)
- `f_bulge` = galaxy-level bulge fraction (from SPARC velocity components)

This `L_gal` is then used to compute the amplitude: **A = A_0 × (L_gal/L_0)^n**

## Implementation

- Added `--lgeom` flag to enable geometry-based path-length mode
- Added `--geo-l-bulge-mult` parameter to set bulge multiplier (default: 1.0)
- Modified `predict_velocity()` to compute `L_gal` from `f_bulge` before the fixed-point loop
- Uses existing `unified_amplitude(L_gal)` to compute galaxy-level `A_base`

## SPARC Galaxy Statistics

From 171 SPARC galaxies:
- **Mean f_bulge**: 0.071 (7.1%)
- **Median f_bulge**: 0.000 (most are pure disks)
- **Max f_bulge**: 0.832 (one very bulge-dominated galaxy)
- **Galaxies with f_bulge > 0.3**: 18 (bulge-dominated)
- **Galaxies with f_bulge > 0.1**: 28

## Test Results

### Baseline vs Geometry-Based (GEO_L_BULGE_MULT = 2.0)

**After fixing the bug** (A_base was being reset in the fixed-point loop):

| Test | Baseline | Geometry-Based (α=2.0) | Change |
|------|----------|------------------------|--------|
| **SPARC Overall RMS** | 17.42 km/s | 21.75 km/s | **+24.8% (worse)** |
| **SPARC Bulge RMS** | 28.93 km/s | 51.91 km/s | **+79.4% (much worse)** |
| **SPARC Disk RMS** | 16.06 km/s | 18.20 km/s | **+13.3% (worse)** |
| **Clusters** | Ratio=0.987 | Ratio=0.987 | **No change** |

### Key Finding

**Increasing A for bulge galaxies makes predictions worse**, not better. This suggests:

1. **Bulges should have LESS enhancement, not more**
   - Bulge RMS increases from 28.93 → 51.91 km/s when A is increased
   - The baseline model already over-predicts for bulge galaxies
   - Increasing A further amplifies the problem

2. **The geometry-based approach is working, but in the wrong direction**
   - The implementation is correct (after the bug fix)
   - But the physical assumption (bulges need larger L → larger A) appears wrong
   - Bulges may need **suppressed** coherence, not enhanced

## Why It Makes Things Worse

1. **Bulge galaxies already have high RMS** (28.93 km/s vs 16.06 km/s for disks)
   - Increasing A increases the enhancement → over-prediction gets worse
   - Example: For f_bulge=0.83, R_d=6.11 kpc, α=2.0:
     - L_gal = 10.2 kpc → A = 2.81 (vs baseline A = 1.17)
     - This 2.4× increase in A significantly over-enhances bulge galaxies

2. **Disk galaxies are less affected**
   - Most have f_bulge ≈ 0, so L_gal ≈ L_0 regardless of α
   - Small increase in RMS (16.06 → 18.20 km/s) from edge cases

3. **The gap widens**
   - Bulge RMS: 28.93 → 51.91 km/s (+79%)
   - Disk RMS: 16.06 → 18.20 km/s (+13%)
   - Gap increases from 12.87 → 33.71 km/s

## Did We Solve the Bulge vs Disk Problem?

**No, the problem got worse.** The geometry-based approach **increases** the gap between:
- **Bulge RMS**: 28.93 → 51.91 km/s (much worse)
- **Disk RMS**: 16.06 → 18.20 km/s (slightly worse)

The problem worsens because:

1. **Wrong direction** - Increasing A for bulges makes over-prediction worse
2. **Bulges need suppression, not enhancement** - The baseline already over-predicts for bulge galaxies
3. **Physical assumption may be wrong** - The idea that "bulges need larger L → larger A" appears incorrect

## Potential Improvements

1. **Reverse the approach**: Use **smaller A for bulges** (e.g., `A_bulge = A_0 × (L_bulge/L_0)^(-n)` or `A_bulge = 0`)
2. **Separate treatment**: Use completely different coherence model for bulge vs disk components (like the SRC model, which sets `A_bulge = 0`)
3. **Radius-dependent suppression**: Suppress enhancement at small radii where bulges dominate
4. **Different physical model**: Bulges may need a fundamentally different coherence mechanism (or none at all)

## Suppression Approach (Opposite Direction)

After trying the opposite direction (suppressing A for bulge galaxies), we tested:

**Formula**: `L_bulge = L_0 / (1 + α × f_bulge)` where α = `GEO_L_BULGE_MULT`

This makes `L_gal < L_0` when `f_bulge > 0`, giving `A < A_0` (suppression).

### Suppression Sweep Results

| α (mult) | Overall RMS | Bulge RMS | Disk RMS | vs Baseline |
|----------|-------------|-----------|----------|-------------|
| **0.0 (baseline)** | **17.42** | **28.93** | **16.06** | - |
| 0.5 | 17.41 | 28.87 | 16.06 | -0.01 km/s |
| 1.0 | 17.41 | 28.87 | 16.07 | -0.01 km/s |
| 2.0 | 17.42 | 28.93 | 16.07 | 0.00 km/s |
| 5.0 | 17.45 | 29.13 | 16.08 | +0.03 km/s |
| 10.0 | 17.48 | 29.33 | 16.08 | +0.06 km/s |
| 20.0 | 17.50 | 29.50 | 16.09 | +0.08 km/s |

**Best result**: α = 0.5 gives 0.01 km/s improvement overall, 0.06 km/s improvement for bulge galaxies.

### Why Suppression Has Minimal Impact

1. **Most galaxies are pure disks** (median f_bulge = 0.0)
   - Suppression only affects the 18 galaxies with f_bulge > 0.3
   - Overall RMS is dominated by disk galaxies

2. **Suppression is too weak**
   - Even with α = 20.0, for f_bulge = 0.83: L_gal = 0.10 kpc → A = 0.81
   - This is only a 31% reduction in A, which may not be enough

3. **The problem may be more fundamental**
   - The gap (28.93 vs 16.06 km/s) might not be solvable by just adjusting A
   - May need a completely different coherence model for bulges (like SRC model with A_bulge = 0)

## Conclusion

The geometry-based path-length approach (`--lgeom`) is **implemented and working correctly** (after fixing the bug where `A_base` was reset in the fixed-point loop). 

**Key Findings**:
1. **Enhancement direction** (increasing A for bulges): Makes things **much worse** (RMS: 17.42 → 21.75 km/s, Bulge: 28.93 → 51.91 km/s)
2. **Suppression direction** (decreasing A for bulges): Has **minimal impact** (best: 17.42 → 17.41 km/s, Bulge: 28.93 → 28.87 km/s)

**Status**: ❌ Does not solve the bulge vs disk problem

**The gap persists**: Bulge RMS (28.93 km/s) vs Disk RMS (16.06 km/s) remains large even with suppression.

**Next Steps**: 
- Try **complete suppression** (A_bulge = 0) for high-bulge galaxies
- Consider **separate coherence model** for bulges (like SRC model)
- Investigate if the problem is in the **coherence calculation** itself, not just the amplitude

