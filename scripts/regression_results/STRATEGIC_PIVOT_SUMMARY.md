# Strategic Pivot: Complete Summary

## Current Reality Check ✓

**What We Built:**
- Flow coherence model fully implemented and working
- Bulge-specific tuning: 28.85 km/s (0.08 km/s better than baseline)
- All 8 core tests passing

**What the Numbers Say:**
- This is a **small correction**, not a major lever
- "Very small parameter values work best" → SPARC only supports tiny topology correction
- Flow coherence behaves like a small correction, not a major new lever

## The Real Insight

**Residual Discovery Revealed:**
- **SPARC**: Gradient proxies dominant (but only 1D rotation curves → weak proxies)
- **SPARC bulges**: Need flow info most, but have least access to it
- **Gaia**: **Vorticity is #1 driver** (r=0.17, p=1e-32) → real 6D flow topology signal

**The Problem:**
SPARC bulges are exactly where you need true flow information, but SPARC gives you the least of it.

## The Solution: Gaia Bulge Calibration

**Strategy:**
1. Use Gaia bulge as calibration lab for **covariant coherence scalar**
2. Compute ω², θ², ρ from 6D star field (not approximated from 1D curves)
3. Build C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)
4. Translate back to SPARC via proxies

**Why This Works:**
- Gaia gives you actual flow invariants (ω², θ² from 6D gradients)
- ρ from baryonic model (not guessed from rotation curves)
- This is the "big lever" the theory points to but hasn't been exploited yet

## Implementation Status

### ✓ Completed
- `C_covariant_coherence()` function implemented
- Test framework designed (`test_gaia_bulge_covariant.py`)
- Strategic pivot documented
- Test design documented

### ⏳ Next Steps
1. Validate bulge selection on Gaia catalog
2. Test binning and gradient computation
3. Integrate into regression suite
4. Run initial validation

## Key Files

- `scripts/run_regression_experimental.py`: Added `C_covariant_coherence()`
- `scripts/test_gaia_bulge_covariant.py`: Test framework
- `scripts/regression_results/STRATEGIC_PIVOT_GAIA_BULGE.md`: Full strategy
- `scripts/regression_results/GAIA_BULGE_TEST_DESIGN.md`: Test design
- `scripts/regression_results/STRATEGIC_PIVOT_SUMMARY.md`: This document

## Conclusion

**Current State:** Flow coherence works but only delivers small corrections on SPARC because SPARC lacks topology information.

**Path Forward:** Use Gaia bulge to learn the covariant coherence scalar, then translate back to SPARC. This is the shortest path to "major" bulge improvements.

**Key Insight:** Don't optimize on SPARC bulges right now. Use SPARC disk + non-galaxy tests as guardrails, but let Gaia bulge teach you the flow topology law.

