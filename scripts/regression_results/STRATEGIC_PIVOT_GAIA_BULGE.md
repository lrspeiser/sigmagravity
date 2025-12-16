# Strategic Pivot: From SPARC Tuning to Gaia Bulge Calibration

## Current Status Assessment

### What We've Built ✓
- Flow coherence model fully implemented (`--coherence=flow`)
- Bulge-specific tuning working (α_bulge=0.0, γ_bulge=0.01)
- All 8 core tests passing
- Residual analysis complete

### What the Numbers Tell Us

**Performance Reality:**
- Baseline bulge RMS: **28.93 km/s**
- Flow bulge-specific bulge RMS: **28.85 km/s** (∆ = **0.08 km/s**)
- This is a **small correction**, not a major lever

**Why It's Small:**
- Flow coherence works best with "very small parameter values" (α=0.02, γ=0.005)
- SPARC only gives 1D rotation curves → weak topology proxies
- The theory direction is right, but SPARC lacks information to compute true invariants

### What Residual Discovery Revealed

**SPARC (all points):**
- Dominant drivers: gradient proxies (`dlnVbar_dlnR`, `dlnGbar_dlnR`)
- Data says "missing variable smells like topology/derivatives"
- But SPARC only provides 1D curves → weak proxies

**SPARC high-bulge regions:**
- Pattern changes: orbital frequency and acceleration-regime variables matter more
- Exactly where you need flow information most
- But SPARC gives you least access to it

**Gaia:**
- **Vorticity is the #1 driver** (r=0.17, p=1e-32)
- This is the real signal: "coherence of a velocity field"
- Gaia gives you actual 6D phase-space structure

## The Strategic Pivot

### Problem Statement
**SPARC bulges are exactly the place where you most need true flow information, and SPARC gives you the least of it.**

### Solution: Gaia Bulge as Calibration Lab

Use Gaia bulge to learn the **covariant coherence scalar**:

\[
\mathcal{C}_{\rm cov} = \frac{\omega^2}{\omega^2 + 4\pi G\rho + \theta^2 + H_0^2}
\]

Then translate back to SPARC via proxies.

### Why This Works

1. **Gaia gives you the real invariants:**
   - ω², θ²: Compute from 6D star field (local velocity gradients)
   - ρ: Evaluate from Milky Way baryonic model (not guessed from rotation curves)

2. **SPARC can only approximate:**
   - Current: C ≈ v²/(v²+σ²) (kinematic approximation)
   - Future: Use proxies grounded in Gaia calibration

3. **The theory already points this way:**
   - Paper states covariant construction
   - Kinematic form is "nonrelativistic limit / practical reduction"
   - This is the "big lever" not yet exploited where it can be measured

## Implementation Plan

### Phase 1: Build Gaia Bulge Dataset

**Goal:** Create binned or pointwise Gaia bulge sample with:
- Mean ⟨v_φ⟩ and dispersions (σ_R, σ_φ, σ_z)
- Enough stars per bin for stable local gradients
- Full 6D phase-space information

**Steps:**
1. Select Gaia bulge stars (|z| > threshold, R < threshold, or kinematic selection)
2. Bin in (R, z) space with sufficient density
3. Compute local velocity field and gradients
4. Evaluate baryonic density model ρ_b(R,z) at star locations

### Phase 2: Compute Covariant Coherence

**Goal:** Build C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)

**Steps:**
1. Compute ω² from smoothed mean velocity field gradients
2. Compute θ² (divergence) from velocity field
3. Evaluate ρ_b(R,z) from MW baryonic model (McMillan-like)
4. Compute C_cov at each bin/point
5. Use existing A(L) and h(g_N) to get Σ

### Phase 3: Validate on Gaia Bulge

**Goal:** Ultra-accurate bulge kinematics prediction

**Metrics:**
- Mean rotation velocity prediction
- Velocity dispersion prediction
- Explicit decision on what "observable" to match

**Regression Test Design:**
- Separate "Gaia Bulge" test alongside existing tests
- Threshold: Better than baseline by significant margin (>1 km/s improvement)
- Avoid data leakage: Use baryonic model, not fitted parameters
- Report bulge-only pass/fail alongside existing suite

### Phase 4: Translate to SPARC

**Goal:** Use Gaia-calibrated coherence to improve SPARC bulges

**Steps:**
1. Identify which SPARC proxies best approximate C_cov
2. Use proxies like: Ω = V/R, |dV/dR|, dlnV/dlnR, compactness (R/L₀)
3. Revisit SPARC bulges with Gaia-grounded proxies
4. Expect larger improvements than current small corrections

## Why This Is Better Than More SPARC Tuning

**Current Tuning Summary Says:**
- "Very small parameter values work best" → SPARC only supports tiny topology correction
- Topology estimate is too proxy-ish

**Gaia Shows:**
- Topology signal is real and strong (vorticity dominates)
- This is what you'd see if theory is right but SPARC lacks information

**Combination Implies:**
- Theory direction (topology/coherence) is correct
- SPARC doesn't contain enough information to compute right invariant
- Only marginal gains possible from SPARC alone

## Concrete Next Steps

### Immediate (This Session)
1. **Design Gaia bulge regression test framework**
   - Define metrics, thresholds, data requirements
   - Specify how to avoid leakage
   - Plan reporting structure

2. **Identify Gaia bulge selection criteria**
   - Spatial cuts (R, z)
   - Kinematic cuts (if needed)
   - Minimum star density per bin

3. **Plan covariant C implementation**
   - Function signature for C_cov(omega2, rho, theta2)
   - Integration with existing predict_velocity framework
   - Testing on existing Gaia disk sample first

### Short-term (Next Sessions)
1. Build Gaia bulge dataset
2. Implement covariant coherence computation
3. Run initial validation
4. Compare to baseline and flow coherence

### Medium-term
1. Calibrate SPARC proxies from Gaia results
2. Apply to SPARC bulges
3. Expect larger improvements (>1 km/s)

## Success Criteria

**Gaia Bulge Test:**
- ✓ Computes C_cov from 6D star field
- ✓ Uses baryonic density model (not fitted)
- ✓ Predicts bulge kinematics accurately
- ✓ Shows improvement over baseline (>1 km/s)

**SPARC Translation:**
- ✓ Identifies best proxies for C_cov
- ✓ Applies to SPARC bulges
- ✓ Shows larger improvements than current small corrections

## Files to Create

1. `test_gaia_bulge.py` - New regression test for Gaia bulge
2. `compute_covariant_coherence.py` - Implementation of C_cov
3. `gaia_bulge_selection.py` - Bulge star selection and binning
4. `gaia_bulge_report.json` - Test results format

## Conclusion

**Current State:** Flow coherence works but only delivers small corrections on SPARC because SPARC lacks topology information.

**Path Forward:** Use Gaia bulge to learn the covariant coherence scalar, then translate back to SPARC. This is the shortest path to "major" bulge improvements.

**Key Insight:** Don't optimize on SPARC bulges right now. Use SPARC disk + non-galaxy tests as guardrails, but let Gaia bulge teach you the flow topology law.


