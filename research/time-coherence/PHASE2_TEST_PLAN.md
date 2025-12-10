# Phase 2 Tests: Deepening the K(Ξ) Relationship

## Overview

Phase 1 established that the roughness picture works. Phase 2 tests **exactly how** K relates to exposure time Ξ and validates the mechanism at deeper levels.

---

## Test 1: Universal K(Ξ) Relation ✅ IMPLEMENTED

### Goal:
Fit a universal function K = F(Ξ) that works across MW, SPARC, and clusters.

### Hypotheses Tested:

1. **Linear**: K = a · Ξ
2. **Extra-time**: K = (a · Ξ) / (1 - b · Ξ)
3. **Power law**: K = a · Ξ^n

### Implementation: `fit_K_vs_Xi_relation.py`

- Fits on MW + SPARC
- Validates on clusters (out-of-sample)
- Reports RMS and correlation for each model
- Saves best fit to `K_vs_Xi_fit.json`

### What Success Looks Like:

- One model gives **high correlation (>0.9)** and **low RMS** for all three domains
- Same parameters work for MW, SPARC, and clusters
- This proves "extra time in field" is the **actual law**, not just a story

---

## Test 2: MW Impulse-Level Test ✅ IMPLEMENTED

### Goal:
Test if K(Ξ) relation works at the **impulse level** (g · T_orb · enhancement) for individual stars.

### Implementation: `test_mw_impulse.py`

- Loads MW star data (Gaia + GR baseline)
- Computes K_obs and Xi for each star
- Uses fitted K(Ξ) relation to predict K_pred
- Compares boost_obs = 1 + K_obs vs boost_pred = 1 + K_pred

### What Success Looks Like:

- **High correlation** (>0.9) between boost_obs and boost_pred
- **Small RMS** difference
- Shows K(Ξ) works **point-by-point**, not just globally

### Physical Meaning:

If this passes, it proves:
> The **same** exposure-time function F(Ξ) that fits global K also works when thinking in terms of impulses along individual orbits.

This is the kind of **internal consistency** that makes "extra time in field" look like a real mechanism, not just a curve-fit.

---

## Test 3: Cluster κ(R) Shape Test ✅ IMPLEMENTED

### Goal:
Test if K_rough(R) reproduces the **observed κ(R) radial shape**, not just the total mass at Einstein radius.

### Implementation: `test_cluster_kappa_shape.py`

- Computes κ_bar(R) from baryon profiles
- Computes κ_eff(R) = κ_bar(R) · (1 + K(R))
- Compares radial slopes (d log κ / d log R)
- Tests if shape matches observed lensing

### What Success Looks Like:

- κ_eff(R) has correct **radial dependence**
- Slopes match observed lensing profiles
- Not just normalization - the **geometry** is right

### Physical Meaning:

If this passes, it proves:
> Roughness gives the right **geometry** for lensing, not just the right normalization.

This shows the mechanism works at the **structural level**, not just integrated quantities.

---

## Expected Results

### If All Tests Pass:

1. **Universal K(Ξ) law** → Functional link between "extra time" and enhancement is nailed down
2. **Impulse-level consistency** → Law works along individual orbits, not just globally
3. **Correct lensing geometry** → Roughness gives right radial shape, not just mass

### This Proves:

- The roughness picture is **not just a fit** - it's a **mechanism**
- The same physics works across **all scales** (Solar System → galaxies → clusters)
- The "extra time in field" interpretation is **quantitatively validated**

---

## Files Generated

- `K_vs_Xi_fit.json` - Best-fit K(Ξ) relation
- `mw_impulse_test.json` - MW impulse-level results
- `cluster_kappa_shape_*.json` - Cluster κ(R) shape profiles

---

## Next Steps After Phase 2

If tests pass:
1. **Write theory chapter** - Explain K(Ξ) relation in first-principles terms
2. **Refine interpretation** - Use numbers to tighten "rough spacetime" story
3. **Paper integration** - Plug into Σ-Gravity paper as first-principles component

The roughness picture will have survived:
- ✅ Solar System constraints
- ✅ MW outer disk
- ✅ SPARC rotation curves
- ✅ Cluster lensing & time delays
- ✅ Universal K(Ξ) law
- ✅ Impulse-level consistency
- ✅ Correct lensing geometry

That's a **strong case** for a viable first-principles component of Σ-Gravity!

